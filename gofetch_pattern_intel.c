/*
 * gofetch_pattern_intel.c -- bare-bones Intel port of the GoFetch (Apple DMP)
 * activation pattern.
 *
 * Apple's DMP dereferences a value that merely LOOKS like a pointer sitting in a
 * loaded cache line, without the program ever using it as an address. This tests
 * whether Intel does the same. Per trial:
 *   aop[0] = &buffer[target]   store a pointer in a line
 *   flush aop[0] and the target
 *   load aop[0]                bring the pointer VALUE into cache -- never deref
 *   time buffer[target]        did it get prefetched anyway?
 *
 * Targets are scattered across a large buffer so only a pointer-chaser (not a
 * stride/adjacent-line prefetcher) could reach them.
 *
 * xor=0: canonical pointer (a GoFetch-style DMP would follow it -> target fast)
 * xor=1: pointer ^ KEY, non-canonical (control -> target stays slow)
 * Run both and compare the average: fast-then-slow => Apple-style activation.
 *
 * Linux + x86 only.  Run: ./gofetch_pattern_intel.out <cpu> <xor 0|1>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <sys/mman.h>
#include <x86intrin.h>
#include "util/util.h"

#define TRIALS    2000
#define XOR_KEY   0xA5A5A5A5A5A5A5A5ULL
#define BUF_LINES (1UL << 20)        /* 64 MB; targets scattered across it */

static uint64_t aop[8] __attribute__((aligned(64)));   /* one line; aop[0] = pointer */

/* Raw latency of one load of *addr; the load goes into __trash (whose address is
 * fed to the rdtscp calls) so it is pinned between the two timestamps. */
static inline uint64_t time_load(uint64_t *addr, uint64_t __trash)
{
    uint64_t t0, t1;
    _mm_lfence();
    t0 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();
    __trash = *addr;
    _mm_lfence();
    t1 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();
    return t1 - t0;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "Usage: %s <cpu> <xor 0|1>\n", argv[0]); return 1; }
    int cpu = atoi(argv[1]);
    int xor = atoi(argv[2]);
    uint64_t key = xor ? XOR_KEY : 0;
    uint64_t trash = 0;

    srand((unsigned)time(NULL));
    pin_cpu(cpu);

    uint64_t *buf = mmap(NULL, sizeof(uint64_t) * BUF_LINES * 8, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(buf != MAP_FAILED);
    /* WRITE one word per line: forces a distinct physical page per virtual page
     * (an unwritten anon mapping aliases the shared zero page) and gives each
     * target real content to point at. */
    for (uint64_t i = 0; i < BUF_LINES; i++) buf[i * 8] = i + 1;

    uint64_t ptr_sum = 0, aop_sum = 0;
    for (int i = 0; i < TRIALS; i++) {
        uint64_t line = (((uint64_t)rand() << 31) ^ (uint64_t)rand()) % BUF_LINES;
        uint64_t *target = &buf[line * 8];

        aop[0] = (uint64_t)target ^ key;      /* store the (masked) pointer */

        /* warm the target page's TLB via a far line in the same 4 KB page (32
         * lines from the target, so adjacent-line prefetch can't reach it), then
         * evict that line: a real DMP prefetch needs a translation, but the
         * target line itself must stay cold */
        uint64_t warm_line = (line & ~63UL) | ((line + 32) & 63UL);
        volatile uint64_t *warm = (volatile uint64_t *)&buf[warm_line * 8];
        (void)*warm;
        _mm_clflush((void *)warm);

        _mm_clflush(aop);
        _mm_clflush(target);
        _mm_mfence();

        aop_sum += time_load(aop, trash);     /* trigger: load pointer value (miss), NO deref */
        trash = c_sleep(200, trash);          /* let a would-be DMP run */
        ptr_sum += time_load(target, trash);  /* time the pointee */
    }

    /* pointee: low at xor=0 & high at xor=1 => GoFetch-style DMP.
     * aop: should be a MISS (high) both ways -- confirms eviction; if it's low,
     * the aop line wasn't evicted and a pointee "hit" would be meaningless. */
    printf("xor=%d: pointee avg=%.1f raw cyc, aop avg=%.1f raw cyc  (trials=%d)\n",
           xor, (double)ptr_sum / TRIALS, (double)aop_sum / TRIALS, TRIALS);
    return 0;
}

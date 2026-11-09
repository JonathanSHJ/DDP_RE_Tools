/*
 * DMP_chase_aop.c -- Array-Of-Pointers DDP prefetch profiler. (Windows port)
 *
 * Build an AoP whose cells point to scattered targets. Each trial:
 *   1. flush every aop cell AND the data each cell points to.
 *   2. walk the whole array: for each i, load aop[i] and un-mask it (xor key,
 *      in asm -- access_aop), then time the deref of the real target with the
 *      shared time_access() (same net-cycle timer as Peek-a-Walk). The timed
 *      load is the first architectural touch of that target, so it is fast only
 *      if the DDP prefetched it while the walk was still at an earlier index
 *      (run-ahead). Per node the CPU does load-cell -> deref-target, which is
 *      the pointer chase the DDP trains on.
 *   3. store the per-index cycle counts in measured_time[].
 * Accumulate measured_time[] across TRIALS and report the average latency of the
 * one index of interest (<dist>). All indices are kept in acc[]; only <dist> is
 * printed for now.
 *
 * xor : 0 -> aop holds canonical pointers (DDP chases -> target prefetched -> fast)
 *       1 -> aop[i] holds (target ^ key): the in-memory value is non-canonical,
 *            so the DDP refuses to chase; software un-masks it to deref.
 *            (expect slow at <dist>)
 *
 * Windows + x86 only (MinGW). No admin for the probe itself.
 *   Run: DMP_chase_aop.exe <cpu> <aop_len> <dist> <xor 0|1>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <assert.h>
#include <windows.h>
#include <x86intrin.h>
#include "util/util.h"

#define TRIALS    2000
#define XOR_KEY   0xA5A5A5A5A5A5A5A5ULL

#define BUF_ADDR  0x666600000000UL    /* scattered data buffer  */
#define AOP_ADDR  0x666700000000UL    /* 2 MB-aligned AOP        */
#define BUF_LINES (1UL << 20)         /* # cache lines in buffer (target space)     */

/* Load the masked pointer that lives in aop[i] and XOR-unmask it in assembly,
 * returning the real target address. The `xor` always executes; key==0 makes it
 * a no-op, so the code path is identical whether masking is on or off. The value
 * the DDP inspects is the masked cell content loaded here; the deref/timing is
 * done by the caller via time_access(). */
__attribute__((noinline))
static uint64_t access_aop(uint64_t **aop, uint64_t i, uint64_t key)
{
    uint64_t real;
    asm volatile(
        "mov (%1, %2, 8), %0\n\t"   // real = aop[i]  (masked pointer)
        "xor %3, %0\n\t"            // un-mask -> real target address
        : "=&r" (real)
        : "r" (aop), "r" (i), "r" (key)
        : "memory"
    );
    return real;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <cpu> <aop_len> <dist> <xor 0|1>\n", argv[0]);
        return 1;
    }
    int      cpu     = atoi(argv[1]);
    uint64_t aop_len = strtoull(argv[2], NULL, 0);
    uint64_t dist    = strtoull(argv[3], NULL, 0);
    int      xor     = atoi(argv[4]);
    if (aop_len < 1 || dist >= aop_len) {
        fprintf(stderr, "need aop_len>=1 and dist<aop_len\n"); return 1;
    }
    uint64_t key = xor ? XOR_KEY : 0;

    uint64_t __trash = 0;
    srand((unsigned)time(NULL) ^ (unsigned)GetCurrentProcessId());
    pin_cpu(cpu);
    __trash = c_sleep(3000, __trash);
    _mm_mfence();

    /* scattered data buffer: one live word per cache line, targets randomly
     * spread over ~BUF_LINES lines so only a pointer-chasing DDP can reach them */
    uint64_t *buffer = (uint64_t *) VirtualAlloc((void *)BUF_ADDR,
                                                 sizeof(uint64_t) * BUF_LINES * 8,
                                                 MEM_COMMIT | MEM_RESERVE,
                                                 PAGE_READWRITE);
    assert(buffer != NULL);
    for (uint64_t i = 0; i < BUF_LINES * 8; i += 8) buffer[i] = rand();

    /* 2 MB-aligned array of masked pointers */
    uint64_t **aop = (uint64_t **) VirtualAlloc((void *)AOP_ADDR,
                                                sizeof(uint64_t *) * aop_len + MB(2),
                                                MEM_COMMIT | MEM_RESERVE,
                                                PAGE_READWRITE);
    assert(aop != NULL);
    assert((uint64_t)aop % MB(2) == 0);     /* align to 2 MB */
    for (uint64_t i = 0; i < aop_len; i++)
        aop[i] = (uint64_t *)((uintptr_t)&buffer[(rand() % BUF_LINES) * 8] ^ key);
    _mm_mfence();

    uint64_t *mt  = malloc(sizeof(uint64_t) * aop_len);   /* this-trial per-index times */
    uint64_t *acc = calloc(aop_len, sizeof(uint64_t));    /* accumulated across trials  */
    assert(mt && acc);

    fprintf(stderr, "# aop_len=%" PRIu64 " dist=%" PRIu64 " xor=%d key=0x%" PRIx64 " cpu=%d trials=%d\n",
            aop_len, dist, xor, key, cpu, TRIALS);

    /* ---- experiment: flush -> walk+time whole array -> accumulate, TRIALS times ---- */
    for (int t = 0; t < TRIALS; t++) {
        /* flush every aop cell AND the data it points to */
        for (uint64_t i = 0; i < aop_len; i++) {
            uint64_t real = (uint64_t)aop[i] ^ key;
            __trash = clflush((uint64_t *)(aop + i), __trash);
            __trash = clflush((uint64_t *)real, __trash);
        }
        _mm_mfence();

        /* walk the array: un-mask each cell (asm) and time its deref (net cyc) */
        for (uint64_t i = 0; i < aop_len; i++) {
            uint64_t real = access_aop(aop, i, key);
            mt[i] = time_access((uint64_t *)real, __trash);
        }

        for (uint64_t i = 0; i < aop_len; i++) acc[i] += mt[i];
    }

    double avg = (double)acc[dist] / TRIALS;
    printf("aop_len=%" PRIu64 " dist=%" PRIu64 " xor=%d: idx[%" PRIu64 "] avg=%.1f net cyc [%" PRIu64 "]\n",
           aop_len, dist, xor, dist, avg, __trash & MSB_MASK);

    free(mt);
    free(acc);
    return 0;
}

/*
 * measure_l1_cache_time.c
 *
 * Very simple L1-hit vs DRAM/miss access-time probe.
 *
 * Pins to a user-chosen logical CPU, then repeatedly times a single load:
 *   - L1 hit : touch the line first, then time it   -> should be fast
 *   - miss   : clflush the line, then time it        -> should be slow (DRAM)
 * Reports min / median / mean (net cycles) for each.
 *
 * Linux + x86 only. No sudo required; sudo only adds the optional nice(-20)
 * boost in pin_cpu, which just reduces timing noise.
 *   Build: make
 *   Run:   ./measure_l1_cache_time.out <logical_cpu_id>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include "util/util.h"

#define TRIALS 10000

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void report(const char *label, uint64_t *v, int n)
{
    qsort(v, n, sizeof(uint64_t), cmp_u64);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += v[i];
    printf("%-6s  min=%4lu  median=%4lu  mean=%6.1f  (net cyc)\n",
           label, v[0], v[n / 2], sum / n);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <logical_cpu_id>\n", argv[0]);
        return 1;
    }
    int cpu = atoi(argv[1]);
    pin_cpu(cpu);

    /* one cache-line-aligned qword to probe */
    uint64_t *target;
    if (posix_memalign((void **)&target, 64, 64) != 0) {
        perror("posix_memalign");
        return 1;
    }
    *target = 0xdeadbeef;

    uint64_t *hit  = malloc(sizeof(uint64_t) * TRIALS);
    uint64_t *miss = malloc(sizeof(uint64_t) * TRIALS);

    /* __trash is the dependency arg time_access needs; keep it MSB-clear so the
     * util's `| (__trash & MSB_MASK)` never pollutes the returned cycle counts. */
    uint64_t __trash = 0;

    /* warmup */
    for (int i = 0; i < 1000; i++) (void)time_access(target, __trash);

    for (int i = 0; i < TRIALS; i++) {
        /* L1 hit: make sure it's cached, then time */
        (void)*target;
        _mm_lfence();
        hit[i] = time_access(target, __trash);

        /* miss: flush to DRAM, then time */
        _mm_clflush(target);
        _mm_mfence();
        miss[i] = time_access(target, __trash);
    }

    printf("%d trials\n", TRIALS);
    report("L1",   hit,  TRIALS);
    report("MISS", miss, TRIALS);

    free(hit);
    free(miss);
    free(target);
    return 0;
}

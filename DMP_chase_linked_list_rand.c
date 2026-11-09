/*
 * DMP_chase_linked_list_rand.c
 *
 * DMP pointer-chase over a singly linked list, RANDOMIZED + SCATTERED layout:
 * the N nodes are each placed on a distinct random cache line inside a large
 * pool (so the node set is SPARSE, like chase_aop's target buffer) and linked
 * into one random N-cycle. A stride/stream/adjacent-line/next-page prefetcher
 * cannot reach the next node -- only a pointer-chasing DMP can.
 *
 * Structured like DMP_chase_aop.c, different workload. Each trial:
 *   1. flush every node from cache (all N nodes cold; one node per 64B line).
 *   2. chase the whole cycle from `start`. For each hop k, time_access_next()
 *      times ONLY the load of the current node's `next` field (offset 0) -- how
 *      fast next comes back -- records it in measured_time[k], then un-masks
 *      that value and advances to the real next node. The timed load is the
 *      first architectural touch of the node, so it is fast only if the DMP
 *      prefetched it while the chase was still at an earlier node (run-ahead).
 *   3. accumulate measured_time[] across TRIALS.
 * Report the average latency of one hop of interest (<dest>).
 *
 * xor : 0 -> canonical next-pointers (DMP can chase them -> prefetched -> fast)
 *       1 -> next ^ XOR_KEY (non-canonical bytes; DMP refuses -> cold miss -> slow)
 * The value the DMP inspects is the masked next-pointer loaded in the timed
 * window; time_access_next un-masks it AFTER timing, so the walk stays correct.
 *
 * The timer is the linked-list twin of the ported util time_access() (rdtscp
 * bracket minus a bare rdtscp/lfence noise window = net cycles), with the net
 * floored at 0 so a fast hit can't underflow into a huge value.
 *
 * Linux + x86 only. No sudo required.
 *   Build: make
 *   Run:   ./DMP_chase_linked_list_rand.out <cpu> <num_nodes> <dest> <xor 0|1>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <x86intrin.h>
#include "util/util.h"

#define TRIALS     1000        /* full-cycle chases recorded                 */
#define WARMUP     50          /* warmup chases (train the DMP), dropped     */
#define XOR_KEY    0xA5A5A5A5A5A5A5A5ULL   /* flips high bits -> non-canonical */
#define POOL_LINES (1UL << 20) /* 64 MB pool of 64B lines; nodes placed sparsely */

/* Sattolo's algorithm: a single N-cycle (not just any permutation). */
static void build_cycle(uint64_t *perm, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++) perm[i] = i;
    for (uint64_t i = N - 1; i > 0; i--) {
        uint64_t j = (((uint64_t)rand() << 31) ^ (uint64_t)rand()) % i;
        uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}

/* Flush every node line so the next chase starts fully cold. */
static void flush_nodes(uint64_t **addr, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
        _mm_clflush(addr[i]);
    _mm_mfence();
}

/* Linked-list twin of the ported util time_access(): times ONLY the load of the
 * current node's `next` field (offset 0) -- how fast next comes back -- then,
 * OUTSIDE the timed window, un-masks it (xor key) and advances *pnode to the
 * real next node. Same net-cycle scheme as time_access (mfence/lfence rdtscp
 * bracket minus a bare noise window), but the net is floored at 0: a fast hit
 * can have load-time < noise-window, which would otherwise wrap to ~2^64.
 *
 * Like util, the timed load goes into __trash -- the same variable whose address
 * is handed to the surrounding rdtscp calls -- so the load is PINNED between the
 * two timestamps by a real memory dependency (load -> __trash -> rdtscp(&__trash))
 * rather than trusting the compiler not to schedule it out of the window. The
 * loaded value is captured into next_masked before the second rdtscp overwrites
 * __trash; the un-mask happens after timing. The value the DMP inspects is the
 * masked next-pointer read in the timed window. */
static uint64_t time_access_next(uint64_t **pnode, uint64_t key, uint64_t __trash)
{
    uint64_t T0, T1;
    uint64_t *node = *pnode;
    uint64_t next_masked;

    // enforce ordering
    _mm_mfence();
    _mm_lfence();

    // time ONLY the next-pointer load (into __trash, as in util time_access)
    T0 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();
    /* asm volatile: a plain C load here is dead code (rdtscp overwrites
     * __trash and, in the discarded warmup, curr is never used), so clang -O2
     * deletes it and the window times nothing. asm volatile keeps the load
     * and keeps it between the two rdtscp's. */
    asm volatile("mov (%1), %0" : "=r"(next_masked) : "r"(node) : "memory");
    __trash = next_masked;
    _mm_lfence();
    T1 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();

    // un-mask and advance (outside the timed window; keeps next_masked live)
    *pnode = (uint64_t *)(next_masked ^ key);

    // raw cycles for next to come back (no noise subtraction)
    return T1 - T0;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <cpu> <num_nodes> <dest> <xor 0|1>\n", argv[0]);
        return 1;
    }
    int      cpu  = atoi(argv[1]);
    uint64_t N    = strtoull(argv[2], NULL, 0);
    uint64_t dest = strtoull(argv[3], NULL, 0);
    int      xor  = atoi(argv[4]);
    if (N < 2)             { fprintf(stderr, "need num_nodes >= 2\n"); return 1; }
    if (dest >= N)         { fprintf(stderr, "need dest < num_nodes\n"); return 1; }
    if (N > POOL_LINES / 2){ fprintf(stderr, "num_nodes too large for a sparse pool\n"); return 1; }

    uint64_t key = xor ? XOR_KEY : 0;

    uint64_t __trash = 0;
    srand((unsigned)time(NULL) ^ (unsigned)getpid());
    pin_cpu(cpu);
    _mm_mfence();

    /* large pool; nodes go on distinct random lines so the node set is sparse
     * and only a pointer-chaser (not a spatial HW prefetcher) can reach them */
    char *pool = (char *) mmap(NULL, (size_t)POOL_LINES * 64,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(pool != MAP_FAILED);

    uint64_t **addr = malloc(N * sizeof(uint64_t *));   /* node line addresses  */
    char      *used = calloc(POOL_LINES, 1);            /* line-occupancy map    */
    uint64_t  *perm = malloc(N * sizeof(uint64_t));
    if (!addr || !used || !perm) { fprintf(stderr, "alloc failed\n"); return 3; }

    /* pick N distinct random lines */
    for (uint64_t i = 0; i < N; i++) {
        uint64_t line;
        do { line = (((uint64_t)rand() << 31) ^ (uint64_t)rand()) % POOL_LINES; }
        while (used[line]);
        used[line] = 1;
        addr[i] = (uint64_t *)(pool + line * 64);
    }
    free(used);

    /* RANDOM ring: follow the Sattolo cycle perm[k] -> perm[k+1], next at
     * offset 0 of each node line, masked with key. */
    build_cycle(perm, N);
    for (uint64_t k = 0; k < N; k++) {
        uint64_t *here = addr[perm[k]];
        uint64_t *next = addr[perm[(k + 1) % N]];
        here[0] = (uint64_t)next ^ key;
    }
    uint64_t *start = addr[perm[0]];
    free(perm);
    _mm_mfence();

    uint64_t *time_measurements = calloc(N, sizeof(uint64_t)); /* per-node cumulative time */
    if (!time_measurements) { fprintf(stderr, "alloc failed\n"); return 3; }

    fprintf(stderr, "# rand N=%lu dest=%lu xor=%d key=0x%lx trials=%d cpu=%d\n",
            N, dest, xor, key, TRIALS, cpu);

    /* warmup chases: train the DMP, discard timings */
    for (int t = 0; t < WARMUP; t++) {
        flush_nodes(addr, N);
        uint64_t *curr = start;
        for (uint64_t k = 0; k < N; k++)
            (void)time_access_next(&curr, key, __trash);
    }

    /* one trial = flush the whole list, then chase; each hop's time is added into
     * its own slot, so every node index accumulates against itself across trials */
    for (int t = 0; t < TRIALS; t++) {
        flush_nodes(addr, N);
        uint64_t *curr = start;
        for (uint64_t k = 0; k < N; k++)
            time_measurements[k] += time_access_next(&curr, key, __trash);
    }

    double avg = (double)time_measurements[dest] / TRIALS;
    printf("N=%lu dest=%lu xor=%d: hop[%lu] avg=%.1f raw cyc  (trials=%d) [%lu]\n",
           N, dest, xor, dest, avg, TRIALS, __trash & MSB_MASK);

    free(time_measurements);
    free(addr);
    munmap(pool, (size_t)POOL_LINES * 64);
    return 0;
}

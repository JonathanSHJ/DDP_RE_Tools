/*
 * DMP_chase_linked_list_seq.c
 *
 * DMP pointer-chase over a singly linked list, SEQUENTIAL layout:
 * node[i].next = &node[i+1] (wrapped into a ring).
 *
 * ONE TRIAL = one full traversal of the N-node list:
 *   1. flush the whole list from cache (all N nodes cold)
 *   2. chase all N nodes (un-fenced, so a prefetcher has room to run ahead)
 *   3. time the whole walk; per-hop = total / N
 * We repeat this for TRIALS trials and report the average per-hop time. A pure
 * pointer chase is inherently serial (you can't know node k+1's address until
 * you've loaded node k), so with NO prefetcher every hop is a cold miss and the
 * per-hop average sits at the miss latency. If the average drops toward the hit
 * latency, something prefetched the nodes ahead of the demand chase. That is
 * the signal: avg near HIT => prefetched, avg near MISS => not.
 *
 * NOTE on layout: because seq nodes are adjacent in memory, an ordinary
 * stride/stream prefetcher can also predict them -- so a low average here does
 * NOT prove the DMP specifically. Use DMP_chase_linked_list_rand.c (scattered
 * visit order) to attribute a low average to the pointer-chasing DMP, and
 * ddp_control_bits.out to toggle the DDP as a control.
 *
 * Args:  <cpu> <num_nodes> <xor 0|1>
 *   xor : 0 -> canonical next-pointers (DMP can chase them)
 *         1 -> next ^ XOR_KEY (non-canonical bytes; DMP should refuse to chase)
 *
 * The un-mask xor is always in the chase loop; only the key differs at runtime
 * (0 vs XOR_KEY), so both modes share an identical instruction stream / load PC
 * (the DMP indexes its history by load PC). The walk still traverses correctly
 * when masked because it un-masks the loaded value in software.
 *
 * The timer is noise-subtracted (net cycles): it also times a bare rdtscp/lfence
 * gap with no load and subtracts it, so the large, machine-specific measurement
 * overhead (~175 cyc on server Xeon, ~45 on client) is removed and a hit reads
 * as a few cycles rather than ~overhead.
 *
 * Linux + x86 only. No sudo required.
 *   Build: make          (the DMP_RE_tools Makefile globs *.c)
 *   Run:   ./DMP_chase_linked_list_seq.out <cpu> <num_nodes> <xor 0|1>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include "util/util.h"

#define TRIALS    1000        /* full-list traversals recorded             */
#define WARMUP    50          /* warmup traversals (train the DMP), dropped */
#define XOR_KEY   0xA5A5A5A5A5A5A5A5ULL   /* flips high bits -> non-canonical */

/* next MUST be the first field: the chase loads offset 0 of the node
 * (mov (%[p]), %[p]) as the next-pointer. If data came first, padding would
 * push next to offset 8 and the load would read garbage -> segfault. */
struct node { struct node *next; int data; };

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Flush every node line so the next traversal starts fully cold. */
static void flush_list(struct node *nodes, uint64_t N)
{
    size_t bytes = (size_t)N * sizeof(struct node);
    for (size_t off = 0; off < bytes; off += 64)
        _mm_clflush((char *)nodes + off);
    _mm_mfence();
}

/* Time ONE full traversal: chase N hops un-fenced (loads may overlap so a
 * prefetcher can run ahead), then subtract the bare rdtscp/lfence overhead.
 * Returns net cycles for the whole walk; caller divides by N for per-hop. */
static inline uint64_t timed_traversal(struct node *start, uint64_t N, uint64_t key)
{
    uint64_t t0, t1, n0, n1;
    uint64_t p = (uint64_t)start, c = N;
    asm volatile(
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[t0]     \n\t"
        "lfence                \n\t"
        "1:                    \n\t"
        "mov  (%[p]), %[p]     \n\t"     /* p = *p  (next node, masked)     */
        "xor  %[key], %[p]     \n\t"     /* recover real pointer (sym.)     */
        "dec  %[c]             \n\t"
        "jnz  1b               \n\t"
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[t1]     \n\t"
        "lfence                \n\t"
        /* noise: same rdtscp+lfence bracket, no load */
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[n0]     \n\t"
        "lfence                \n\t"
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[n1]     \n\t"
        "lfence                \n\t"
        : [p] "+r"(p), [c] "+r"(c), [t0] "=&r"(t0), [t1] "=&r"(t1),
          [n0] "=&r"(n0), [n1] "=&r"(n1)
        : [key] "r"(key)
        : "rax", "rcx", "rdx", "cc", "memory"
    );
    asm volatile("" :: "r"(p) : "memory");   /* keep the chase result live */
    uint64_t total = t1 - t0, noise = n1 - n0;
    return total > noise ? total - noise : 0;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <cpu> <num_nodes> <xor 0|1>\n", argv[0]);
        return 1;
    }
    int      cpu = atoi(argv[1]);
    uint64_t N   = strtoull(argv[2], NULL, 0);
    int      xor = atoi(argv[3]);
    if (N < 2) { fprintf(stderr, "need num_nodes >= 2\n"); return 1; }

    uint64_t key = xor ? XOR_KEY : 0;

    pin_cpu(cpu);
    _mm_mfence();

    struct node *nodes;
    if (posix_memalign((void **)&nodes, 64, (size_t)N * sizeof(struct node)) != 0) {
        perror("posix_memalign");
        return 2;
    }

    /* SEQUENTIAL ring: node[i] -> node[i+1], last -> first. */
    for (uint64_t i = 0; i < N; i++) {
        struct node *next = &nodes[(i + 1) % N];
        nodes[i].data = (int)i;
        nodes[i].next = (struct node *)((uintptr_t)next ^ key);
    }
    struct node *start = &nodes[0];
    _mm_mfence();

    fprintf(stderr, "# seq  N=%lu xor=%d key=0x%lx trials=%d cpu=%d\n",
            N, xor, key, TRIALS, cpu);

    /* warmup traversals: train the DMP, discard timings */
    for (int t = 0; t < WARMUP; t++) {
        flush_list(nodes, N);
        (void)timed_traversal(start, N, key);
    }

    /* one trial = flush the whole list, then time one full traversal */
    static uint64_t net[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        flush_list(nodes, N);
        net[t] = timed_traversal(start, N, key);
    }

    /* per-hop = traversal_net / N; report average + median across trials */
    uint64_t sum = 0;
    for (int t = 0; t < TRIALS; t++) sum += net[t];
    double avg = (double)sum / TRIALS / (double)N;

    qsort(net, TRIALS, sizeof(uint64_t), cmp_u64);
    double median = (double)net[TRIALS / 2] / (double)N;

    printf("per-hop avg=%.1f median=%.1f net cyc  (N=%lu, trials=%d, xor=%d)\n",
           avg, median, N, TRIALS, xor);

    /* Keep the per-node data stores alive so the compiler can't drop them
     * (leaving the non-pointer node bytes uninitialized). Read them (volatile)
     * into a trash sink AFTER timing so it never warms a chased line. */
    uint64_t trash = 0;
    for (uint64_t i = 0; i < N; i++)
        trash += (uint64_t)((volatile struct node *)&nodes[i])->data;
    fprintf(stderr, "# trash=%lu\n", trash);

    free(nodes);
    return 0;
}

/*
 * indirect.c -- indirect / base+index DDP prefetch test.
 *
 * Adapted from Peek-a-Walk INTEL_DDP/paper_experiments/indirect.c to the
 * DMP_RE_tools conventions (local util/, <cpu> as arg 1). Where the AoP test
 * chases pointers, this one chases INDICES: the gadget loads Z[i] from the index
 * array, then (when the consumer bit is set) does the second load Y[Z[i]] -- the
 * indirect/double-dereference pattern the DMP trains on.
 *
 * The target index is planted ONE PAST the walked window, at
 * training_idx[training_size + test_ptr_offset], so ptr_under_test lives OUTSIDE
 * the range the gadget actually accesses. A fast access to ptr_under_test then
 * means the DMP ran ahead of the walk and speculatively computed Y[Z[.]] for an
 * index we never touched -- i.e. it prefetched beyond the trained range.
 *
 * Trials alternate base/attack: odd trials plant the real index (measure the
 * hit), even trials plant NULL and assert no hit (background check).
 *
 * Linux + x86 only. No sudo for the probe itself.
 *   Run: ./indirect.out <cpu> <training_size> <test_ptr_offset>
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

#define TRIALS   10000

#define BUF_ADDR 0x666600000000UL    /* scattered data buffer Y        */
#define IDX_ADDR 0x666700000000UL    /* 2 MB-aligned index array Z     */

/* Training / testing gadget: walk `size` indices; for each, load Z[i] into rax,
 * and (iff consumer_load_bit) do the indirect consumer load Y[Z[i]]. The load of
 * Z[i] followed by Y[Z[i]] is the double-dereference the DMP trains on. */
__attribute__((noinline))
static uint64_t access_aop(uint64_t *buf, uint64_t *indexes, uint64_t size,
                           uint64_t consumer_load_bit, uint64_t __trash)
{
    asm volatile(
        "xor %%r12, %%r12\n"
        "xor %%r13, %%r13\n"
        "xor %%rbx, %%rbx\n"

        "1:\n"
        "mov (%2, %%r12, 8), %%rax\n"  // rax = Z[i]  (load the index)
        "cmp $1, %5\n"
        "jne 2f\n"
        "mfence\n"                     // spec barrier
        "or (%1, %%rax, 8), %0\n"      // trash |= Y[Z[i]]  (indirect consumer load)

        "2:\n"
        "add $1, %%r12\n"              // stride
        "add $1, %%r13\n"              // count
        "cmp %4, %%r13\n"

        "jl 1b\n"
        : "=r" (__trash)
        : "r" (buf), "r" (indexes), "0" (__trash), "r" (size), "r" (consumer_load_bit)
        : "cc", "r11", "r12", "r13", "rax", "rbx"
    );
    return __trash & MSB_MASK;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <cpu> <training_size> <test_ptr_offset>\n", argv[0]);
        return 1;
    }
    int      cpu             = atoi(argv[1]);
    uint64_t training_size   = strtoull(argv[2], NULL, 0);
    uint64_t test_ptr_offset = strtoull(argv[3], NULL, 0);

    /* init ourselves */
    uint64_t __trash = 0;
    pin_cpu(cpu);
    __trash = c_sleep(3000, __trash);
    _mm_mfence();

    /* data buffer Y: one live word per cache line, spread over buffer_size lines
     * so only a pointer-chasing DMP (not a spatial prefetcher) can reach them */
    uint64_t buffer_size = 1 << 20;
    uint64_t *training_buffer = (uint64_t *) mmap((void *)BUF_ADDR,
                                                  sizeof(uint64_t) * buffer_size * 8,
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_POPULATE,
                                                  -1, 0);
    assert(training_buffer != MAP_FAILED);
    for (uint64_t i = 0; i < buffer_size * 8; i += 8)
        training_buffer[i] = rand();

    /* index array Z: each entry 8*line so entries land on unique cache lines */
    uint64_t *training_idx = (uint64_t *) mmap((void *)IDX_ADDR,
                                               sizeof(uint64_t *) * training_size + 100 + MB(2),
                                               PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_POPULATE,
                                               -1, 0);
    assert(training_idx != MAP_FAILED);
    assert((uint64_t)training_idx % MB(2) == 0);     /* align to 2 MB */
    for (uint64_t i = 0; i < training_size + test_ptr_offset; i++)
        training_idx[i] = 8 * (rand() % buffer_size);

    /* plant the target index one past the walked window */
    uint64_t index_under_test = 8 * (rand() % buffer_size);
    training_idx[training_size + test_ptr_offset] = index_under_test;

    /* ptr_under_test lives at Y[index_under_test], outside the trained range */
    volatile uint64_t *ptr_under_test = &training_buffer[training_idx[training_size + test_ptr_offset]];
    assert((uint64_t)ptr_under_test % 64 == 0);      /* align to cache line */
    *ptr_under_test = 0xdeadbeef;

    sleep(0);

    /* main loop: alternate base (even) and attack (odd) trials */
    uint64_t times[TRIALS] = {0};
    for (int i = 0; i < TRIALS * 2; i++) {

        /* kill the DMP: walk 20 indices with the consumer load off */
        __trash = access_aop(training_buffer, training_idx, 20, 0, __trash);
        _mm_mfence();

        /* flush the trained window: both Y[Z[j]] and Z[j] */
        for (uint64_t j = 0; j < training_size; j++) {
            __trash = clflush(&training_buffer[training_idx[j]], __trash);
            __trash = clflush(&training_idx[j], __trash);
        }
        _mm_mfence();

        /* attack trial -> plant the real index; base trial -> plant NULL */
        if (i % 2)
            training_idx[training_size + test_ptr_offset] = index_under_test;
        else
            training_idx[training_size + test_ptr_offset] = 0;

        /* flush ptr_under_test */
        __trash = clflush((uint64_t *)ptr_under_test, __trash);
        _mm_mfence();

        /* training + activation: double-deref walk over training_size indices */
        __trash = access_aop(training_buffer, training_idx, training_size, 1, __trash);
        _mm_mfence();

        /* give the DMP time to run ahead */
        __trash = c_sleep(200, __trash);

        /* time the target: record on attack trials, sanity-check on base trials */
        uint64_t trial_access_time = time_access((uint64_t *)ptr_under_test, __trash);
        if (i % 2)
            times[i / 2] = trial_access_time;
        else
            assert(trial_access_time > 100 && "Base mode registered a hit of the ptr_under_test!");
    }

    /* output */
    for (int i = 0; i < TRIALS; i++)
        fprintf(stderr, "%lu ", times[i]);
    fprintf(stderr, "\n");

    return 0;
}

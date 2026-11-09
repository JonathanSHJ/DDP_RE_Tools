/*
 * ddp_chase_isolate.c
 *
 * Isolate and quantify the Data-Dependent Prefetcher (DMP) effect with an
 * amat-style dependent pointer chase, run as TWO SEPARATED PHASES.
 *
 * Two isolation mechanisms (from the amat pair):
 *
 *  1. Random single-cycle layout (Sattolo). The chase visits node addresses
 *     in a random permuted order, so the classic stride/stream prefetchers
 *     (L2 streamer, adjacent-line, DCU, DCU-IP) cannot predict the next
 *     address and contribute nothing -- no MSR writes needed. A *pointer*
 *     prefetcher (the DMP) still works, because it dereferences the pointer
 *     value in the loaded line ahead of the demand load.
 *
 *  2. XOR-mask toggle. Each node stores  next ^ key, and the chase recovers
 *     the real pointer in-loop with  xor key. Two rings are built:
 *        masked ring: key = XOR_KEY  -> stored bytes non-canonical -> DMP OFF
 *        plain  ring: key = 0        -> stored bytes canonical     -> DMP ON
 *     The chase loop is IDENTICAL for both (the xor is always present, only
 *     the key differs), so the sole difference is byte canonicality.
 *
 * WHY THE MODES ARE NOT INTERLEAVED
 * ---------------------------------
 * Both modes execute the same chase load, hence the SAME load PC, and the DMP
 * indexes its history/confidence table by load PC. Alternating modes would
 * feed non-pointer values to the very table entry the plain mode is training,
 * so the DMP would never reach steady-state confidence and its effect would be
 * diluted or hidden. Instead each mode runs as an uninterrupted block, with
 * isolation between blocks:
 *     - rings mapped at deliberately distant virtual addresses
 *     - full cache flush (walk a buffer 2x the LLC)
 *     - syscalls (sched_yield + usleep) to flush DMP state
 *     - warmup trials at the start of each block are discarded
 * Blocks run in ABBA order (masked, plain, plain, masked) so linear drift in
 * frequency/temperature cancels without ever interleaving at trial level.
 *
 * median(masked) - median(plain), in cycles/access, is the isolated DMP
 * contribution to AMAT.
 *
 * Linux + x86 only. No sudo required -- everything here is unprivileged
 * (sched_setaffinity, mmap, rdtscp). sudo only adds the optional nice(-20)
 * boost in pin_cpu, which just reduces timing noise.
 *   Build: make
 *   Run:   ./ddp_chase_isolate.out <cpu> <num_nodes> <stride_bytes> [huge 0|1]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <assert.h>
#include <x86intrin.h>
#include <sys/mman.h>
#include "util/util.h"

#define TRIALS      10000          /* total, split evenly across the two modes */
#define BLOCKS      4              /* ABBA: masked, plain, plain, masked       */
#define WARMUP      50             /* discarded trials at the start of a block */
#define CHASE_LEN   4096           /* dependent loads timed per trial          */
#define XOR_KEY     0xA5A5A5A5A5A5A5A5ULL
#define MIN_STRIDE  64
#define MASKED_HINT ((void *)0x100000000000UL)   /* rings kept far apart */
#define PLAIN_HINT  ((void *)0x500000000000UL)

/* ---------- ring construction ---------- */

/* Sattolo's algorithm: a single N-cycle (not just any permutation). */
static void build_cycle(uint64_t *perm, uint64_t N) {
    for (uint64_t i = 0; i < N; i++) perm[i] = i;
    for (uint64_t i = N - 1; i > 0; i--) {
        uint64_t j = (((uint64_t)rand() << 31) ^ (uint64_t)rand()) % i;
        uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}

/* Store next-pointer XOR'd with `key`. key=XOR_KEY -> non-canonical (DMP off);
 * key=0 -> raw canonical pointer (DMP on). Same layout/order for both rings. */
static void build_ring(char *buf, uint64_t N, uint64_t stride,
                       uint64_t *perm, uint64_t key) {
    for (uint64_t k = 0; k < N; k++) {
        uint64_t *here = (uint64_t *)(buf + perm[k]           * stride);
        uint64_t *next = (uint64_t *)(buf + perm[(k + 1) % N] * stride);
        *here = ((uint64_t)next) ^ key;
    }
}

static char *map_ring(void *hint, size_t bytes, int huge) {
    char *buf = mmap(hint, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (buf == MAP_FAILED)   /* hint unavailable: let the kernel choose */
        buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %zu bytes: %s\n", bytes, strerror(errno));
        exit(2);
    }
    if (huge) madvise(buf, bytes, MADV_HUGEPAGE);
    for (size_t off = 0; off < bytes; off += 4096) buf[off] = 0;  /* pre-fault */
    return buf;
}

/* ---------- isolation between blocks ---------- */

static char  *flush_buf;
static size_t flush_bytes;

/* LLC size from sysfs (e.g. "24576K"); generous fallback if unavailable. */
static size_t llc_bytes(int cpu) {
    char path[128];
    for (int idx = 4; idx >= 1; idx--) {           /* prefer the largest index */
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, idx);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char v[32] = {0};
        if (fgets(v, sizeof v, f)) {
            fclose(f);
            size_t n = strtoul(v, NULL, 10);
            if (strchr(v, 'M')) n *= 1024 * 1024;
            else if (strchr(v, 'K')) n *= 1024;
            if (n) return n;
        } else fclose(f);
    }
    return 32UL * 1024 * 1024;
}

/* Evict every cache level, then flush DMP state via syscalls. */
static void isolate(void) {
    for (size_t i = 0; i < flush_bytes; i += 64) flush_buf[i] = (char)i;
    _mm_mfence();
    /* Peek-a-Walk's syscall_test.c: a syscall flushes DMP state */
    sched_yield();
    usleep(0);
    _mm_mfence();
}

/* ---------- the chase ---------- */

/* lfence-bracketed rdtscp around chase_len dependent loads; the xor recovers
 * the real next pointer. Identical for both modes (only key differs). */
static inline uint64_t timed_chase(uint64_t **pp, uint64_t chase_len, uint64_t key) {
    uint64_t t0, t1;
    uint64_t p = (uint64_t)(*pp);
    asm volatile(
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[t0]     \n\t"
        "lfence                \n\t"
        "mov  %[len], %%rcx    \n\t"
        "1:                    \n\t"
        "mov  (%[ptr]), %[ptr] \n\t"     /* ptr = *ptr            */
        "xor  %[key],  %[ptr]  \n\t"     /* recover real pointer  */
        "dec  %%rcx            \n\t"
        "jnz  1b               \n\t"
        "lfence                \n\t"
        "rdtscp                \n\t"
        "shl  $32, %%rdx       \n\t"
        "or   %%rdx, %%rax     \n\t"
        "mov  %%rax, %[t1]     \n\t"
        "lfence                \n\t"
        : [ptr] "+r"(p), [t0] "=&r"(t0), [t1] "=&r"(t1)
        : [len] "r"(chase_len), [key] "r"(key)
        : "rax", "rcx", "rdx", "cc", "memory"
    );
    *pp = (uint64_t *)p;
    return t1 - t0;
}

/* One uninterrupted block: warm the DMP up, then record `n` trials. */
static void run_block(uint64_t **cur, uint64_t key, uint64_t *out, int n) {
    for (int i = 0; i < WARMUP; i++) (void)timed_chase(cur, CHASE_LEN, key);
    for (int i = 0; i < n; i++)      out[i] = timed_chase(cur, CHASE_LEN, key);
}

/* ---------- stats ---------- */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* sorts v; reports min / p10 / median in cycles-per-access */
static void report(const char *label, uint64_t *v, int n,
                   double *median_cpa) {
    qsort(v, n, sizeof(uint64_t), cmp_u64);
    double mn  = (double)v[0]        / CHASE_LEN;
    double p10 = (double)v[n / 10]   / CHASE_LEN;
    double med = (double)v[n / 2]    / CHASE_LEN;
    printf("  %-18s min=%7.2f  p10=%7.2f  median=%7.2f  cyc/access\n",
           label, mn, p10, med);
    if (median_cpa) *median_cpa = med;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <cpu> <num_nodes> <stride_bytes> [huge 0|1]\n",
                argv[0]);
        return 1;
    }
    int      cpu    = atoi(argv[1]);
    uint64_t N      = strtoull(argv[2], NULL, 0);
    uint64_t stride = strtoull(argv[3], NULL, 0);
    int      huge   = (argc == 5) ? atoi(argv[4]) : 0;

    assert(N >= 2);
    assert(stride >= MIN_STRIDE && (stride & (stride - 1)) == 0);

    srand((unsigned)time(NULL) ^ (unsigned)getpid());
    pin_cpu(cpu);
    _mm_mfence();

    size_t bytes = (size_t)N * (size_t)stride;
    char *masked = map_ring(MASKED_HINT, bytes, huge);   /* DMP off */
    char *plain  = map_ring(PLAIN_HINT,  bytes, huge);   /* DMP on  */

    /* cache-flush buffer for isolation: 2x LLC */
    flush_bytes = 2 * llc_bytes(cpu);
    flush_buf = mmap(NULL, flush_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (flush_buf == MAP_FAILED) { perror("mmap flush_buf"); return 2; }

    /* one permutation, identical structure for both rings */
    uint64_t *perm = malloc(N * sizeof(uint64_t));
    if (!perm) { fprintf(stderr, "perm alloc failed\n"); return 3; }
    build_cycle(perm, N);
    build_ring(masked, N, stride, perm, XOR_KEY);
    build_ring(plain,  N, stride, perm, 0);
    uint64_t *cur_m = (uint64_t *)(masked + perm[0] * stride);
    uint64_t *cur_p = (uint64_t *)(plain  + perm[0] * stride);
    free(perm);
    _mm_mfence();

    fprintf(stderr, "# cpu=%d N=%lu stride=%lu huge=%d chase_len=%d "
                    "bytes=%zu/ring flush=%zuMB\n",
            cpu, N, stride, huge, CHASE_LEN, bytes, flush_bytes >> 20);
    fprintf(stderr, "# ABBA blocks of %d trials, %d warmup discarded per block\n",
            (TRIALS / BLOCKS), WARMUP);

    int per_block = TRIALS / BLOCKS;          /* trials recorded per block */
    uint64_t *off = malloc(sizeof(uint64_t) * per_block * 2);  /* masked */
    uint64_t *on  = malloc(sizeof(uint64_t) * per_block * 2);  /* plain  */

    /* ABBA: masked, plain, plain, masked -- long uncontaminated training
     * blocks, but linear drift still cancels across the pair ordering. */
    const int order[BLOCKS] = {0, 1, 1, 0};   /* 0 = masked, 1 = plain */
    int n_off = 0, n_on = 0;
    for (int b = 0; b < BLOCKS; b++) {
        isolate();
        if (order[b] == 0) { run_block(&cur_m, XOR_KEY, off + n_off, per_block); n_off += per_block; }
        else               { run_block(&cur_p, 0,       on  + n_on,  per_block); n_on  += per_block; }
        fprintf(stderr, "# block %d/%d done (%s)\n", b + 1, BLOCKS,
                order[b] == 0 ? "masked/DMP-off" : "plain/DMP-on");
    }

    double cpa_off, cpa_on;
    printf("\n");
    report("DMP off (masked)", off, n_off, &cpa_off);
    report("DMP on  (plain) ", on,  n_on,  &cpa_on);

    double delta = cpa_off - cpa_on;
    double pct   = cpa_off > 0 ? 100.0 * delta / cpa_off : 0.0;
    printf("\nisolated DMP effect: %.2f cyc/access faster (%.1f%%)\n", delta, pct);
    printf("VERDICT: %s\n", (delta > 0 && pct > 15.0)
        ? "DMP PRESENT (canonical pointers chased noticeably faster)"
        : "no DMP effect (canonical vs masked chase within noise)");

    munmap(masked, bytes);
    munmap(plain,  bytes);
    munmap(flush_buf, flush_bytes);
    return 0;
}

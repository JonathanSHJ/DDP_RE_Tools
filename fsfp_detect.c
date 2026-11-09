/*
 * fsfp_detect.c -- bare-bones, no-root behavioral detector for a Fast
 * Store-Forwarding Predictor (FSFP) on x86-64 Intel.
 *
 * FSFP is a load/store-unit predictor that speculatively forwards a store's data
 * to a younger load BEFORE the addresses are fully checked, and LEARNS which
 * store->load pairs alias. It is NOT a data prefetcher -- there is no stride or
 * cache-line-fill detection here. This tool only answers "does the predictor
 * exist and learn"; it reads no secret and crosses no boundary.
 *
 * ORACLE SUBSTITUTE (no PSFD MSR, so behavioral inference -- weaker by design):
 *   COLD : a store->load pair measured with the predictor reset (untrained) -> if
 *          FSFP exists the load cannot be speculatively forwarded, so load-to-use
 *          is slower.
 *   WARM : the IDENTICAL pair (same addresses, same code) after K training reps ->
 *          the predictor has learned it -> speculative forward -> faster.
 * Everything except training is held identical, so a stable COLD>WARM drop is the
 * primary evidence. ADDRESS ALIASING (load addr == store addr vs. offset so it
 * cannot alias) is reported as corroboration only.
 *
 * The measured kernel is a serial store->load chain to one buffer line. The store
 * ADDRESS is made data-dependent (`and $0`) so it resolves late; a load to the
 * same line must then either wait for that resolution (no prediction) or be
 * speculatively forwarded (prediction). That is exactly the latency the predictor
 * changes, which is why this kernel isolates forwarding rather than cache latency.
 *
 * Reused: util pin_cpu/clflush, <cpuid.h>, the rdtscp/lfence timing bracket and
 * qsort-median idiom from the repo. Added: config parser, stats, sysfs env
 * checks, the FSFP kernel + predictor thrash.
 *
 * Linux + x86-64, no root.  Build: make    Run: ./fsfp_detect.out [config]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cpuid.h>
#include <sys/mman.h>
#include <x86intrin.h>
#include "util/util.h"

/* ------------------------------- config ---------------------------------- */
typedef struct {
    int      cpu_core;       /* logical core to pin to                         */
    uint64_t training_iters; /* K: training reps before each WARM measurement  */
    int      trials;         /* averaged measurements per condition            */
    int      warmup;         /* discarded trials up front                      */
    uint64_t chain_len;      /* store->load pairs per timed kernel             */
    uint64_t offset_bytes;   /* load offset for the NON-aliasing condition     */
    uint64_t thrash_lines;   /* distinct lines touched to reset the predictor  */
    int      allow_unsafe;   /* 1 = proceed even if env prereqs fail           */
} config_t;

static void config_defaults(config_t *c) {
    c->cpu_core = 4; c->training_iters = 64; c->trials = 4000; c->warmup = 200;
    c->chain_len = 128; c->offset_bytes = 256; c->thrash_lines = 4096;
    c->allow_unsafe = 0;
}

/* simple key=value parser (repo has no shared config convention) */
static void config_load(config_t *c, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[cfg] '%s' not found; using defaults\n", path); return; }
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        char key[64]; unsigned long long val;
        if (sscanf(line, " %63[a-z_] = %llu", key, &val) != 2) continue;
        if      (!strcmp(key, "cpu_core"))      c->cpu_core      = (int)val;
        else if (!strcmp(key, "training_iters"))c->training_iters= val;
        else if (!strcmp(key, "trials"))        c->trials        = (int)val;
        else if (!strcmp(key, "warmup"))        c->warmup        = (int)val;
        else if (!strcmp(key, "chain_len"))     c->chain_len     = val;
        else if (!strcmp(key, "offset_bytes"))  c->offset_bytes  = val;
        else if (!strcmp(key, "thrash_lines"))  c->thrash_lines  = val;
        else if (!strcmp(key, "allow_unsafe"))  c->allow_unsafe  = (int)val;
    }
    fclose(f);
}

/* ------------------------------- cpuid ----------------------------------- */
static void report_cpu(void) {
    unsigned a, b, c, d;
    __get_cpuid(1, &a, &b, &c, &d);
    int family = ((a >> 8) & 0xf) + ((a >> 20) & 0xff);   /* extended family  */
    int model  = ((a >> 4) & 0xf) | (((a >> 16) & 0xf) << 4);
    int step   = a & 0xf;
    char brand[49] = {0};
    unsigned *bp = (unsigned *)brand;
    for (unsigned leaf = 0x80000002; leaf <= 0x80000004; leaf++)
        __get_cpuid(leaf, &bp[0], &bp[1], &bp[2], &bp[3]), bp += 4;
    printf("CPU: %s\n", brand[0] ? brand : "(unknown)");
    printf("     family=0x%x model=0x%x stepping=%d\n", family, model, step);
}

/* --------------------------- env prerequisites --------------------------- */
/* Read a small sysfs string; returns 1 on success. */
static int read_sysfs(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    int ok = fgets(out, n, f) != NULL; fclose(f);
    if (ok) out[strcspn(out, "\n")] = 0;
    return ok;
}

/* Fail loudly if turbo/governor/SMT/C-states aren't set for stable timing. */
static int env_check(const config_t *c) {
    char v[64]; int problems = 0;

    if (read_sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo", v, sizeof v)) {
        if (strcmp(v, "1")) { printf("[env] turbo ENABLED -> `echo 1 | sudo tee "
            "/sys/devices/system/cpu/intel_pstate/no_turbo`\n"); problems++; }
    } else if (read_sysfs("/sys/devices/system/cpu/cpufreq/boost", v, sizeof v)) {
        if (strcmp(v, "0")) { printf("[env] boost ENABLED -> `echo 0 | sudo tee "
            "/sys/devices/system/cpu/cpufreq/boost`\n"); problems++; }
    }

    char p[128];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", c->cpu_core);
    if (read_sysfs(p, v, sizeof v) && strcmp(v, "performance")) {
        printf("[env] governor='%s' (want performance) -> `sudo cpupower frequency-set -g performance`\n", v);
        problems++;
    }

    /* SMT is a NOTE, not a hard prereq: SMT on with an idle sibling is fine. */
    if (read_sysfs("/sys/devices/system/cpu/smt/control", v, sizeof v)) {
        if (!strcmp(v, "on")) printf("[env] note: SMT ON -- ensure the sibling of core %d is "
            "idle (or `echo off | sudo tee /sys/devices/system/cpu/smt/control`).\n", c->cpu_core);
    }

    /* deep C-states: state0/1 are fine; anything deeper should be disabled */
    for (int s = 2; s <= 4; s++) {
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpuidle/state%d/disable", c->cpu_core, s);
        if (read_sysfs(p, v, sizeof v) && !strcmp(v, "0")) {
            printf("[env] C-state%d enabled -> disable deep C-states (cpupower idle-set -D 1) for stable timing\n", s);
            problems++; break;
        }
    }

    if (problems && !c->allow_unsafe) {
        printf("[env] %d prerequisite(s) unmet; fix them or set allow_unsafe=1 in the config.\n", problems);
        return 0;
    }
    if (problems) printf("[env] proceeding with %d unmet prereq(s) (allow_unsafe=1); numbers will be noisy.\n", problems);
    return 1;
}

/* ------------------------------- stats ----------------------------------- */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static double median_d(uint64_t *v, int n) {  /* v is sorted by caller */
    return n & 1 ? (double)v[n/2] : (v[n/2 - 1] + v[n/2]) / 2.0;
}
static double trimmed_mean(uint64_t *v, int n, double trim) {  /* v sorted */
    int k = (int)(n * trim), cnt = 0; double s = 0;
    for (int i = k; i < n - k; i++) { s += v[i]; cnt++; }
    return cnt ? s / cnt : 0;
}

/* ---------------------------- FSFP kernel -------------------------------- */
/* One timed serial store->load chain over `base`'s line. Each iteration:
 *   store r -> [base + (r & 0)]   (store address DEPENDS on r -> resolves late)
 *   load  r <- [base + load_off]  (aliases the store when load_off == 0)
 * The load-to-use of r gates the next store's address, so the whole chain runs at
 * the store->load forwarding latency. When aliasing + predicted, the load is
 * forwarded speculatively (fast); otherwise it waits for the late store address
 * (slow). Returns raw cycles for chain_len iterations. */
static uint64_t volatile g_sink;
static inline uint64_t sl_chain(uint64_t *base, long load_off, uint64_t chain, uint64_t seed) {
    uint64_t t0, t1, r = seed;
    asm volatile(
        "lfence\n\t rdtscp\n\t shl $32,%%rdx\n\t or %%rdx,%%rax\n\t mov %%rax,%[t0]\n\t lfence\n\t"
        "mov %[cl], %%rcx\n\t"
        "1:\n\t"
        "mov %[base], %%r8\n\t"       /* r8 = base                                */
        "mov %[r],   %%r9\n\t"
        "and $0,     %%r9\n\t"        /* r9 = 0 but data-dependent on r           */
        "add %%r9,   %%r8\n\t"        /* r8 = base, resolved only after r is ready */
        "mov %[r],   (%%r8)\n\t"      /* STORE r -> [base]                        */
        "mov %[base], %%r10\n\t"
        "add %[loff], %%r10\n\t"
        "mov (%%r10), %[r]\n\t"       /* LOAD  -> r (forwarded iff aliasing+pred) */
        "add $1,      %[r]\n\t"       /* keep the chain live                      */
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        "lfence\n\t rdtscp\n\t shl $32,%%rdx\n\t or %%rdx,%%rax\n\t mov %%rax,%[t1]\n\t lfence\n\t"
        : [r] "+r"(r), [t0] "=&r"(t0), [t1] "=&r"(t1)
        : [base] "r"(base), [loff] "r"(load_off), [cl] "r"(chain)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "cc", "memory"
    );
    g_sink = r;                        /* keep r live so nothing is optimized out */
    return t1 - t0;
}

/* Reset the predictor: run many distinct store->load pairs so the target pair's
 * learned entry is evicted (address-indexed) / aged out. Same flush is applied
 * separately; this only churns predictor state. */
static void thrash_predictor(uint64_t *tbuf, uint64_t lines) {
    for (uint64_t i = 0; i < lines; i++) {
        volatile uint64_t *p = &tbuf[i * 8];   /* one distinct line per iter */
        *p = i;
        g_sink += *p;
    }
    _mm_mfence();
}

int main(int argc, char **argv) {
    config_t cfg; config_defaults(&cfg);
    config_load(&cfg, argc > 1 ? argv[1] : "fsfp.cfg");

    report_cpu();
    printf("core=%d K=%lu trials=%d chain_len=%lu offset_bytes=%lu\n",
           cfg.cpu_core, cfg.training_iters, cfg.trials, cfg.chain_len, cfg.offset_bytes);

    pin_cpu(cfg.cpu_core);
    if (!env_check(&cfg)) return 2;

    /* our own measurement buffer (one line used) and a thrash buffer */
    uint64_t *buf = mmap(NULL, MB(2), PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    uint64_t *tbuf = mmap(NULL, (size_t)cfg.thrash_lines * 64 + 4096, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED || tbuf == MAP_FAILED) { perror("mmap"); return 3; }
    uint64_t *base = &buf[64];                    /* a settled, aligned line   */
    long off = (long)cfg.offset_bytes;

    uint64_t *cold = malloc(sizeof(uint64_t) * cfg.trials);
    uint64_t *warm = malloc(sizeof(uint64_t) * cfg.trials);
    uint64_t *alia = malloc(sizeof(uint64_t) * cfg.trials);
    uint64_t *nona = malloc(sizeof(uint64_t) * cfg.trials);

    for (int t = -cfg.warmup; t < cfg.trials; t++) {
        /* ---- COLD: reset predictor, identical flush, first observation ---- */
        thrash_predictor(tbuf, cfg.thrash_lines);
        _mm_clflush(base); _mm_mfence();
        uint64_t c = sl_chain(base, 0, cfg.chain_len, 0x1234);

        /* ---- WARM: train K reps of the identical pair, identical flush ---- */
        for (uint64_t k = 0; k < cfg.training_iters; k++)
            (void)sl_chain(base, 0, cfg.chain_len, 0x1234);
        _mm_clflush(base); _mm_mfence();
        uint64_t w = sl_chain(base, 0, cfg.chain_len, 0x1234);

        /* ---- aliasing corroboration (predictor already warm) ---- */
        _mm_clflush(base); _mm_mfence();
        uint64_t al = sl_chain(base, 0,   cfg.chain_len, 0x1234);   /* load aliases store */
        _mm_clflush(base); _mm_mfence();
        uint64_t nl = sl_chain(base, off, cfg.chain_len, 0x1234);   /* load cannot alias  */

        if (t >= 0) { cold[t] = c; warm[t] = w; alia[t] = al; nona[t] = nl; }
    }

    /* per-iteration latency = raw cycles / chain_len */
    qsort(cold, cfg.trials, 8, cmp_u64);
    qsort(warm, cfg.trials, 8, cmp_u64);
    qsort(alia, cfg.trials, 8, cmp_u64);
    qsort(nona, cfg.trials, 8, cmp_u64);
    double L = (double)cfg.chain_len;
    double cold_m = median_d(cold, cfg.trials) / L, cold_t = trimmed_mean(cold, cfg.trials, 0.1) / L;
    double warm_m = median_d(warm, cfg.trials) / L, warm_t = trimmed_mean(warm, cfg.trials, 0.1) / L;
    double alia_m = median_d(alia, cfg.trials) / L;
    double nona_m = median_d(nona, cfg.trials) / L;
    /* robust dispersion of warm (IQR) for the separation check */
    double warm_iqr = (warm[(int)(cfg.trials*0.75)] - warm[(int)(cfg.trials*0.25)]) / L;

    printf("\n--- per-iteration store->load latency (cycles) ---\n");
    printf("  COLD (untrained) : median=%.2f  trimmed=%.2f\n", cold_m, cold_t);
    printf("  WARM (trained)   : median=%.2f  trimmed=%.2f\n", warm_m, warm_t);
    printf("  ALIAS  (warm)    : median=%.2f\n", alia_m);
    printf("  NON-ALIAS (warm) : median=%.2f  (load offset %lu B)\n", nona_m, cfg.offset_bytes);

    double d_train = cold_m - warm_m;          /* primary: training effect      */
    double d_alias = nona_m - alia_m;          /* corroboration: aliasing effect */
    printf("\n  delta COLD-WARM      = %+.2f cyc/iter  (primary)\n", d_train);
    printf("  delta NONALIAS-ALIAS = %+.2f cyc/iter  (corroboration)\n", d_alias);
    printf("  warm IQR             = %.2f cyc/iter  (noise floor)\n", warm_iqr);

    /* Verdict: require the training drop to clear the noise floor and be a real
     * fraction of warm latency, corroborated (same sign) by aliasing. Heuristic
     * thresholds -- numbers are printed above so the result can be judged. */
    const double MIN_DELTA = 0.5;              /* cyc/iter above measurement noise */
    int primary  = d_train > MIN_DELTA && d_train > 3.0 * warm_iqr && d_train > 0.10 * warm_m;
    int corrob   = d_alias > MIN_DELTA;
    int flat     = d_train < 0.2 && d_alias < 0.2;

    printf("\nVERDICT: ");
    if (primary && corrob) printf("PASS -- FSFP detected (trains + aliasing-dependent forwarding).\n");
    else if (flat)         printf("FAIL -- no store-forwarding prediction detected.\n");
    else                   printf("INCONCLUSIVE -- effect present but not cleanly separated; "
                                  "raise trials/chain_len/K, ensure turbo+SMT off, re-run.\n");

    free(cold); free(warm); free(alia); free(nona);
    return 0;
}

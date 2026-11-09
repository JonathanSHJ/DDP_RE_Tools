/*
 * ddp_control_bits.c
 *
 * Reads the architectural Data-Dependent Prefetcher (DDP) control bits that
 * Intel documents on its DDP guidance page, to see whether the DDP control is
 * exposed and whether the DDP is currently enabled:
 *
 *   CPUID.(EAX=7,ECX=2).EDX[3]  = DDPD_U supported (CPU exposes the control)
 *   IA32_SPEC_CTRL (MSR 0x48) bit 8 = DDPD
 *        DDPD = 0 -> DDP ENABLED (active)
 *        DDPD = 1 -> DDP DISABLED
 *
 * It can also TOGGLE the DDP system-wide by writing DDPD on every logical CPU
 * (read-modify-write, so the other SPEC_CTRL bits -- IBRS/STIBP/SSBD -- are
 * preserved).
 *
 * CAVEAT: Linux owns IA32_SPEC_CTRL for its Spectre mitigations and may rewrite
 * it on context switch / return-to-user, which can clobber the DDPD bit you set.
 * So: set it, then run the pinned test immediately (high priority, few switches),
 * and re-check with `get` afterwards. For a durable disable use a BIOS/firmware
 * setting if your platform exposes one.
 *
 * Linux + x86 only. Needs root: it reads/writes /dev/cpu/N/msr (root-only, and
 * needs `modprobe msr`). The CPUID part works unprivileged.
 *   Build: make
 *   Run:   sudo ./ddp_control_bits.out <logical_cpu_id> [get|disable|enable]
 *            get      (default) print CPUID support + current DDPD on <cpu>
 *            disable  set   DDPD=1 on ALL logical CPUs (DDP off)
 *            enable   clear DDPD=0 on ALL logical CPUs (DDP on)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cpuid.h>
#include "util/util.h"

#define IA32_SPEC_CTRL 0x48
#define DDPD_BIT       8

static uint64_t read_msr(int cpu, uint32_t reg)
{
    char path[64];
    snprintf(path, sizeof path, "/dev/cpu/%d/msr", cpu);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open /dev/cpu/N/msr (need root and `modprobe msr`)");
        exit(1);
    }
    uint64_t val;
    if (pread(fd, &val, sizeof val, reg) != (ssize_t)sizeof val) {
        perror("pread msr");
        exit(1);
    }
    close(fd);
    return val;
}

static void write_msr(int cpu, uint32_t reg, uint64_t val)
{
    char path[64];
    snprintf(path, sizeof path, "/dev/cpu/%d/msr", cpu);
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        perror("open /dev/cpu/N/msr for write (need root and `modprobe msr`)");
        exit(1);
    }
    if (pwrite(fd, &val, sizeof val, reg) != (ssize_t)sizeof val) {
        perror("pwrite msr");
        exit(1);
    }
    close(fd);
}

/* Set DDPD to `disable` on every logical CPU (read-modify-write). */
static void toggle_ddp_all_cpus(int disable)
{
    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    for (int c = 0; c < ncpu; c++) {
        uint64_t v  = read_msr(c, IA32_SPEC_CTRL);
        uint64_t nv = disable ? (v |  (1ULL << DDPD_BIT))
                              : (v & ~(1ULL << DDPD_BIT));
        write_msr(c, IA32_SPEC_CTRL, nv);
        uint64_t chk = read_msr(c, IA32_SPEC_CTRL);
        printf("cpu %2d: 0x%016lx -> 0x%016lx  DDPD=%d (%s)\n",
               c, v, chk, (int)((chk >> DDPD_BIT) & 1),
               ((chk >> DDPD_BIT) & 1) ? "DISABLED" : "ENABLED");
    }
    printf("=> DDP now %s on all %ld CPUs. Re-run `get` after your test to "
           "confirm the kernel didn't clear it.\n",
           disable ? "DISABLED" : "ENABLED", ncpu);
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <logical_cpu_id> [get|disable|enable]\n",
                argv[0]);
        return 1;
    }
    int cpu = atoi(argv[1]);
    const char *cmd = (argc == 3) ? argv[2] : "get";

    if (strcmp(cmd, "disable") == 0) { toggle_ddp_all_cpus(1); return 0; }
    if (strcmp(cmd, "enable")  == 0) { toggle_ddp_all_cpus(0); return 0; }
    if (strcmp(cmd, "get") != 0) {
        fprintf(stderr, "unknown command '%s' (use get|disable|enable)\n", cmd);
        return 1;
    }

    pin_cpu(cpu);   /* CPUID then reflects this core (matters on hybrid parts) */

    /* Is the DDPD control exposed? */
    unsigned a, b, c, d;
    __cpuid_count(7, 2, a, b, c, d);
    int ddpd_supported = (d >> 3) & 1;
    printf("CPUID.(7,2).EDX = 0x%08x  -> DDPD_U supported: %s\n",
           d, ddpd_supported ? "YES" : "NO");

    /* Current DDP enable state */
    uint64_t spec = read_msr(cpu, IA32_SPEC_CTRL);
    int ddpd = (spec >> DDPD_BIT) & 1;
    printf("IA32_SPEC_CTRL (0x48) = 0x%016lx  -> DDPD(bit8) = %d\n", spec, ddpd);
    printf("=> DDP is %s\n", ddpd ? "DISABLED" : "ENABLED (active)");

    if (!ddpd_supported)
        printf("note: DDPD control not enumerated; the SPEC_CTRL bit may be "
               "reserved/unused on this CPU.\n");
    return 0;
}

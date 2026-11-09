#define _GNU_SOURCE
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/resource.h>
#include <x86intrin.h>

/* Adapted from Peek-a-Walk util/util.c, with a pin-verification print.
 *
 * Only the priority boost needs privilege. Pinning itself (sched_setaffinity)
 * works as a normal user, so these tools do NOT require sudo -- running
 * unprivileged just means slightly noisier timing. */
void pin_cpu(size_t core_ID)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_ID, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0) {
        printf("Unable to Set Affinity\n");
        exit(EXIT_FAILURE);
    }

    /* verify the pin actually took effect */
    int actual = sched_getcpu();
    if (actual == (int)core_ID)
        printf("[pin_cpu] OK: pinned to logical CPU %d\n", actual);
    else
        printf("[pin_cpu] WARNING: requested CPU %zu but running on CPU %d\n",
               core_ID, actual);

    /* optional: highest priority to avoid preemption noise. Needs root or
     * CAP_SYS_NICE; harmless (just noisier) if it fails. */
    if (setpriority(PRIO_PROCESS, 0, -20) == -1)
        printf("[pin_cpu] note: no nice(-20) boost (needs sudo or "
               "`sudo setcap cap_sys_nice+ep <binary>`); timing may be noisier\n");
    else
        printf("[pin_cpu] nice(-20) priority boost applied\n");
}

/* Ported verbatim from Peek-a-Walk INTEL_DDP/util/util.c: time a single load of
 * *addr, noise-subtracted (net cycles). __trash threads a data dependency. */
uint64_t time_access(uint64_t *addr, uint64_t __trash) {
    uint64_t T0, T1, time;

    // enforce ordering
    _mm_mfence();
    _mm_lfence();

    // measuring timing
    T0 = __rdtscp( (unsigned int *)&__trash );

    // serialize the rdtscp
    _mm_lfence();

    __trash = *addr;

    // serialize the rdtscp
    _mm_lfence();

    // measuring timing
    T1 = __rdtscp( (unsigned int *)&__trash );

    // serialize
    _mm_lfence();
    time = T1 - T0;

    // noise measurements
    _mm_mfence();
    _mm_lfence();
    T0 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();
    _mm_lfence();
    T1 = __rdtscp( (unsigned int *)&__trash );
    _mm_lfence();

    // subtract out noise
    return (time - (T1 - T0)) | (__trash & MSB_MASK);
}

/* Ported verbatim from Peek-a-Walk util/util.c. */
uint64_t clflush(uint64_t *addr, uint64_t __trash) {
    _mm_mfence();

    _mm_clflush(addr);

    return (__trash | (uint64_t)addr) & (MSB_MASK - 1);
}

/* Ported verbatim from Peek-a-Walk util/util.c. */
uint64_t c_sleep(uint64_t duration, uint64_t __trash) {
    // mulitplication to drive up time
    __trash = __trash & MSB_MASK; // __trash = 0
    for(int i = 1; i < (duration | (__trash & MSB_MASK)); i++)
        __trash = (__trash * duration * i) & MSB_MASK;
    return __trash;
}

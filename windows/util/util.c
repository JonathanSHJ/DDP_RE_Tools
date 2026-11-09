#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <x86intrin.h>

/* Windows (MinGW) twin of the Linux util pin_cpu. Same intent: pin this thread
 * to one logical CPU and raise priority so timing is quieter.
 *
 * Only the priority boost is privileged-ish (REALTIME needs it or it quietly
 * downgrades). Affinity works as a normal user, so these tools do NOT need an
 * elevated prompt -- running unprivileged just means slightly noisier timing. */
void pin_cpu(size_t core_ID)
{
    DWORD_PTR mask = (DWORD_PTR)1 << core_ID;
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        printf("Unable to Set Affinity\n");
        exit(EXIT_FAILURE);
    }

    /* the affinity change takes effect at the next schedule; yield so the
     * verification read below sees the new CPU */
    Sleep(0);

    /* verify the pin actually took effect */
    DWORD actual = GetCurrentProcessorNumber();
    if (actual == (DWORD)core_ID)
        printf("[pin_cpu] OK: pinned to logical CPU %lu\n", actual);
    else
        printf("[pin_cpu] WARNING: requested CPU %zu but running on CPU %lu\n",
               core_ID, actual);

    /* optional: highest priority to avoid preemption noise. */
    if (SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS) &&
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL))
        printf("[pin_cpu] realtime priority boost applied\n");
    else
        printf("[pin_cpu] note: no realtime priority boost (needs admin); "
               "timing may be noisier\n");
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

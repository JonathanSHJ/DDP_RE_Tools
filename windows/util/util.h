#ifndef DMP_RE_UTIL
#define DMP_RE_UTIL

#include <stddef.h>
#include <stdint.h>

#define MSB_MASK 0x8000000000000000ULL
#define MB(x) ((x) * 1024 * 1024)

/* Pin the calling thread to a logical CPU and raise its priority.
 * Verifies the pin with GetCurrentProcessorNumber() and prints the result. */
void pin_cpu(size_t core_ID);

/* Time a single load of *addr, noise-subtracted (net cycles). __trash threads a
 * data dependency. Same implementation as Peek-a-Walk INTEL_DDP/util. */
uint64_t time_access(uint64_t *addr, uint64_t __trash);

/* Flush addr from cache (clflush + mfence). __trash carries a dependency to
 * keep the compiler from reordering it away; return value threads it on. */
uint64_t clflush(uint64_t *addr, uint64_t __trash);

/* Busy-wait proportional to duration; __trash threads a data dependency so the
 * loop can't be optimized out. Ported verbatim from Peek-a-Walk util. */
uint64_t c_sleep(uint64_t duration, uint64_t __trash);

#endif

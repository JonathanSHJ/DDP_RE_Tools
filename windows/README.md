# windows/

Windows-only port of the DMP/DDP pointer-chase tests, kept as close to the
Linux tree (`..`) as possible. This folder is **self-contained**: it has its own
`util/`, and nothing here references the Linux sources.

Ported so far:
- `DMP_chase_aop.c`            -- array-of-pointers DDP prefetch profiler
- `DMP_chase_linked_list_rand.c` -- randomized/scattered linked-list chase
- `indirect.c`                -- indirect (base+index) DDP test; target index
                                 planted one past the trained window

## Build (MinGW-w64)

```
mingw32-make
```

Needs a MinGW-w64 `gcc` on PATH. The microarchitectural core (rdtscp + lfence/
mfence timing, clflush, the `access_aop` inline asm) is identical to the Linux
version; only the OS calls differ:

| Linux               | Windows (here)                         |
|---------------------|----------------------------------------|
| `sched_setaffinity` | `SetThreadAffinityMask`                |
| `sched_getcpu`      | `GetCurrentProcessorNumber`            |
| `setpriority(-20)`  | `SetPriorityClass(REALTIME)`           |
| `mmap(ANON)`        | `VirtualAlloc(MEM_COMMIT\|MEM_RESERVE)` |
| `munmap`            | `VirtualFree(MEM_RELEASE)`             |
| `getpid`            | `GetCurrentProcessId`                  |

## Run

```
DMP_chase_aop.exe <cpu> <aop_len> <dist> <xor 0|1>
DMP_chase_linked_list_rand.exe <cpu> <num_nodes> <dest> <xor 0|1>
indirect.exe <cpu> <training_size> <test_ptr_offset>
```

## Caveats (same hardware reality as the Linux tree)

- Meaningful only on real Intel x86-64 silicon; numbers from a VM/emulator are
  noise.
- `DMP_chase_aop` requests fixed virtual addresses (`BUF_ADDR`, `AOP_ADDR`).
  `VirtualAlloc` will not overwrite an existing reservation, so if an assert
  fires on the `VirtualAlloc`/2 MB-alignment check, just re-run.
- Windows timing is noisier than an isolated Linux core (no `isolcpus` /
  `nohz_full`). Use an idle core, and disable Turbo / fix the power plan in
  BIOS / Windows power settings for stable results.

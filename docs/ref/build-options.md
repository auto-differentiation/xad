# Build Options

XAD's defaults are safe for any code and any number of threads.
A few CMake options make XAD faster for some kinds of code and slower for others,
so they are off by default.
This page says what each one does, when it helps and when it hurts,
and which set to start from for common workloads.
No single set is fastest for every workload: measure on your own code before you settle on one.

## Setting the Options

Pass the options when configuring XAD, e.g. `cmake -S xad -B build -DXAD_USE_STRONG_INLINE=ON`,
or set them before `FetchContent_MakeAvailable(XAD)` in your own project:

```cmake
set(XAD_USE_STRONG_INLINE ON CACHE BOOL "" FORCE)
set(XAD_SKIP_ZERO_PARTIALS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(XAD)
```

The tape, threading and inlining options are written to the generated header `XAD/Config.hpp`, so XAD and your code always agree on them; `XAD_SIMD_OPTION`, link-time optimisation and position-independent code are compile flags, which your own code has to match.
Of the options below, only `XAD_USE_STRONG_INLINE` and `XAD_OUTLINE_RECORDING` may also be defined in your own code
after XAD is built; all others must be set when building XAD.

Build in `Release` mode: `XAD_TAPE_CHECKS` is on by default when configured with `CMAKE_BUILD_TYPE=Debug` and slows
everything down; it is meant for finding variables used on the wrong tape.

## Compiler and Linking

### `XAD_USE_STRONG_INLINE`

Forces inlining of XAD's operators and expression templates.
Helps most code, in particular finance code and code with long expressions,
and was not slower on any workload we measured. It makes compiles slower and larger.

### `CMAKE_INTERPROCEDURAL_OPTIMIZATION` (link-time optimisation)

Standard CMake option. Lets the compiler inline XAD's tape code into your code across
translation units. It helped every workload we measured. Set it to `ON` in your top-level project
before adding XAD, so XAD and your code are both built with it.

### `XAD_SIMD_OPTION`

Instruction set for XAD's own sources (`AVX` by default on x86).
`NATIVE` builds for the host CPU and helps finance code a little;
build your own code with the same flag (e.g. `-march=native`).
The binaries then need a CPU at least as new as the build host.

### `XAD_POSITION_INDEPENDENT_CODE`

On by default, so XAD can be linked into a shared library. It costs a little speed;
turn it off when XAD is linked only into executables.

### `XAD_SHARED_LIBS`

Builds XAD as a shared library. Slower than the default static library; use it only when
several shared modules must share one active tape per thread.

### `XAD_OUTLINE_RECORDING`

Records each operation with one out-of-line call. This is for compile time:
much faster compiles with `XAD_USE_STRONG_INLINE` in large code bases, at some cost in run time.

## Threads

### `XAD_NO_THREADLOCAL`

Makes the active tape a plain global instead of a thread-local variable, which saves a
thread-local lookup on every recorded operation. There is then only one active tape in the
whole process: use it only when a single thread records and computes adjoints.
Running several processes in parallel is fine.

### `XAD_TLS_INITIAL_EXEC`

Multi-threaded alternative on GCC and Clang: the thread-local active tape uses the faster
initial-exec model, which helps when XAD is built as position-independent code.
Leave it off for libraries loaded with `dlopen`, such as Python extensions.

## Tape Layout

These options change what the tape records and how it stores it.
Each helps some code and slows down other code.

### `XAD_SKIP_ZERO_PARTIALS`

Does not record partial derivatives that are exactly zero.
A large win for code with many zero partials: ODE solvers, piecewise functions,
code heavy in `max`, `min` or branches. Slower for dense code where almost no partial is zero,
which includes most finance models, as every partial is checked when recording.
Only partials of the built-in types `float`, `double` and `long double` are skipped;
higher-order partials (e.g. with `AReal<FReal<double>>`) are always recorded.

### `XAD_REDUCED_MEMORY`

Keeps the tape's multipliers and slots in separate arrays (see [`computeAdjoints`](tape.md#computeadjoints)).
Less memory and faster adjoints for large tapes that do not fit in cache, and for
recurrent, sequence-style code. Small recordings run many times can be slower.

### `XAD_COPY_ELISION`

Copies of active variables share the source's slot and record nothing; the last user frees the slot.
Fewer recorded statements for copy-heavy code, such as dense linear algebra written with
temporary matrices. Counting the users of each slot costs a little on every copy.
Register each input once, before copying it: registering it again while a copy taken since is
alive, in that or a later recording, may give wrong derivatives without an error.
It is meant for code that records and sweeps, with `clearAll` or `newRecording` between
recordings. Nothing recorded may outlive `clearAll` except inputs, which are registered again, in
the same order, before any new variable; such an input must not be assigned to in a recording, and
its copies must be gone by `clearAll`. Anything else may give wrong derivatives without an error.
`getPosition`/`resetTo`, `computeAdjointsTo`, checkpoint callbacks and nested recordings are not
supported with it: they may give wrong derivatives without an error.
Combining it with `XAD_LINEAR_SLOTS`, `XAD_TAPE_REUSE_SLOTS` or `XAD_TAPE_TRACK_LIVE_SLOTS` is a compile error.

### `XAD_LINEAR_SLOTS`

Copies share the source's slot, every assignment takes a new slot, and destructors free no slots.
Fast recording for copy-heavy code; the number of slots grows with every assignment,
so it suits recordings of moderate size. Excludes `XAD_COPY_ELISION`.

### `XAD_HUGE_PAGES`

On Linux, backs tape chunks of 2 MiB or more with transparent huge pages.
Helps a lot for large tapes recorded once, as there are fewer page faults and TLB misses.
On hosts with fragmented memory, the kernel can stall while it looks for huge pages.
Needs transparent huge pages set to `madvise` or `always`.

### `XAD_TAPE_REUSE_SLOTS`

Reuses any freed slot. This is for memory, not speed: recording and freeing are slower.
Use it when you record many times on one tape with
[`newRecording`](tape.md#newrecording) and the tape's memory keeps growing.

### `XAD_TAPE_TRACK_LIVE_SLOTS`

With [`resetTo`](tape.md#resetto), variables registered after the position that are still alive
keep their slots. Needed when such a result is kept across resets, e.g. a path-wise
Monte Carlo result kept across paths. Also, [`newRecording`](tape.md#newrecording) reuses the
previous recording's slots once none of its variables is alive. Recording is slightly slower.

## Recommended Starting Sets

For every workload, start from a `Release` build with `XAD_USE_STRONG_INLINE` and
link-time optimisation, and add `XAD_NO_THREADLOCAL` if only one thread uses XAD.
Then add the tape options below, and keep a change only if it is faster on your code.

| Workload | Add on top |
|---|---|
| Finance: Monte Carlo pricing, XVA, LIBOR market models, stochastic volatility | `XAD_SIMD_OPTION=NATIVE` if it runs on hosts like the build host; measure before enabling any tape option |
| ODE solvers, piecewise or `max`/`min`-heavy code | `XAD_SKIP_ZERO_PARTIALS` alone |
| Dense linear-algebra-style code | `XAD_COPY_ELISION`, `XAD_SKIP_ZERO_PARTIALS` and `XAD_REDUCED_MEMORY` |
| Recurrent, sequence-style models | `XAD_REDUCED_MEMORY` |
| One very large tape, recorded once | `XAD_HUGE_PAGES`, and try `XAD_REDUCED_MEMORY` |
| Many recordings on one tape, memory growing | `XAD_TAPE_TRACK_LIVE_SLOTS`, or `XAD_TAPE_REUSE_SLOTS` (for memory; slower) |
| Large code base where compile time matters | `XAD_OUTLINE_RECORDING` |

Options that help alone do not always help together:
for ODE-style code, adding `XAD_REDUCED_MEMORY` or `XAD_COPY_ELISION` to `XAD_SKIP_ZERO_PARTIALS`
makes it slower.
The tape options are off by default because on dense finance code
`XAD_SKIP_ZERO_PARTIALS`, `XAD_REDUCED_MEMORY` and `XAD_COPY_ELISION` can be slower than the defaults.

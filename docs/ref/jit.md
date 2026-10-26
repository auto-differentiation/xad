# JIT (optional)

XAD provides **optional** JIT recording/execution support that can be compiled into the library.
When enabled, XAD expressions can be recorded into a compact `JITGraph` and executed via a pluggable `JITBackend`.

!!! note "Compile-time feature flag"

    The JIT API is only available when XAD is compiled with `XAD_ENABLE_CODEGEN`.
    This is a compile-time configuration (it is compiled into the library).
    With it, the top quarter of the slots are JIT variables, passive on tapes (and tape variables are
    passive in JIT recordings), so a tape holds up to 3·2^30 slots and a JIT graph up to 2^30 - 2 nodes.
    Beyond them, `computeAdjoints` and adding a node throw `OutOfRange`.

## Main building blocks

### `JITCompiler`

`#!c++ template <class Real, std::size_t N = 1> class JITCompiler;`

Records computations into a `JITGraph`, compiles them using a backend, and executes forward/adjoint computations.

See: [JITCompiler](jit-compiler.md)

### `JITGraph`

Defines the graph representation (`JITOpCode`, `JITNode`, and `JITGraph`).

See: [JITGraph](jit-graph.md)

### `JITBackend` and `JITGraphInterpreter`

`JITBackend` is the abstract interface for executing a `JITGraph`.
`JITGraphInterpreter` is the reference implementation that interprets the graph.

See: [JIT Backend Interface](jit-backend.md)

### `ABool`

Trackable boolean helper used for JIT-recorded comparisons and conditional selection (`If`).

See: [ABool](jit-abool.md)

# ABool (JIT)

## Overview

`#!c++ template <class Scalar, std::size_t N = 1> class ABool;`

`ABool` is a helper type for **trackable booleans** in JIT-enabled builds. It supports:

- comparisons that can be recorded into a [`JITGraph`](jit-graph.md) when a [`JITCompiler`](jit-compiler.md) is active
- conditional selection via `If(trueVal, falseVal)`

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

## Conditional selection

### `If`

`#!c++ template <class AD> AD If(const AD& trueVal, const AD& falseVal) const;`

If the `ABool` is passive (no JIT recording), this behaves like a normal conditional expression and returns the selected value.

If a JIT compiler is active, the selection is recorded as an `If` node in the graph.

## Comparisons

`ABool` integrates with the comparison helpers (e.g. `less`, `greater`, …) so that comparisons can either:

- remain passive (normal `bool` behaviour) if no JIT compiler is active, or
- record comparison nodes when JIT is active


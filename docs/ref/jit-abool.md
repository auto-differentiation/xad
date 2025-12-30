# ABool (JIT)

## Overview

`#!c++ template <class Scalar, std::size_t N = 1> class ABool;`

`ABool` is a helper type for **trackable booleans** in JIT-enabled builds. It supports:

- comparisons that can be recorded into a [`JITGraph`](jit-graph.md) when a [`JITCompiler`](jit-compiler.md) is active
- conditional selection via `If(trueVal, falseVal)`

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

## Convenience Typedef

`#!c++ using ADBool = ABool<double, 1>;`

Convenience alias for the common case of `double` scalar type with scalar mode (N=1).

## Constructor

`#!c++ explicit ABool(bool b = false);`

Constructs an `ABool` from a plain `bool`. The result has no JIT slot (passive only).

## Member Functions

| Function          | Returns     | Description                                            |
|-------------------|-------------|--------------------------------------------------------|
| `passive()`       | `bool`      | Returns the passive (C++) boolean value                |
| `slot()`          | `slot_type` | Returns the JIT graph slot ID, or INVALID_SLOT if none |
| `hasSlot()`       | `bool`      | Returns true if this ABool has a valid JIT slot        |
| `operator bool()` | `bool`      | Implicit conversion to bool (returns passive value)    |

## Conditional Selection

| Function                             | Description                                         |
|--------------------------------------|-----------------------------------------------------|
| `If(trueVal, falseVal)`              | Member: returns trueVal if true, falseVal otherwise |
| `ABool::If(cond, trueVal, falseVal)` | Static: equivalent to `cond.If(trueVal, falseVal)`  |

When a JIT compiler is active and the ABool has a slot, an `If` node is recorded in the graph, allowing the branch to vary at runtime. If no JIT is active (e.g., Tape mode), this behaves like a normal conditional.

## Comparison Functions

The following free functions in the `xad` namespace create `ABool` values from comparisons. When a JIT compiler is active, comparison nodes are recorded in the graph. Otherwise, they return a passive `ABool`.

All functions have two overloads: `(AReal, AReal)` and `(AReal, Scalar)`.

| Function             | Description              | Equivalent |
|----------------------|--------------------------|------------|
| `less(a, b)`         | Returns true if a < b    | `a < b`    |
| `greater(a, b)`      | Returns true if a > b    | `a > b`    |
| `lessEqual(a, b)`    | Returns true if a <= b   | `a <= b`   |
| `greaterEqual(a, b)` | Returns true if a >= b   | `a >= b`   |

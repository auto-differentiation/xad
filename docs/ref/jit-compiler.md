# JITCompiler

## Overview

`#!c++ template <class Real, std::size_t N = 1> class JITCompiler;`

`JITCompiler` is a *tape-like* recorder/executor for JIT-enabled builds. It:

- records computations into a [`JITGraph`](jit-graph.md)
- compiles the graph using a pluggable [`JITBackend`](jit-backend.md)
- provides forward execution and adjoint propagation

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

## Typical usage

    using AD = xad::AReal<double, 1>;

    xad::JITCompiler<double, 1> jit;

    AD x = 2.0;
    AD y = 3.0;

    jit.registerInput(x);
    jit.registerInput(y);

    jit.newRecording();
    AD z = x * y + x;
    jit.registerOutput(z);

    jit.compile();

    double out = 0.0;
    jit.forward(&out, 1);

## Recording control (TLS)

`JITCompiler` mirrors the tape pattern: a thread-local “active” compiler can be set, so that XAD expression construction/assignment can record into the active JIT compiler when no tape is active.

### `isActive`

`#!c++ bool isActive() const`

Checks whether this instance is the active compiler for the current thread.

### `getActive`

`#!c++ static JITCompiler* getActive()`

Returns the active compiler for the current thread, or `#!c++ nullptr` if none is active.

### `activate` / `deactivate` / `deactivateAll`

`#!c++ void activate()`, `#!c++ void deactivate()`, `#!c++ static void deactivateAll()`

Manage the thread-local active compiler pointer.

## Graph and backend

### `getGraph`

`#!c++ JITGraph& getGraph()`

Access the recorded graph (mainly for backends and debugging).

### `setBackend`

`#!c++ void setBackend(std::unique_ptr<JITBackend> backend)`

Replaces the execution backend (requires recompilation of the current graph).

### `compile`

`#!c++ void compile()`

Compiles the currently recorded graph with the current backend.

## Inputs/outputs

### `registerInput` / `registerInputs`

Registers independent variables as graph inputs.

### `registerOutput` / `registerOutputs`

Registers dependent variables as graph outputs.

### `newRecording`

`#!c++ void newRecording()`

Starts a new recording using the existing registered inputs.

## Execution

### `forward`

`#!c++ void forward(double* outputs, std::size_t numOutputs)`

Executes the forward pass and fills the output array.

### `computeAdjoints`

`#!c++ void computeAdjoints()`

Computes adjoints for the currently recorded graph (after seeding output derivatives).

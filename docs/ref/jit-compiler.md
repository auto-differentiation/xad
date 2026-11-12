# JITCompiler

## Overview

`#!c++ template <class Real, std::size_t N = 1> class JITCompiler;`

Only scalar mode is supported (`N == 1`); other values of `N` fail to compile.

`JITCompiler` is a *tape-like* recorder/executor for JIT-enabled builds. It:

- records computations into a [`JITGraph`](jit-graph.md)
- compiles the graph using a pluggable [`JITBackend`](jit-backend.md)
- provides forward execution and adjoint propagation

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_CODEGEN`,
    which needs the separately licensed xad-codegen library.

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
    jit.forward(&out);              // out = 8

    jit.setDerivative(z.getSlot(), 1.0);
    jit.computeAdjoints();
    double dzdx = jit.getDerivative(x.getSlot());   // 4

    x = 3.0;                        // run again with new input values
    jit.clearDerivatives();
    jit.forward(&out);              // out = 12

## Construct

`#!c++ explicit JITCompiler(bool activate = true)` uses the [`JITGraphInterpreter`](jit-backend.md#jitgraphinterpreter) backend.

`#!c++ explicit JITCompiler(std::unique_ptr<JITBackend<Real>> backend, bool activate = true)` uses the given backend.

With `activate`, the new compiler becomes the active one for the thread.
A tape that is active then is deactivated while the compiler records,
and activated again when the compiler is destroyed (unless another tape or compiler is active by then).
A `JITCompiler` can be moved, but not copied.

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

`#!c++ void setBackend(std::unique_ptr<JITBackend<Real>> backend)`

Replaces the execution backend (requires recompilation of the current graph).

### `compile`

`#!c++ void compile()`

Compiles the currently recorded graph with the current backend.

## Inputs/outputs

### `registerInput` / `registerInputs`

`#!c++ void registerInput(active_type& inp)`,
`#!c++ template <class Inner> void registerInputs(std::vector<Inner>& v)`,
`#!c++ template <class It> void registerInputs(It first, It last)`

Registers independent variables as graph inputs, in this order.
The variables must stay alive: `forward` and `computeAdjoints` read their current values.

### `registerOutput` / `registerOutputs`

`#!c++ void registerOutput(active_type& outp)`,
`#!c++ template <class Inner> void registerOutputs(std::vector<Inner>& v)`,
`#!c++ template <class It> void registerOutputs(It first, It last)`

Registers dependent variables as graph outputs, in this order.
An output that does not depend on the inputs is recorded as a constant.

### `newRecording`

`#!c++ void newRecording()`

Starts a new recording using the existing registered inputs, which must stay alive; `clearAll` drops them.

## Execution

### `vectorWidth` / `numInputs` / `numOutputs`

`#!c++ std::size_t vectorWidth() const`, `#!c++ std::size_t numInputs() const`, `#!c++ std::size_t numOutputs() const`

The backend's number of parallel evaluations per execution (lanes), and the number of inputs and outputs
of the compiled graph.

### `forward`

`#!c++ void forward(Real* outputs)`

Executes the forward pass with the current values of the registered inputs,
and writes the `numOutputs()` output values to `outputs`.
With a backend whose `vectorWidth()` is more than 1, every lane gets the same inputs and lane 0 is returned.

### `computeAdjoints`

`#!c++ void computeAdjoints()`

Runs the graph with the current values of the registered inputs and propagates the output adjoints
set with `setDerivative` (or `derivative`) back to the inputs, whose derivatives can then be read with `getDerivative`.

### `setInput`

`#!c++ void setInput(std::size_t inputIndex, const Real* values)`

Sets the values of one input directly on the backend: `vectorWidth()` values, one per lane.
For use with `forwardAndBackward`; `forward` and `computeAdjoints` overwrite them with the registered inputs' values.

### `forwardAndBackward`

`#!c++ void forwardAndBackward(Real* outputs, Real* inputGradients)`

Runs the backend's forward and backward passes on the inputs set with `setInput`, with all output adjoints 1.
`outputs` takes `numOutputs() * vectorWidth()` values and `inputGradients` `numInputs() * vectorWidth()` values.
It does not change the derivatives stored in the compiler.

## Derivatives

The slot `s` is that of a recorded variable (`x.getSlot()`) or a graph node.

### `derivative`

`#!c++ derivative_type& derivative(slot_type s)` gets a reference to the derivative of `s`.
The const version returns 0 for a slot without a stored derivative.

### `getDerivative`

`#!c++ derivative_type getDerivative(slot_type s) const` returns the derivative of `s` (0 if none is stored).

### `setDerivative`

`#!c++ void setDerivative(slot_type s, const derivative_type& d)` sets the derivative of `s`,
e.g. to seed an output adjoint before `computeAdjoints`.

### `clearDerivatives`

`#!c++ void clearDerivatives()` resets all stored derivatives to 0, keeping the graph.

### `clearAll`

`#!c++ void clearAll()` clears the graph, the registered inputs and the derivatives, and resets the backend.

## Status

### `getMemory`

`#!c++ std::size_t getMemory() const` returns an estimate of the memory in bytes taken by the graph and the derivatives.

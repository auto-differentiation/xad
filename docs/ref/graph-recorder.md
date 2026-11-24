# GraphRecorder

## Overview

`#!c++ template <class Real, std::size_t N = 1> class GraphRecorder;`

Only scalar mode is supported (`N == 1`); other values of `N` fail to compile.

`GraphRecorder` is a *tape-like* recorder/executor for graph-enabled builds. It:

- records computations into a [`Graph`](graph-class.md)
- compiles the graph using a pluggable [`GraphBackend`](graph-backend.md)
- provides forward execution and adjoint propagation

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_GRAPH=ON`
    (see [Build Options](build-options.md#xad_enable_graph)).

## Typical usage

    using AD = xad::AReal<double, 1>;

    xad::GraphRecorder<double, 1> recorder;

    AD x = 2.0;
    AD y = 3.0;

    recorder.registerInput(x);
    recorder.registerInput(y);

    recorder.newRecording();
    AD z = x * y + x;
    recorder.registerOutput(z);

    recorder.compile();

    double out = 0.0;
    recorder.forward(&out);              // out = 8

    recorder.setDerivative(z.getSlot(), 1.0);
    recorder.computeAdjoints();
    double dzdx = recorder.getDerivative(x.getSlot());   // 4

    x = 3.0;                        // run again with new input values
    recorder.clearDerivatives();
    recorder.forward(&out);              // out = 12

## Construct

`#!c++ explicit GraphRecorder(bool activate = true)` uses the [`GraphInterpreter`](graph-backend.md#graphinterpreter) backend.

`#!c++ explicit GraphRecorder(std::unique_ptr<GraphBackend<Real>> backend, bool activate = true)` uses the given backend.

With `activate`, the new recorder becomes the active one for the thread.
A tape that is active then is deactivated while the recorder records,
and activated again when the recorder is destroyed (unless another tape or recorder is active by then).
A `GraphRecorder` can be moved, but not copied.

## Recording control (TLS)

`GraphRecorder` mirrors the tape pattern: a thread-local “active” recorder can be set, so that XAD expression construction/assignment can record into the active graph recorder when no tape is active.

### `isActive`

`#!c++ bool isActive() const`

Checks whether this instance is the active recorder for the current thread.

### `getActive`

`#!c++ static GraphRecorder* getActive()`

Returns the active recorder for the current thread, or `#!c++ nullptr` if none is active.

### `activate` / `deactivate` / `deactivateAll`

`#!c++ void activate()`, `#!c++ void deactivate()`, `#!c++ static void deactivateAll()`

Manage the thread-local active recorder pointer.

## Graph and backend

### `getGraph`

`#!c++ Graph& getGraph()`

Access the recorded graph (mainly for backends and debugging).

### `setBackend`

`#!c++ void setBackend(std::unique_ptr<GraphBackend<Real>> backend)`

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
It does not change the derivatives stored in the recorder.

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

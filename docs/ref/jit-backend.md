## JIT Backend Interface

### Overview

JIT backends execute a recorded [`JITGraph`](jit-graph.md). XAD exposes:

- `JITBackend`: abstract execution interface
- `JITGraphInterpreter`: reference backend that interprets the graph

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

### `JITBackend`

`#!c++ class JITBackend`

Backends are responsible for:

- compiling a `JITGraph` into an executable form (optional)
- executing forward evaluation
- optionally executing forward+backward to produce input adjoints

#### Main virtual functions

##### `compile`

`#!c++ virtual void compile(const JITGraph& graph) = 0;`

Prepare the backend for executing the given graph.

##### `forward`

`#!c++ virtual void forward(const JITGraph& graph, const double* inputs, std::size_t numInputs, double* outputs, std::size_t numOutputs) = 0;`

Run a forward pass.

##### `forwardAndBackward`

`#!c++ virtual void forwardAndBackward(const JITGraph& graph, const double* inputs, std::size_t numInputs, const double* outputAdjoints, std::size_t numOutputs, double* outputs, double* inputAdjoints) = 0;`

Run forward and backward, producing input adjoints for the given output adjoints.

##### `reset`

`#!c++ virtual void reset() = 0;`

Reset any cached backend state.

### `JITGraphInterpreter`

`#!c++ class JITGraphInterpreter : public JITBackend`

Reference backend that interprets the graph directly. It is mainly intended as:

- a correctness reference implementation
- a simple fallback backend
- a baseline for testing and debugging

# Graph Backend Interface

## Overview

Graph backends execute a recorded [`Graph`](graph-class.md). XAD exposes:

- `GraphBackend<Scalar>`: abstract execution interface
- `GraphInterpreter<Scalar>`: reference backend that interprets the graph

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_GRAPH=ON`
    (see [Build Options](build-options.md#xad_enable_graph)).

## `GraphBackend`

`#!c++ template <class Scalar> class GraphBackend`

Abstract base class for graph execution backends. The `Scalar` template parameter specifies the floating-point type used for computations and must match the type used by `GraphRecorder<Real, N>`:

- `GraphBackend<double>` for use with `GraphRecorder<double, N>`
- `GraphBackend<float>` for use with `GraphRecorder<float, N>`

Most applications use `double` precision. The `float` option is available for performance-sensitive applications where single precision is sufficient.

Backends are responsible for:

- compiling a `Graph` into an executable form (optional)
- executing forward evaluation
- optionally executing forward+backward to produce input adjoints

### Main virtual functions

#### `compile`

`#!c++ virtual void compile(const Graph& graph) = 0;`

Prepare the backend for executing the given graph.

#### `vectorWidth`

`#!c++ virtual std::size_t vectorWidth() const = 0;`

Returns the number of parallel evaluations per execution (1 for scalar backends, 4 for AVX2).

#### `numInputs` / `numOutputs`

`#!c++ virtual std::size_t numInputs() const = 0;`
`#!c++ virtual std::size_t numOutputs() const = 0;`

Return the number of inputs/outputs in the compiled graph.

#### `setInput`

`#!c++ virtual void setInput(std::size_t inputIndex, const Scalar* values) = 0;`

Set input values for an input variable. The `values` array must contain `vectorWidth()` elements.

#### `forward`

`#!c++ virtual void forward(Scalar* outputs) = 0;`

Run a forward pass. The `outputs` array must have space for `numOutputs() * vectorWidth()` elements.

#### `forwardAndBackward`

`#!c++ virtual void forwardAndBackward(Scalar* outputs, Scalar* inputGradients) = 0;`

Run forward and backward passes combined, with all output adjoints seeded to 1. The `outputs` array must have space for `numOutputs() * vectorWidth()` elements, and `inputGradients` must have space for `numInputs() * vectorWidth()` elements.

#### `forwardAndBackwardSeeded`

`#!c++ virtual void forwardAndBackwardSeeded(Scalar* outputs, const Scalar* outputAdjoints, Scalar* inputGradients);`

As `forwardAndBackward`, but the backward pass starts from the given output adjoints
(`numOutputs() * vectorWidth()` values). `GraphRecorder::computeAdjoints` uses it.

It is not pure virtual: the default implementation calls `forwardAndBackward` and scales the gradients,
which works only when all output adjoints are equal; otherwise it throws `#!c++ std::runtime_error`.
Backends that support different output adjoints override it, as `GraphInterpreter` does.

#### `reset`

`#!c++ virtual void reset() = 0;`

Reset any cached backend state.

## `GraphInterpreter`

`#!c++ template <class Scalar> class GraphInterpreter : public GraphBackend<Scalar>`

Reference backend that interprets the graph directly, one evaluation at a time (`vectorWidth()` is 1). It is mainly intended as:

- a correctness reference implementation
- a simple fallback backend
- a baseline for testing and debugging

### Example Usage

For double backend:

    auto backend = std::make_unique<xad::GraphInterpreter<double>>();
    xad::GraphRecorder<double, 1> recorder(std::move(backend));
    // ... record graph ...
    recorder.compile();

For float backend:

    auto backend = std::make_unique<xad::GraphInterpreter<float>>();
    xad::GraphRecorder<float, 1> recorder(std::move(backend));
    // ... record graph ...
    recorder.compile();

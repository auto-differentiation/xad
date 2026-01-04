# JIT Backend Interface

## Overview

JIT backends execute a recorded [`JITGraph`](jit-graph.md). XAD exposes:

- `JITBackend`: abstract execution interface
- `JITGraphInterpreter`: reference backend that interprets the graph

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

## `JITBackend`

`#!c++ class JITBackend`

Backends are responsible for:

- compiling a `JITGraph` into an executable form (optional)
- executing forward evaluation
- optionally executing forward+backward to produce input adjoints

### Main virtual functions

#### `compile`

`#!c++ virtual void compile(const JITGraph& graph) = 0;`

Prepare the backend for executing the given graph.

#### `vectorWidth`

`#!c++ virtual std::size_t vectorWidth() const = 0;`

Returns the number of parallel evaluations per execution (1 for scalar backends, 4 for AVX2).

#### `numInputs` / `numOutputs`

`#!c++ virtual std::size_t numInputs() const = 0;`
`#!c++ virtual std::size_t numOutputs() const = 0;`

Return the number of inputs/outputs in the compiled graph.

#### `setInput`

`#!c++ virtual void setInput(std::size_t inputIndex, const double* values) = 0;`

Set input values for an input variable. The `values` array must contain `vectorWidth()` elements.

#### `forward`

`#!c++ virtual void forward(double* outputs) = 0;`

Run a forward pass. The `outputs` array must have space for `numOutputs() * vectorWidth()` elements.

#### `forwardAndBackward`

`#!c++ virtual void forwardAndBackward(double* outputs, double* inputGradients) = 0;`

Run forward and backward passes combined. The `outputs` array must have space for `numOutputs() * vectorWidth()` elements, and `inputGradients` must have space for `numInputs() * vectorWidth()` elements.

#### `reset`

`#!c++ virtual void reset() = 0;`

Reset any cached backend state.

## `JITGraphInterpreter`

`#!c++ class JITGraphInterpreter : public JITBackend`

Reference backend that interprets the graph directly. It is mainly intended as:

- a correctness reference implementation
- a simple fallback backend
- a baseline for testing and debugging

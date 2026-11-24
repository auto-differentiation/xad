# Graph

## Overview

`Graph` is the recorded representation of an XAD expression DAG in graph-enabled builds.

It consists of:

- `GraphOpCode`: operation codes (add, mul, sin, comparisons, `If`, …)
- `GraphNode`: one recorded node (opcode + operands + immediate + flags)
- `Graph`: the full node list plus constant/input/output metadata

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_GRAPH=ON`
    (see [Build Options](build-options.md#xad_enable_graph)).

## `GraphOpCode`

`#!c++ enum class GraphOpCode : std::uint16_t`

Defines the opcode values used by graph backends.

## `GraphNode`

`#!c++ struct GraphNode`

Each node contains:

- `op`: opcode (`GraphOpCode`, stored as `#!c++ uint16_t`)
- `a`, `b`, `c`: operand node IDs (stored as `#!c++ uint32_t`)
- `imm`: immediate value, a `#!c++ double` (used for op-specific data, e.g. constant pool index or integer exponent)
- `flags`: node flags (e.g. “active”)

## `Graph`

`#!c++ struct Graph`

### Node storage

Nodes are stored in a chunked container (`ChunkContainer<GraphNode, 262144>`, 8 MiB chunks) to avoid large reallocation/copy spikes when building very large graphs, without reserving much memory for small ones.

### Constant pool

`const_pool` stores unique constants. `addConstant(value)` deduplicates by value and records a `Constant` node that references the pool via `imm`.

### Inputs/outputs

- `input_ids`: node IDs that correspond to inputs
- `output_ids`: node IDs marked as outputs

### Construction helpers

`Graph` provides convenience methods:

- `addInput()`
- `addConstant(double value)`
- `addUnary(...)`, `addBinary(...)`, `addTernary(...)`
- `markOutput(nodeId)`

### Format version

Backends that consume the graph rely on its format: the `GraphOpCode` values and the operand roles of each opcode,
how nodes use `const_pool` and `imm`, the `GraphNode` flags, and the order of `input_ids` and `output_ids`. A change to any of these declares
`#!c++ static constexpr unsigned formatVersion = <n+1>` as a member of `Graph`, so that a backend built for another format
can refuse it. A `Graph` without that declaration has format 1. Appending new opcodes changes none of these and needs
no new format: backends refuse opcodes they don't know.

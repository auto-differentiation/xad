# JITGraph

## Overview

`JITGraph` is the recorded representation of an XAD expression DAG in JIT-enabled builds.

It consists of:

- `JITOpCode`: operation codes (add, mul, sin, comparisons, `If`, …)
- `JITNode`: one recorded node (opcode + operands + immediate + flags)
- `JITGraph`: the full node list plus constant/input/output metadata

!!! note "Compile-time feature flag"

    This API is only available when XAD is compiled with `XAD_ENABLE_JIT`.

## `JITOpCode`

`#!c++ enum class JITOpCode : std::uint16_t`

Defines the opcode values used by JIT backends.

## `JITNode`

`#!c++ struct JITNode`

Each node contains:

- `op`: opcode (`JITOpCode`, stored as `#!c++ uint16_t`)
- `a`, `b`, `c`: operand node IDs (stored as `#!c++ uint32_t`)
- `imm`: immediate value (used for op-specific data, e.g. constant pool index or integer exponent)
- `flags`: node flags (e.g. “active”)

## `JITGraph`

`#!c++ struct JITGraph`

### Node storage

Nodes are stored in a chunked container (`ChunkContainer<JITNode>`) to avoid large reallocation/copy spikes when building very large graphs.

### Constant pool

`const_pool` stores unique constants. `addConstant(value)` deduplicates by value and records a `Constant` node that references the pool via `imm`.

### Inputs/outputs

- `input_ids`: node IDs that correspond to inputs
- `output_ids`: node IDs marked as outputs

### Construction helpers

`JITGraph` provides convenience methods:

- `addInput()`
- `addConstant(double value)`
- `addUnary(...)`, `addBinary(...)`, `addTernary(...)`
- `markOutput(nodeId)`


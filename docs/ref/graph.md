# Graph recording (optional)

XAD provides **optional** graph recording/execution support that can be compiled into the library.
When enabled, XAD expressions can be recorded into a compact `Graph` and executed via a pluggable `GraphBackend`.

!!! note "Compile-time feature flag"

    The graph recording API is only available when XAD is compiled with `XAD_ENABLE_GRAPH=ON`
    (see [Build Options](build-options.md#xad_enable_graph)).
    This is a compile-time configuration (it is compiled into the library).
    With it, the top quarter of the slots are graph variables, passive on tapes (and tape variables are
    passive in graph recordings), so a tape holds up to 3·2^30 slots (2^30 with `XAD_COPY_ELISION`) and a graph up to 2^30 - 2 nodes.
    Beyond them, `computeAdjoints` and adding a node throw `OutOfRange`; with `XAD_COPY_ELISION`, registering the
    variable that would take a slot beyond the limit already throws it.

## Main building blocks

### `GraphRecorder`

`#!c++ template <class Real, std::size_t N = 1> class GraphRecorder;`

Records computations into a `Graph`, compiles them using a backend, and executes forward/adjoint computations.

See: [GraphRecorder](graph-recorder.md)

### `Graph`

Defines the graph representation (`GraphOpCode`, `GraphNode`, and `Graph`).

See: [Graph](graph-class.md)

### `GraphBackend` and `GraphInterpreter`

`GraphBackend` is the abstract interface for executing a `Graph`.
`GraphInterpreter` is the reference implementation that interprets the graph.

See: [Graph Backend Interface](graph-backend.md)

### `ABool`

Trackable boolean helper used for graph-recorded comparisons and conditional selection (`If`).

See: [ABool](graph-abool.md)

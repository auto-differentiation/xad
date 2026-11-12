# Headers and Namespaces

All XAD data types and operations are defined in the `xad` namespace.
For brevity, this namespace has been omitted in the reference section.

XAD provides a general header `XAD/XAD.hpp`,
which includes all headers that are commonly needed to work with XAD.
Typically, this is all that clients need to include.
Among them are:

* `XAD/Complex.hpp` - For using complex numbers with XAD data types
  (see [Complex](complex.md)).
* `XAD/Version.hpp` - The version macros (see [Version Information](version.md)).
* `XAD/Config.hpp` and `XAD/Export.hpp` - Generated when XAD is configured:
  the build options XAD was built with (see [Build Options](build-options.md)),
  and the export macros for a shared library build.

There are three additional headers provided that can be included on demand:

* `XAD/StdCompatibility.hpp` - This header imports the XAD math functions
  into the `std` namespace, for compatibility reasons.
  It enables using constructs like [`#!c++ std::sin(x)`](https://en.cppreference.com/w/cpp/numeric/math/sin) where `x` is an XAD type.
  Additionally, it also specialises [`#!c++ std::numeric_limits`](https://en.cppreference.com/w/cpp/types/numeric_limits) for the XAD data types,
  so that it provides traits similar to the standard floating point types.
  This partially violates the C++ standard's "don't specialize std templates"
  rule but is necessary for integration with other libraries.
* `XAD/Hessian.hpp` - Imports methods for computing the Hessian matrix of a
  single output function into the `xad` namespace.
* `XAD/Jacobian.hpp` - Imports methods for computing the Jacobian matrix of a
  function with multiple inputs and multiple outputs into the `xad` namespace.

## Optional JIT headers

When XAD is compiled with `XAD_ENABLE_CODEGEN`, which needs the separately licensed xad-codegen library,
additional JIT headers are available (`XAD/XAD.hpp` includes `XAD/JITCompiler.hpp` and `XAD/ABool.hpp`):

* `XAD/JITCompiler.hpp` - JIT recorder/executor (see [JITCompiler](jit-compiler.md)).
* `XAD/JITGraph.hpp` - Graph representation (see [JITGraph](jit-graph.md)).
* `XAD/JITBackendInterface.hpp` - Backend interface (see [JIT Backend Interface](jit-backend.md)).
* `XAD/JITGraphInterpreter.hpp` - Reference interpreter backend (see [JIT Backend Interface](jit-backend.md)).
* `XAD/ABool.hpp` - Trackable boolean helper for comparisons/`If` (see [ABool (JIT)](jit-abool.md)).

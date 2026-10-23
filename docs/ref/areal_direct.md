# Direct Adjoint Mode Type `ARealDirect`

## Overview

```c++
template <class Scalar, std::size_t N=1>
struct ARealDirect : public RealDirect<AReal<Scalar, N>, ARealDirect<Scalar, N>>
```

The class `ARealDirect` has the same interface as `AReal`, but does not use expression templates for the operations.
This is to make debugging easier and to fit better with existing code bases with may have
issues with expression templates.

## Compatibility with `AReal`

`ARealDirect` is designed as a drop-in replacement for `AReal` in generic code, for example
large numerical libraries that are templated on (or typedef) their floating point type.
For first-order and higher-order direct mode types (such as `ARealDirect<FRealDirect<double>>`),
XAD provides the same functionality as for `AReal`:

- **Arithmetic and comparisons** with the same type, with any built-in arithmetic type
  (including integer literals, e.g. `x < 0` or `2 * x`), with XAD expressions,
  and with types that implicitly convert to `ARealDirect`
  (for example a type with a conversion operator, like QuantLib's `Null<Real>`).
  As for `AReal`, higher-order types do not mix in arithmetic with their inner active type.
  Compound assignments and `++` / `--` return a reference to `ARealDirect`.
- **Conversions**: implicit conversion from built-in arithmetic types,
  no implicit conversion to `double` or `int` (as for `AReal`),
  explicit conversion to `bool` and (with `XAD_ALLOW_INT_CONVERSION`, the default)
  explicit conversion to integer types via `static_cast`.
  `ARealDirect` is not constructible from unrelated types, so that `std::is_constructible`
  and `std::is_convertible` give the same answers as for `AReal`.
- **Mathematical functions**: all functions listed in [Mathematical Operations](math.md),
  in the `xad` namespace and (after including `XAD/StdCompatibility.hpp`) in the `std`
  namespace, with the same mixed-argument overloads as `AReal`.
  All of them propagate derivatives.
- **`std` compatibility** (`XAD/StdCompatibility.hpp`): `std::numeric_limits`
  (all members, returning the underlying passive type as for `AReal`), `std::hash`,
  `std::to_string`, and the type traits `std::is_floating_point`, `std::is_arithmetic`,
  `std::is_signed`, `std::is_fundamental`, `std::is_scalar` and `std::is_compound`
  (including the C++17 `_v` variable templates).
- **Complex numbers** (`XAD/Complex.hpp`): `std::complex<ARealDirect<T>>` with the same
  operators and functions as `std::complex<AReal<T>>`.
- **Stream operators** `<<` and `>>`.
- **Mode interface**: [`adjd<T>`](interface.md) and the higher-order mode classes use
  `ARealDirect` as the active type, and the [Tape](tape.md) registration functions
  (`registerInput`, `registerOutput`, `registerInputs`, `registerOutputs`) accept
  `ARealDirect` variables and `std::complex<ARealDirect<T>>` directly.
  [Jacobian](jacobian.md) and [Hessian](hessian.md) helpers are available for
  `ARealDirect<T>` and `ARealDirect<FRealDirect<T>>`.

!!! note "See also"

    [AReal](areal.md), [Tape](tape.md), [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

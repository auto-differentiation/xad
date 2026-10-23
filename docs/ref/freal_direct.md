# Direct Forward Mode Type `FRealDirect`

## Overview

```c++
template <class Scalar, std::size_t N = 1>
struct FRealDirect : public RealDirect<FReal<Scalar, N>, FRealDirect<Scalar, N>>
```

The class `FRealDirect` has the same interface as `FReal`, but does not use expression templates for the operations.
This is to make debugging easier and to fit better with existing code bases,
which may have issues integrating with expression templates.

## Compatibility with `FReal`

`FRealDirect` is designed as a drop-in replacement for `FReal` in generic code, for example
large numerical libraries that are templated on (or typedef) their floating point type.
For first-order and higher-order direct mode types (such as `FRealDirect<FRealDirect<double>>`),
XAD provides the same functionality as for `FReal`:

- **Arithmetic and comparisons** with the same type, with any built-in arithmetic type
  (including integer literals, e.g. `x < 0` or `2 * x`), with XAD expressions,
  and with types that implicitly convert to `FRealDirect`
  (for example a type with a conversion operator, like QuantLib's `Null<Real>`).
  As for `FReal`, higher-order types do not mix in arithmetic with their inner active type.
  Compound assignments and `++` / `--` return a reference to `FRealDirect`.
- **Conversions**: implicit conversion from built-in arithmetic types,
  no implicit conversion to `double` or `int` (as for `FReal`),
  explicit conversion to `bool` and (with `XAD_ALLOW_INT_CONVERSION`, the default)
  explicit conversion to integer types via `static_cast`.
  `FRealDirect` is not constructible from unrelated types, so that `std::is_constructible`
  and `std::is_convertible` give the same answers as for `FReal`.
- **Mathematical functions**: all functions listed in [Mathematical Operations](math.md),
  in the `xad` namespace and (after including `XAD/StdCompatibility.hpp`) in the `std`
  namespace, with the same mixed-argument overloads as `FReal`.
  All of them propagate derivatives.
- **`std` compatibility** (`XAD/StdCompatibility.hpp`): `std::numeric_limits`
  (all members, returning the underlying passive type as for `FReal`), `std::hash`,
  `std::to_string`, and the type traits `std::is_floating_point`, `std::is_arithmetic`,
  `std::is_signed`, `std::is_fundamental`, `std::is_scalar`, `std::is_compound` and
  `std::is_trivially_copyable` (including the C++17 `_v` variable templates).
- **Complex numbers** (`XAD/Complex.hpp`): `std::complex<FRealDirect<T>>` with the same
  operators and functions as `std::complex<FReal<T>>`.
- **Stream operators** `<<` and `>>`.
- **Mode interface**: [`fwdd<T>`](interface.md) and the higher-order mode classes use
  `FRealDirect` as the active type.
  [Jacobian](jacobian.md) and [Hessian](hessian.md) helpers are available for
  `FRealDirect<T>` and `FRealDirect<FRealDirect<T>>`.

!!! note "See also"

    [FReal](freal.md), [Tape](tape.md), [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

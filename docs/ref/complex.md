---
description: >
  Specialisation of the std::complex type for XAD active data types.
---

# Complex

## Overview

XAD implements specialisations of [`#!c++ std::complex`](https://en.cppreference.com/w/cpp/numeric/complex) for the XAD active data types
[`AReal`](areal.md) and [`FReal`](freal.md).
They are are provided in the header `XAD/Complex.hpp`,
along with all the mathematical operations defined in the standard.

Note that the complex header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.

#### `std::complex` Specialisations

Both `#!c++ std::complex<AReal<T>>` and `#!c++ std::complex<FReal<T>>` are
provided specialisations for the standard complex type (in `std` namespace),
for adjoint and forward modes respectively.

## Member Functions

All [standard complex member functions](https://en.cppreference.com/w/cpp/numeric/complex) are implemented.

Below are the non-standard additions and changes of the interface only,
using the placeholder `#!c++ XReal<T>` as a placeholder inner type, which can be
`#!c++ FReal<T>` or `#!c++ AReal<T>`.

#### `real`

`#!c++ XReal<T>& complex::real()` returns a reference rather than a copy of the real part,
to allow for easy access and adjusting of derivatives using [`derivative()`](XXX).
This applies to both the modifyable and the `#!c++ const` versions.

#### `imag`

Returns a reference rather than a copy, for both the modifyable and the `#!c++ const` versions.

#### `setDerivative`

`#!c++ void complex::setDerivative(const T& real_derivative, const T& imag_derivative = T())`
sets the derivatives (either $\dot{x}$ or $\bar{x}$) for both the real
and imaginary parts.

#### `setAdjoint`

`#!c++ void complex::setAdjoint(const T& real_derivative, const T& imag_derivative = T())` is an alias for `setDerivative`

#### `getDerivative`

`#!c++ std::complex<T> getDerivative() const` gets the derivatives (either $\dot{x}$ or $\bar{x}$ for both the real and imaginary parts, represented as a complex of the underlying (`#!c++ double`) type.

#### `getAdjoint`

`#!c++ std::complex<T> getAdjoint() const` is an alias for `getDerivative`

## None-Member Functions

#### `derivative`

```c++
template <typename T> 
std::complex<T> derivative(const std::complex<XReal<T> >& z)
```

Returns the adjoints of the `z` variable, represented
as a complex number of the underlying double type.

Note that since the return type is not a reference, setting
derivatives should be done by using the member functions
[`setDerivative`](#setderivative)
or using the [`real`](#real) and  [`imag`](#imag) member functions instead.

#### `value`

```c++
template <typename T> 
std::complex<T> value(const std::complex<XReal<T> >& z)
```

Returns the value of the `z` variable (underlying double type),
represented as a complex number.

#### `real`

```c++
template <typename T>
XReal<T>& real(std::complex<XReal<T> >& z)
```

Access to the real part by reference.

#### `imag`

```c++
template <typename T> 
XReal<T>& imag(std::complex<XReal<T> >& z)
```

Access to the imaginary part by reference.

## Math Operations

All arithmetic operators and mathematical functions in the C++11 standard
have been specialised with the XAD complex data types as well.
This also includes the stream read and write operations.

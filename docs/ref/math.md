# Mathematical Operations

In the following, the data type `T`
refer to arithmetic data types on which the operation is defined mathematically.
This includes the active XAD types as well as the standard passive types.

The functions listed here are defined in the `xad` namespace,
and C++ ADL rules (argument-dependent lookup) typically find these functions
automatically if they are applied to XAD types.
However, for this the calls must be unqualified, i.e. without a namespace specifier.

Alternatively, fully qualified names work as usual (e.g. `#!c++ xad::sin(x)`),
also for `#!c++ float` and `#!c++ double`.

For convenience, if the header `XAD/StdCompatibility.hpp` is included,
the XAD variables are imported into the `std` namespace,
so that existing calls to `#!c++ std::sin` and similar functions are working as expected.

## Absolute Values, Max, Min, and Rounding

#### `abs`

`#!c++ T abs(T x)` computes the absolute value of `x`. Note that for defined second-order derivatives, this computes `#!c++ (x>0)-(x<0)`

#### `max`

`#!c++ T max(T x, T y)` returns the maximum of `x` and `y`.

Note that for well-defined second order derivative, this is implemented as
`#!c++ (x + y + abs(x-y)) / 2`.

#### `fmax`

`#!c++ T fmax(T x, T y)` is synonym for `#!c++ max`

#### `min`

`#!c++ T min(T x, T y)` returns the minimum of `x` and `y`. Note that for well-defined second order derivative, this is implemented as `(x + y - abs(x-y)) / 2`.

#### `fmin`

`#!c++ T fmin(T x, T y)` is a synonym for `#!c++ min`.

#### `floor`

`#!c++ T floor(T x)` rounds towards negative infinity.

#### `ceil`

`#!c++ T ceil(T x)` rounds towards positive infinity.

#### `trunc`

`#!c++ T trunc(T x)` rounds towards 0.

#### `round`

`#!c++ T round(T x)` rounds to the nearest integer value.

#### `lround`

`#!c++ long lround(T x)` is like `#!c++ round`, but converts the result to a `#!c++ long` type.

#### `llround`

`#!c++ long long llround(T x)` is like `#!c++ round`, but converts the result to a `#!c++ long long` type.

#### `fmod`

`#!c++ T fmod(T x, T y)` returns the floating-point remainder of the division operation `x/y`, i.e.exactly the value `x - n*y`, where `n` is `x/y` with its fractional part truncated.

#### `remainder`

`#!c++ T remainder(T x, T y)` calculates the IEEE floating-point remainder of the division operation `x/y`, i.e. exactly the value `x - n*y`, where the value `n` is the integral value nearest the exact value `x/y`.

When `abs(n-x/y) = 0.5`, the value `n` is chosen to be even.

In contrast to `#!c++ fmod`, the returned value is not guaranteed to have the same sign as `x`.

#### `remquo`

`#!c++ T remquo(T x, T y, int* n)` is the same as `#!c++ remainder`, but returns the integer factor `n` in addition.

#### `modf`

`#!c++ T modf(T x, T* iptr)` decomposes `x` into integral and fractional parts, each with the same type and sign as `x`. The integral part is stored in `iptr`.

#### `nextafter`

`#!c++ T nextafter(T from, T to)` returns the next representable value of `from` in the direction of `to`.

Mathmatically, the difference of `from` to the return value is very small.
For derivatives, we therefore consider them both the same and calculate derivative accordingly.

#### `copysign`

`#!c++ T copysign(T x, T y)` copies the sign of the floating point value `y` to the value `x`, correctly treating positive/negative zero, NaN, and Inf values. It uses the function `signbit` internally to determine the sign of `y`.

## Trigonometric Functions

#### `degrees`

`#!c++ T degrees(T x)` converts the given value in radians to degrees.

#### `radians`

`#!c++ T radians(T x)` converts the given value in degrees to radians.

#### `cos`

`#!c++ T cos(T x)` computes the cosine of `x`.

#### `sin`

`#!c++ T sin(T x)` computes the sine of `x`.

#### `tan`

`#!c++ T tan(T x)` computes the tangent of `x`.

#### `asin`

`#!c++ T asin(T x)` computes the inverse sine of `x`.

#### `acos`

`#!c++ T acos(T x)` computes the inverse cosine of `x`

#### `atan`

`#!c++ T atan(T x)` computes the inverse tangent of `x`

#### `atan2`

`#!c++ T atan2(T x, T y)` computes the four-quadrant inverse tangent of a point located at `(x, y)`.

#### `sinh`

`#!c++ T sinh(T x)` computes the hyperbolic sine of `x`.

#### `cosh`

`#!c++ T cosh(T x)` computes the hyperbolic cosine of `x`.

#### `tanh`

`#!c++ T tanh(T x)` computes the hyperbolic tangent of `x`.

#### `asinh`

`#!c++ T asinh(T x)` computes the inverse hyperbolic sine of `x`.

#### `acosh`

`#!c++ T acosh(T x)` computes the inverse hyperbolic cosine of `x`.

#### `atanh`

`#!c++ T atanh(T x)` computes the inverse hyperbolic tangent of `x`.

## Powers, Exponentials, and Logarithms

#### `log`

`#!c++ T log(T x)` computes the natural logarithm of `x`.

#### `log10`

`#!c++ T log10(T x)` computes the base 10 logarithm of `x`.

#### `log2`

`#!c++ T log2(T x)` computes the base 2 logarithm of `x`.

#### `exp`

`#!c++ T exp(T x)` computes the exponential of `x` (base e).

#### `expm1`

`#!c++ T expm1(T x)` computes `exp(x) - 1` with higher precision around 0.

#### `exp2`

`#!c++ T exp2(T x)` computes 2 to the power of `x`.

#### `log1p`

`#!c++ T log1p(T x)` cmputes `log(1 + x)` with higher precision around 0.

#### `sqrt`

`#!c++ T sqrt(T x)` computes the square root of `x`.

#### `cbrt`

`#!c++ T cbrt(T x)` computes the cubic root of `x`.

#### `hypot`

`#!c++ T hypot(T x, T y)` computes `sqrt(x*x + y*y)` without undue overflow or underflow at 
intermediate stages of the computation.

#### `pow`

`#!c++ T pow(T x, T y)` computes `x` to the power of `y`.

#### `ldexp`

`#!c++ T ldexp(T x, int exp)` multiplies `x` by two to the power of `exp`.

#### `frexp`

`#!c++ T frexp(T arg, int* exp)` decomposes the given floating point value arg into a normalised fraction and an integral power of two.

#### `ilogb`

`#!c++ int ilogb(T arg)` returns the integral part of the logarithm of `abs(x)`,
using `FLT_RADIX` as base for the log.

#### `scalbn`

`#!c++ T scalbn(T arg, int exp)` calculates `arg * pow(FLT_RADIX, exp)`.

## Error Functions

***

#### `erf`

`#!c++ T erf(T x)` computes the error function of `x`, if provided by the compiler's math
library.

#### `erfc`

`#!c++ T erfc(T x)` computes the complementary error function of `x`, if provided by the compiler's math library.

## Floating Point Classification

#### `isinf`

`#!c++ bool isinf(T x)` checks if value is infinity (positive or negative).

#### `isnan`

`#!c++ bool isnan(T x)` checks if value is NaN.

#### `isfinite`

`#!c++ bool isfinite(T x)` checks if value is finite (not infinite and not NaN).

#### `signbit`

`#!c++ bool signbit(T x)` returns true if `x` is negative and false otherwise. Also detects sign bit of zeros.

#### `isnormal`

`#!c++ bool isnormal(T x)` checks if the value is a normal floating point number, i.e. not zero, subnormal, infinite, or NaN.

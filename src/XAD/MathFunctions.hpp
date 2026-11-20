/*******************************************************************************

   Importing or declaring of math functions in our namespace.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace xad
{
// import std functions to our namespace, and re-implement as needed
// (all for float and double types)

// functions that always are in cmath
using ::std::abs;
using ::std::acos;
using ::std::asin;
using ::std::atan;
using ::std::atan2;
using ::std::ceil;
using ::std::cos;
using ::std::cosh;
using ::std::exp;
using ::std::fabs;
using ::std::floor;
using ::std::fmod;
using ::std::frexp;
using ::std::hypot;
using ::std::ldexp;
using ::std::log;
using ::std::log10;
using ::std::max;
using ::std::min;
using ::std::modf;
using ::std::pow;
using ::std::sin;
using ::std::sinh;
using ::std::sqrt;
using ::std::tan;
using ::std::tanh;

XAD_INLINE double smooth_abs(double x, double c = 0.001)
{
    if (abs(x) > c)
        return abs(x);
    if (x < 0.0)
        return x * x * (2. / c + x / (c * c));
    else
        return x * x * (2. / c - x / (c * c));
}

XAD_INLINE float smooth_abs(float x, float c = 0.001f)
{
    if (abs(x) > c)
        return abs(x);
    if (x < 0.0f)
        return x * x * (2.f / c + x / (c * c));
    else
        return x * x * (2.f / c - x / (c * c));
}

XAD_INLINE double smooth_max(double x, double y, double c = 0.001)
{
    return 0.5 * (x + y + smooth_abs(x - y, c));
}

XAD_INLINE float smooth_max(float x, float y, float c = 0.001f)
{
    return 0.5f * (x + y + smooth_abs(x - y, c));
}

XAD_INLINE double smooth_min(double x, double y, double c = 0.001)
{
    return 0.5 * (x + y - smooth_abs(x - y, c));
}

XAD_INLINE float smooth_min(float x, float y, float c = 0.001f)
{
    return 0.5f * (x + y - smooth_abs(x - y, c));
}

// the same factors as degrees_op and radians_op
XAD_INLINE double degrees(double x) { return x * 57.2957795130823208767981548141051703324054725; }
XAD_INLINE float degrees(float x) { return x * 57.2957795130823208767981548141051703324054725f; }
XAD_INLINE double radians(double x) { return x * 0.0174532925199432957692369076848861271344287; }
XAD_INLINE float radians(float x) { return x * 0.0174532925199432957692369076848861271344287f; }

using ::std::acosh;
using ::std::asinh;
using ::std::atan2;
using ::std::atanh;
using ::std::cbrt;
using ::std::erf;
using ::std::erfc;
using ::std::exp2;
using ::std::expm1;
using ::std::fmax;
using ::std::fmin;
using ::std::fmod;
using ::std::fpclassify;
using ::std::hypot;
using ::std::isfinite;
using ::std::isinf;
using ::std::isnan;
using ::std::isnormal;
using ::std::lgamma;
using ::std::log1p;
using ::std::log2;
using ::std::nextafter;
using ::std::remainder;
using ::std::round;
using ::std::signbit;
using ::std::tgamma;
using ::std::trunc;

namespace detail
{
// n in r = a - n * b, from r = fmod(a, b) or remainder(a, b): a / b - r / b is within about
// (|n| + 1) ulp of n, so this is exact for |n| < 2^(digits - 2); +-inf where n overflows, NaN where
// r is NaN
template <class T>
XAD_INLINE typename std::enable_if<!is_active<T>::value, T>::type remainder_quotient(const T& a,
                                                                                     const T& b,
                                                                                     const T& r)
{
    using std::round;
    return round(a / b - r / b);
}

// n is piecewise constant: computed from the passive values
template <class T>
XAD_INLINE typename std::enable_if<is_active<T>::value, typename ExprTraits<T>::nested_type>::type
remainder_quotient(const T& a, const T& b, const T& r)
{
    return remainder_quotient(value(a), value(b), value(r));
}

// log(base) * v, the derivative of v = base^y in y, but 0 where v is 0 (as for base 0)
template <class Scalar>
XAD_INLINE Scalar powExponentDerivative(const Scalar& logBase, const Scalar& v)
{
    const Scalar d = logBase * v;  // unconditional, so the choice is a select, not a branch
    return v == Scalar() ? Scalar() : d;
}

// n / (x^2 + y^2), also where the squares overflow or underflow
template <class Scalar>
XAD_INLINE Scalar divBySumOfSquares(const Scalar& n, const Scalar& x, const Scalar& y)
{
    typedef typename ExprTraits<Scalar>::nested_type nested;
    const Scalar d = x * x + y * y;
    if (XAD_LIKELY(d >= (std::numeric_limits<nested>::min)() &&
                   d <= (std::numeric_limits<nested>::max)()))
        return n / d;
    const Scalar h = hypot(x, y);
    return n / h / h;
}

// 1 / sqrt(x^2 + c) for |c| = 1, also where x^2 overflows (and x^2 + c is x^2)
template <class Scalar>
XAD_INLINE Scalar invSqrtOfSquarePlus(const Scalar& x, const Scalar& c)
{
    typedef typename ExprTraits<Scalar>::nested_type nested;
    const Scalar d = x * x + c;
    if (d <= (std::numeric_limits<nested>::max)())
        return Scalar(1) / sqrt(d);
    return Scalar(1) / abs(x);
}
}  // namespace detail

namespace detail
{
// pi and pi^2 in the precision of T's innermost scalar
template <class T>
XAD_INLINE T piOf()
{
    typedef typename ExprTraits<T>::nested_type nested;
    return T(nested(3.14159265358979323846264338327950288419716939937510));
}

template <class T>
XAD_INLINE T quietNaNOf()
{
    typedef typename ExprTraits<T>::nested_type nested;
    return T(std::numeric_limits<nested>::quiet_NaN());
}

// digamma (psi) function, the derivative of lgamma: the recurrence psi(x) = psi(x + 1) - 1 / x
// shifts x to at least 10, where the asymptotic series is accurate to double precision; negative
// arguments use the reflection psi(x) = psi(1 - x) - pi / tan(pi x), with poles (NaN) at the
// non-positive integers
template <class T>
T digamma(const T& x0)
{
    using std::floor;
    using std::log;
    using std::round;
    using std::tan;
    T x = x0;
    T result = T(0);
    if (x != x)
        return x;
    if (x <= T(0))
    {
        if (x == floor(x))
            return quietNaNOf<T>();
        // x - round(x) is exact, so pi times it keeps its relative precision near the poles
        result = -piOf<T>() / tan(piOf<T>() * (x - round(x)));
        x = T(1) - x;
    }
    while (x < T(10))
    {
        result -= T(1) / x;
        x += T(1);
    }
    const T ix2 = T(1) / (x * x);
    // Bernoulli terms B_2k / (2k x^2k), k = 1..7
    const T series =
        ix2 *
        (T(1.0 / 12) -
         ix2 * (T(1.0 / 120) -
                ix2 * (T(1.0 / 252) -
                       ix2 * (T(1.0 / 240) - ix2 * (T(1.0 / 132) - ix2 * (T(691.0 / 32760) -
                                                                          ix2 * T(1.0 / 12)))))));
    return result + log(x) - T(0.5) / x - series;
}

// trigamma function, the derivative of digamma, in the same way: psi1(x) = psi1(x + 1) + 1 / x^2
// up to x >= 10, then the asymptotic series; reflection psi1(x) = pi^2 / sin^2(pi x) - psi1(1 - x).
// Written with arithmetic only, so it also works on active types (higher-order derivatives)
template <class T>
T trigamma(const T& x0)
{
    using std::floor;
    using std::round;
    using std::sin;
    T x = x0;
    if (x != x)
        return x;
    T result = T(0);
    T sign = T(1);
    if (x <= T(0))
    {
        if (x == floor(x))
            return quietNaNOf<T>();
        const T s = sin(piOf<T>() * (x - round(x)));
        result = piOf<T>() * piOf<T>() / (s * s);
        sign = T(-1);
        x = T(1) - x;
    }
    T shifted = T(0);
    while (x < T(10))
    {
        shifted += T(1) / (x * x);
        x += T(1);
    }
    const T ix = T(1) / x;
    const T ix2 = ix * ix;
    // 1/x + 1/(2x^2) + sum of B_2k / x^(2k+1), k = 1..7
    const T series =
        ix + T(0.5) * ix2 +
        ix * ix2 *
            (T(1.0 / 6) -
             ix2 * (T(1.0 / 30) -
                    ix2 * (T(1.0 / 42) -
                           ix2 * (T(1.0 / 30) - ix2 * (T(5.0 / 66) - ix2 * (T(691.0 / 2730) -
                                                                            ix2 * T(7.0 / 6)))))));
    return result + sign * (shifted + series);
}
}  // namespace detail

// digamma (psi) function, the logarithmic derivative of the gamma function
XAD_INLINE double digamma(double x) { return detail::digamma(x); }
XAD_INLINE float digamma(float x) { return static_cast<float>(detail::digamma(double(x))); }

}  // namespace xad

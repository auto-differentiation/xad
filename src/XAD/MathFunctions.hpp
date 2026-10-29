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
using ::std::log1p;
using ::std::log2;
using ::std::nextafter;
using ::std::remainder;
using ::std::round;
using ::std::signbit;
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

}  // namespace xad

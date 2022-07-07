/*******************************************************************************

   Importing or declaring of math functions in our namespace.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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

namespace detail
{

template <class T1, class T2>
struct PromoteFloat
{
    typedef typename std::conditional<
        std::is_same<T1, float>::value && std::is_same<T2, float>::value, float,
        typename std::conditional<std::is_same<T1, long double>::value ||
                                      std::is_same<T2, long double>::value,
                                  long double, double>::type>::type type;
};
}  // namespace detail

template <class Arithmetic1, class Arithmetic2>
XAD_INLINE typename std::enable_if<
    std::is_arithmetic<Arithmetic1>::value && std::is_arithmetic<Arithmetic2>::value &&
        std::is_fundamental<Arithmetic1>::value && std::is_fundamental<Arithmetic2>::value,
    typename detail::PromoteFloat<Arithmetic1, Arithmetic2>::type>::type
fmod(const Arithmetic1& x, const Arithmetic2& y)
{
    typedef typename detail::PromoteFloat<Arithmetic1, Arithmetic2>::type type;
    return ::std::fmod((type)x, (type)y);
}

template <class Arithmetic1, class Arithmetic2>
XAD_INLINE typename std::enable_if<
    std::is_arithmetic<Arithmetic1>::value && std::is_arithmetic<Arithmetic2>::value &&
        std::is_fundamental<Arithmetic1>::value && std::is_fundamental<Arithmetic2>::value,
    typename detail::PromoteFloat<Arithmetic1, Arithmetic2>::type>::type
atan2(Arithmetic1 x, Arithmetic2 y)
{
    typedef typename detail::PromoteFloat<Arithmetic1, Arithmetic2>::type type;
    return ::std::atan2((type)x, (type)y);
}

using ::std::acosh;
using ::std::asinh;
using ::std::atanh;
using ::std::cbrt;
using ::std::erf;
using ::std::erfc;
using ::std::exp2;
using ::std::expm1;
using ::std::fmax;
using ::std::fmin;
using ::std::fpclassify;
using ::std::isfinite;
using ::std::isinf;
using ::std::isnan;
using ::std::isnormal;
using ::std::log1p;
using ::std::log2;
using ::std::remainder;
using ::std::remquo;
using ::std::round;
using ::std::signbit;
using ::std::trunc;
using ::std::nextafter;

}  // namespace xad

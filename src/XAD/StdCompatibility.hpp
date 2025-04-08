/*******************************************************************************

   Placing XAD math functions into the std namespace for std::log type
   expressions to work, as well as specialising numeric_limits.

   This partially violates the C++ standard's "don't specialize std templates"
   rule but is necessary for integration with other libraries.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

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


#include <XAD/BinaryOperators.hpp>
#include <XAD/Literals.hpp>
#include <XAD/MathFunctions.hpp>
#include <XAD/UnaryOperators.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

namespace std
{
using xad::abs;
using xad::acos;
using xad::acosh;
using xad::asin;
using xad::asinh;
using xad::atan;
using xad::atan2;
using xad::atanh;
using xad::cbrt;
using xad::ceil;
using xad::copysign;
using xad::cos;
using xad::cosh;
using xad::erf;
using xad::erfc;
using xad::exp;
using xad::exp2;
using xad::expm1;
using xad::fabs;
using xad::floor;
using xad::fmax;
using xad::fmin;
using xad::fmod;
using xad::fpclassify;
using xad::frexp;
using xad::hypot;
using xad::ilogb;
using xad::isfinite;
using xad::isinf;
using xad::isnan;
using xad::isnormal;
using xad::ldexp;
using xad::llround;
using xad::log;
using xad::log10;
using xad::log1p;
using xad::log2;
using xad::lround;
using xad::max;
using xad::min;
using xad::modf;
using xad::nextafter;
using xad::pow;
using xad::remainder;
using xad::remquo;
using xad::round;
using xad::scalbn;
using xad::signbit;
using xad::sin;
using xad::sinh;
using xad::sqrt;
using xad::tan;
using xad::tanh;
using xad::trunc;

#if defined(_MSC_VER)

#include <cmath> // must come first to prevent ODR issues
// already included but there for clarity for now

inline double copysign(const xad::AReal<double>& x, const xad::AReal<double>& y) noexcept {
    return ::xad::value(::xad::copysign(x, y));
}

inline double copysign(const xad::AReal<double>& x, double y) noexcept {
    return ::xad::value(::xad::copysign(x, y));
}

inline double copysign(double x, const xad::AReal<double>& y) noexcept {
    return ::xad::value(::xad::copysign(x, y));
}

#endif

template <class Scalar, class Derived>
inline std::string to_string(const xad::Expression<Scalar, Derived>& _Val)
{
    return to_string(value(_Val));
}

}  // namespace std

namespace std
{

// note that these return the underlying doubles, not the active type,
// but since they are constant and convertible, it's the right behaviour
// for the majority of cases

template <class T>
struct numeric_limits<xad::AReal<T>> : std::numeric_limits<T>
{
};

template <class T>
struct numeric_limits<xad::FReal<T>> : std::numeric_limits<T>
{
};

}  // namespace std

// hashing for active types
namespace std
{

template <class T>
struct hash<xad::AReal<T>>
{
    std::size_t operator()(xad::AReal<T> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

template <class T>
struct hash<xad::FReal<T>>
{
    std::size_t operator()(xad::FReal<T> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

// type traits
template <class T>
struct is_floating_point<xad::AReal<T>> : std::is_floating_point<T>
{
};
template <class T>
struct is_floating_point<xad::FReal<T>> : std::is_floating_point<T>
{
};
template <class T>
struct is_arithmetic<xad::AReal<T>> : std::is_arithmetic<T>
{
};
template <class T>
struct is_arithmetic<xad::FReal<T>> : std::is_arithmetic<T>
{
};
template <class T>
struct is_signed<xad::AReal<T>> : std::is_signed<T>
{
};
template <class T>
struct is_signed<xad::FReal<T>> : std::is_signed<T>
{
};
template <class T>
struct is_pod<xad::AReal<T>> : std::false_type
{
};
template <class T>
struct is_pod<xad::FReal<T>> : std::false_type
{
};
template <class T>
struct is_fundamental<xad::AReal<T>> : std::false_type
{
};
template <class T>
struct is_fundamental<xad::FReal<T>> : std::false_type
{
};
#if !(defined(__GNUC__) && __GNUC__ < 5) || defined(__clang__)
template <class T>
struct is_trivially_copyable<xad::FReal<T>> : std::is_trivially_copyable<T>
{
};
#endif
template <class T>
struct is_scalar<xad::AReal<T>> : std::false_type
{
};
template <class T>
struct is_scalar<xad::FReal<T>> : std::false_type
{
};
template <class T>
struct is_compound<xad::AReal<T>> : std::true_type
{
};
template <class T>
struct is_compound<xad::FReal<T>> : std::true_type
{
};

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)

// For some reason, in VS 2022, a generic template for is_floating_point_v is not used
// in overload resolution. We need to fully specialise the template for common types
// here (first and second order only for now)

#define XAD_TEMPLATE_TRAIT_FUNC_FIRST(name_v, value)                                               \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<double>> = value;                                      \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<float>> = value;                                       \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<long double>> = value;                                 \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<double>> = value;                                      \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<float>> = value;                                       \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<long double>> = value

#define XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, value)                                              \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<double>>> = value;                          \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<float>>> = value;                           \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<long double>>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<double>>> = value;                          \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<float>>> = value;                           \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<long double>>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<double>>> = value;                          \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<float>>> = value;                           \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<long double>>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<double>>> = value;                          \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<float>>> = value;                           \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<long double>>> = value

#define XAD_TEMPLATE_TRAIT_FUNC(name_v, value)                                                     \
    XAD_TEMPLATE_TRAIT_FUNC_FIRST(name_v, value);                                                  \
    XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, value)

XAD_TEMPLATE_TRAIT_FUNC(is_floating_point_v, true);
XAD_TEMPLATE_TRAIT_FUNC(is_arithmetic_v, true);
XAD_TEMPLATE_TRAIT_FUNC(is_integral_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_fundamental_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_scalar_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_compound_v, true);

#undef XAD_TEMPLATE_TRAIT_FUNC
#undef XAD_TEMPLATE_TRAIT_FUNC_FIRST
#undef XAD_TEMPLATE_TRAIT_FUNC_SECOND

template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<double>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<float>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<long double>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<float>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<long double>>> = true;

#endif

#if defined(_MSC_VER)

// for MSVC, we need this workaround so that the safety checks in their STL
// for floating point types are also passing for the XAD types
#if (_MSC_VER > 1900)
// VS 2017+, when the STL checks if a type is in the list of built-in floating point types,
// this should forward the check to the wrapped type by AReal or FReal.
//
// (In GCC, std::is_floating_point is used instead, where traits above work)

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L || defined(__clang__))
#define _XAD_INLINE_VAR inline
#else
#define _XAD_INLINE_VAR
#endif

template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<double>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<float>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<long double>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<double>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<float>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<long double>, float, double, long double> =
    true;

template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::AReal<double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::AReal<float>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::AReal<long double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::AReal<double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::AReal<float>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::AReal<long double>>, float, double, long double> = true;

template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::FReal<double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::FReal<float>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<xad::FReal<long double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::FReal<double>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::FReal<float>>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<xad::FReal<long double>>, float, double, long double> = true;

#undef _XAD_INLINE_VAR

#else
}
#include <random>
namespace std
{

// prior versions of MSVC (2015) use a different check
template <class T>
struct _Is_RealType<xad::AReal<T>> : public _Is_RealType<T>
{
};

template <class T>
struct _Is_RealType<xad::FReal<T>> : public _Is_RealType<T>
{
};

#endif

#endif
}  // namespace std

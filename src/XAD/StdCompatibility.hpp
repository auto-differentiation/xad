/*******************************************************************************

   Placing XAD math functions into the std namespace for std::log type
   expressions to work, as well as specialising numeric_limits.

   This partially violates the C++ standard's "don't specialize std templates"
   rule but is necessary for integration with other libraries.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, version 3 of the License only.

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
using xad::fma;
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

#if defined(_MSC_VER) || (__clang_major__ > 16 && defined(_LIBCPP_VERSION))
// we need these explicit instantiation to disambiguate templates in MSVC & Clang

template <class T, std::size_t N>
XAD_INLINE T copysign(const T& x, const xad::AReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::AReal<T, N> copysign(const T2& x, const xad::AReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, std::size_t N>
XAD_INLINE xad::AReal<T, N> copysign(const xad::AReal<T, N> x, const T& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::AReal<T, N> copysign(const xad::AReal<T, N> x, const T2& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, std::size_t N>
XAD_INLINE T copysign(const T& x, const xad::FReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::FReal<T, N> copysign(const T2& x, const xad::FReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, std::size_t N>
XAD_INLINE xad::FReal<T, N> copysign(const xad::FReal<T, N> x, const T& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::FReal<T, N> copysign(const xad::FReal<T, N> x, const T2& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> copysign(const T2& x, const xad::ARealDirect<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> copysign(const xad::ARealDirect<T, N> x, const T& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> copysign(const xad::ARealDirect<T, N> x, const T2& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> copysign(const T2& x, const xad::FRealDirect<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> copysign(const xad::FRealDirect<T, N> x, const T& y)
{
    return ::xad::copysign(x, xad::value(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> copysign(const xad::FRealDirect<T, N> x, const T2& y)
{
    return ::xad::copysign(x, xad::value(y));
}

#endif

template <class Scalar, std::size_t N>
inline std::string to_string(const xad::ARealDirect<Scalar, N>& _Val);

template <class Scalar, std::size_t N>
inline std::string to_string(const xad::FRealDirect<Scalar, N>& _Val);

template <class Scalar, class Derived, class Deriv>
inline std::string to_string(const xad::Expression<Scalar, Derived, Deriv>& _Val)
{
    return to_string(value(_Val));
}

template <class Scalar, std::size_t N>
inline std::string to_string(const xad::ARealDirect<Scalar, N>& _Val)
{
    return to_string(xad::value(_Val));
}

template <class Scalar, std::size_t N>
inline std::string to_string(const xad::FRealDirect<Scalar, N>& _Val)
{
    return to_string(xad::value(_Val));
}

}  // namespace std

namespace std
{

// note that these return the underlying doubles, not the active type,
// but since they are constant and convertible, it's the right behaviour
// for the majority of cases

template <class T, std::size_t N>
struct numeric_limits<xad::AReal<T, N>> : std::numeric_limits<T>
{
};

template <class T, std::size_t N>
struct numeric_limits<xad::FReal<T, N>> : std::numeric_limits<T>
{
};

template <class T, std::size_t N>
struct numeric_limits<xad::ARealDirect<T, N>> : std::numeric_limits<T>
{
};

template <class T, std::size_t N>
struct numeric_limits<xad::FRealDirect<T, N>> : std::numeric_limits<T>
{
};

}  // namespace std

// hashing for active types
namespace std
{

template <class T, std::size_t N>
struct hash<xad::AReal<T, N>>
{
    std::size_t operator()(xad::AReal<T, N> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

template <class T, std::size_t N>
struct hash<xad::FReal<T, N>>
{
    std::size_t operator()(xad::FReal<T, N> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

template <class T, std::size_t N>
struct hash<xad::ARealDirect<T, N>>
{
    std::size_t operator()(xad::ARealDirect<T, N> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

template <class T, std::size_t N>
struct hash<xad::FRealDirect<T, N>>
{
    std::size_t operator()(xad::FRealDirect<T, N> const& s) const noexcept
    {
        return std::hash<T>{}(xad::value(s));
    }
};

// type traits
// libc++ 21 marks the standard type traits [[clang::no_specializations]],
// which raises -Winvalid-specialization (a default-error warning) on the
// specializations below; suppress it to keep behaviour identical across
// standard libraries (libc++ internals use builtins, not the public traits).
// MSVC 14.5x reports the same as C5285 (an error with /WX).
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Winvalid-specialization"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 5285)
#endif
template <class T, std::size_t N>
struct is_floating_point<xad::AReal<T, N>> : std::is_floating_point<T>
{
};
template <class T, std::size_t N>
struct is_floating_point<xad::FReal<T, N>> : std::is_floating_point<T>
{
};
template <class T, std::size_t N>
struct is_floating_point<xad::ARealDirect<T, N>> : std::is_floating_point<T>
{
};
template <class T, std::size_t N>
struct is_floating_point<xad::FRealDirect<T, N>> : std::is_floating_point<T>
{
};
template <class T, std::size_t N>
struct is_arithmetic<xad::AReal<T, N>> : std::is_arithmetic<T>
{
};
template <class T, std::size_t N>
struct is_arithmetic<xad::FReal<T, N>> : std::is_arithmetic<T>
{
};
template <class T, std::size_t N>
struct is_arithmetic<xad::ARealDirect<T, N>> : std::is_arithmetic<T>
{
};
template <class T, std::size_t N>
struct is_arithmetic<xad::FRealDirect<T, N>> : std::is_arithmetic<T>
{
};
template <class T, std::size_t N>
struct is_signed<xad::AReal<T, N>> : std::is_signed<T>
{
};
template <class T, std::size_t N>
struct is_signed<xad::FReal<T, N>> : std::is_signed<T>
{
};
template <class T, std::size_t N>
struct is_signed<xad::ARealDirect<T, N>> : std::is_signed<T>
{
};
template <class T, std::size_t N>
struct is_signed<xad::FRealDirect<T, N>> : std::is_signed<T>
{
};
template <class T, std::size_t N>
struct is_pod<xad::AReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_pod<xad::FReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_fundamental<xad::AReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_fundamental<xad::FReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_fundamental<xad::ARealDirect<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_fundamental<xad::FRealDirect<T, N>> : std::false_type
{
};
#if !(defined(__GNUC__) && __GNUC__ < 5) || defined(__clang__)
template <class T, std::size_t N>
struct is_trivially_copyable<xad::FReal<T, N>> : std::is_trivially_copyable<T>
{
};
template <class T, std::size_t N>
struct is_trivially_copyable<xad::FRealDirect<T, N>> : std::is_trivially_copyable<T>
{
};
#endif
template <class T, std::size_t N>
struct is_scalar<xad::AReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_scalar<xad::FReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_scalar<xad::ARealDirect<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_scalar<xad::FRealDirect<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_compound<xad::AReal<T, N>> : std::true_type
{
};
template <class T, std::size_t N>
struct is_compound<xad::FReal<T, N>> : std::true_type
{
};
template <class T, std::size_t N>
struct is_compound<xad::ARealDirect<T, N>> : std::true_type
{
};
template <class T, std::size_t N>
struct is_compound<xad::FRealDirect<T, N>> : std::true_type
{
};

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)

// For some reason, in VS 2022, a generic template for is_floating_point_v is not used
// in overload resolution. We need to fully specialise the template for common types
// here (first and second order only for now)

#define XAD_TEMPLATE_TRAIT_FUNC_FIRST(name_v, N, value)                                            \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<double, N>> = value;                                   \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<float, N>> = value;                                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<long double, N>> = value;                              \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<double, N>> = value;                                   \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<float, N>> = value;                                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<long double, N>> = value

#define XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, M, value)                                        \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<double, M>, N>> = value;                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<float, M>, N>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::AReal<long double, M>, N>> = value;               \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<double, M>, N>> = value;                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<float, M>, N>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::AReal<long double, M>, N>> = value;               \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<double, M>, N>> = value;                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<float, M>, N>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::AReal<xad::FReal<long double, M>, N>> = value;               \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<double, M>, N>> = value;                    \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<float, M>, N>> = value;                     \
    template <>                                                                                    \
    inline constexpr bool name_v<xad::FReal<xad::FReal<long double, M>, N>> = value

// direct mode types, first order and all second order combinations involving a direct type
#define XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, ...)                                           \
    template <>                                                                                    \
    inline constexpr bool name_v<__VA_ARGS__> = value

#define XAD_TEMPLATE_TRAIT_FUNC_DIRECT_FIRST(name_v, N, value)                                     \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::ARealDirect<double, N>);                      \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::ARealDirect<float, N>);                       \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::ARealDirect<long double, N>);                 \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::FRealDirect<double, N>);                      \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::FRealDirect<float, N>);                       \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::FRealDirect<long double, N>)

#define XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, Outer, Inner, N, M, value)                          \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::Outer<xad::Inner<double, M>, N>);             \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::Outer<xad::Inner<float, M>, N>);              \
    XAD_TEMPLATE_TRAIT_FUNC_TYPE(name_v, value, xad::Outer<xad::Inner<long double, M>, N>)

#define XAD_TEMPLATE_TRAIT_FUNC_DIRECT_SECOND(name_v, N, M, value)                                 \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, ARealDirect, AReal, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, ARealDirect, FReal, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, ARealDirect, ARealDirect, N, M, value);                 \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, ARealDirect, FRealDirect, N, M, value);                 \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FRealDirect, AReal, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FRealDirect, FReal, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FRealDirect, ARealDirect, N, M, value);                 \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FRealDirect, FRealDirect, N, M, value);                 \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, AReal, ARealDirect, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, AReal, FRealDirect, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FReal, ARealDirect, N, M, value);                       \
    XAD_TEMPLATE_TRAIT_FUNC_NESTED(name_v, FReal, FRealDirect, N, M, value)

#define XAD_TEMPLATE_TRAIT_FUNC1(name_v, N, value)                                                 \
    XAD_TEMPLATE_TRAIT_FUNC_FIRST(name_v, N, value);                                               \
    XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 1, value);                                           \
    XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 2, value);                                           \
    XAD_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 4, value);                                           \
    XAD_TEMPLATE_TRAIT_FUNC_DIRECT_FIRST(name_v, N, value);                                        \
    XAD_TEMPLATE_TRAIT_FUNC_DIRECT_SECOND(name_v, N, 1, value);                                    \
    XAD_TEMPLATE_TRAIT_FUNC_DIRECT_SECOND(name_v, N, 2, value);                                    \
    XAD_TEMPLATE_TRAIT_FUNC_DIRECT_SECOND(name_v, N, 4, value)

#define XAD_TEMPLATE_TRAIT_FUNC(name_v, value)                                                     \
    XAD_TEMPLATE_TRAIT_FUNC1(name_v, 1, value);                                                    \
    XAD_TEMPLATE_TRAIT_FUNC1(name_v, 2, value);                                                    \
    XAD_TEMPLATE_TRAIT_FUNC1(name_v, 4, value)

XAD_TEMPLATE_TRAIT_FUNC(is_floating_point_v, true);
XAD_TEMPLATE_TRAIT_FUNC(is_arithmetic_v, true);
XAD_TEMPLATE_TRAIT_FUNC(is_integral_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_fundamental_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_scalar_v, false);
XAD_TEMPLATE_TRAIT_FUNC(is_compound_v, true);

#undef XAD_TEMPLATE_TRAIT_FUNC
#undef XAD_TEMPLATE_TRAIT_FUNC1
#undef XAD_TEMPLATE_TRAIT_FUNC_FIRST
#undef XAD_TEMPLATE_TRAIT_FUNC_SECOND
#undef XAD_TEMPLATE_TRAIT_FUNC_DIRECT_FIRST
#undef XAD_TEMPLATE_TRAIT_FUNC_DIRECT_SECOND
#undef XAD_TEMPLATE_TRAIT_FUNC_NESTED
#undef XAD_TEMPLATE_TRAIT_FUNC_TYPE

template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<double, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<double, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<double, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<float, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<float, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<float, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<long double, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<long double, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<long double, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<float>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FReal<long double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<double, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<double, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<double, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<float, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<float, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<float, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<long double, 1>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<long double, 2>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<long double, 4>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FRealDirect<double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FRealDirect<float>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FRealDirect<long double>>> =
    true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FReal<double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FReal<float>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FRealDirect<xad::FReal<long double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FRealDirect<double>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FRealDirect<float>>> = true;
template <>
inline constexpr bool is_trivially_copyable_v<xad::FReal<xad::FRealDirect<long double>>> = true;

#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)

// for MSVC, we need this workaround so that the safety checks in their STL
// for floating point types are also passing for the XAD types

// VS 2017+, when the STL checks if a type is in the list of built-in floating point types,
// this should forward the check to the wrapped type by AReal or FReal.
//
// (In GCC, std::is_floating_point is used instead, where traits above work)

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L || defined(__clang__))
#define _XAD_INLINE_VAR inline
#else
#define _XAD_INLINE_VAR
#endif
// clang declares these inline in every mode (C++11/14 included): keep that, without -Wc++17-extensions
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<double, 1>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<float, 1>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<long double, 1>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<double, 2>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<float, 2>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<long double, 2>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<double, 4>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::AReal<float, 4>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::AReal<long double, 4>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<double, 1>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<float, 1>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<long double, 1>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<double, 2>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<float, 2>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<long double, 2>, float, double, long double> = true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<double, 4>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool _Is_any_of_v<xad::FReal<float, 4>, float, double, long double> =
    true;
template <>
_XAD_INLINE_VAR constexpr bool
    _Is_any_of_v<xad::FReal<long double, 4>, float, double, long double> = true;

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

#define XAD_DIRECT_IS_ANY_OF(...)                                                                  \
    template <>                                                                                    \
    _XAD_INLINE_VAR constexpr bool _Is_any_of_v<__VA_ARGS__, float, double, long double> = true

#define XAD_DIRECT_IS_ANY_OF_SCALARS(Outer, N)                                                     \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<double, N>);                                                   \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<float, N>);                                                    \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<long double, N>)

#define XAD_DIRECT_IS_ANY_OF_NESTED(Outer, Inner)                                                  \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<xad::Inner<double>>);                                          \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<xad::Inner<float>>);                                           \
    XAD_DIRECT_IS_ANY_OF(xad::Outer<xad::Inner<long double>>)

XAD_DIRECT_IS_ANY_OF_SCALARS(ARealDirect, 1);
XAD_DIRECT_IS_ANY_OF_SCALARS(ARealDirect, 2);
XAD_DIRECT_IS_ANY_OF_SCALARS(ARealDirect, 4);
XAD_DIRECT_IS_ANY_OF_SCALARS(FRealDirect, 1);
XAD_DIRECT_IS_ANY_OF_SCALARS(FRealDirect, 2);
XAD_DIRECT_IS_ANY_OF_SCALARS(FRealDirect, 4);
XAD_DIRECT_IS_ANY_OF_NESTED(ARealDirect, AReal);
XAD_DIRECT_IS_ANY_OF_NESTED(ARealDirect, FReal);
XAD_DIRECT_IS_ANY_OF_NESTED(ARealDirect, ARealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(ARealDirect, FRealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(FRealDirect, AReal);
XAD_DIRECT_IS_ANY_OF_NESTED(FRealDirect, FReal);
XAD_DIRECT_IS_ANY_OF_NESTED(FRealDirect, ARealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(FRealDirect, FRealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(AReal, ARealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(AReal, FRealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(FReal, ARealDirect);
XAD_DIRECT_IS_ANY_OF_NESTED(FReal, FRealDirect);

#undef XAD_DIRECT_IS_ANY_OF_NESTED
#undef XAD_DIRECT_IS_ANY_OF_SCALARS
#undef XAD_DIRECT_IS_ANY_OF
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef _XAD_INLINE_VAR

#endif

// https://github.com/auto-differentiation/xad/issues/169
#if __clang_major__ >= 16 && defined(_LIBCPP_VERSION)

template <typename T, std::size_t N>
struct __libcpp_random_is_valid_realtype<xad::AReal<T, N>> : true_type
{
};

template <typename T, std::size_t N>
struct __libcpp_random_is_valid_realtype<xad::FReal<T, N>> : true_type
{
};

template <typename T, std::size_t N>
struct __libcpp_random_is_valid_realtype<xad::ARealDirect<T, N>> : true_type
{
};

template <typename T, std::size_t N>
struct __libcpp_random_is_valid_realtype<xad::FRealDirect<T, N>> : true_type
{
};

#endif

}  // namespace std

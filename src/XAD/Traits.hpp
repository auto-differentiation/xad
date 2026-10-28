/*******************************************************************************

   Declaration of traits classes.

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

#include <XAD/Vec.hpp>
#include <cmath>
#include <type_traits>

namespace xad
{

enum Direction
{
    DIR_NONE,
    DIR_FORWARD,
    DIR_REVERSE
};

template <class T>
struct ExprTraits
{
    static const bool isExpr = false;
    static const int numVariables = 0;
    static const bool isForward = false;
    static const bool isReverse = false;
    static const bool isLiteral = false;
    static const Direction direction = Direction::DIR_NONE;
    static const std::size_t vector_size = 1;

    typedef T nested_type;
    typedef T value_type;
    typedef T scalar_type;
};

template <class T>
struct ExprTraits<const T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<volatile T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<const volatile T> : ExprTraits<T>
{
};

template <class Op>
struct OperatorTraits
{
    enum
    {
        useResultBasedDerivatives = 0
    };
};

template <class T>
struct float_or_double : public std::false_type
{
};
template <>
struct float_or_double<float> : public std::true_type
{
};
template <>
struct float_or_double<double> : public std::true_type
{
};

template <class, std::size_t>
struct ARealDirect;
template <class, std::size_t>
struct FRealDirect;

namespace detail
{

template <class T>
struct is_direct : public std::false_type
{
};

template <class T, std::size_t N>
struct is_direct<ARealDirect<T, N>> : public std::true_type
{
};

template <class T, std::size_t N>
struct is_direct<FRealDirect<T, N>> : public std::true_type
{
};

// built-in arithmetic types (is_fundamental: StdCompatibility.hpp makes active types arithmetic)
template <class T>
struct is_builtin_arithmetic
    : public std::integral_constant<bool,
                                    std::is_arithmetic<T>::value && std::is_fundamental<T>::value>
{
};

// operands forwarded to the underlying AReal / FReal: built-in arithmetic types, expressions
template <class T>
struct direct_operand
    : public std::integral_constant<bool, is_builtin_arithmetic<T>::value || ExprTraits<T>::isExpr>
{
};

// other operands implicitly convertible to D (e.g. QuantLib's Null<Real>); not enums
template <class D, class T>
struct direct_convertible_operand
    : public std::integral_constant<bool, !direct_operand<T>::value && !is_direct<T>::value &&
                                              !std::is_enum<T>::value &&
                                              std::is_convertible<const T&, D>::value>
{
};

// float, double and active types based on them (higher-order direct mode)
template <class T>
struct direct_scalar : public float_or_double<typename ExprTraits<T>::nested_type>
{
};

// active types: expressions (AReal, FReal, ...) and direct mode types
template <class T>
struct is_active : public std::integral_constant<bool, ExprTraits<T>::isExpr || is_direct<T>::value>
{
};

// Forward mode: a zero tangent adds nothing, even with an infinite or NaN partial (as for a
// constant operand). Derivatives are computed as usual and redone with tangentTerm if NaN,
// except for ops whose partials are finite where their arguments are, or which give the
// tangent themselves (op.tangent).
template <class Op>
struct HasFinitePartials : std::false_type
{
};

template <class Op>
struct HasTangentRule : std::false_type
{
};

template <class D>
XAD_INLINE typename std::enable_if<float_or_double<D>::value, bool>::type isNaNTangent(const D& d)
{
    return std::isnan(d);
}

// active tangents (higher order) can carry dependencies with a zero value, so they are kept
template <class D>
XAD_INLINE typename std::enable_if<!float_or_double<D>::value, bool>::type isNaNTangent(const D&)
{
    return false;
}

template <class T, std::size_t N>
XAD_INLINE bool isNaNTangent(const Vec<T, N>& d)
{
    for (std::size_t i = 0; i < N; ++i)
        if (isNaNTangent(d[i]))
            return true;
    return false;
}

// isNaNTangent(d), or false for ops with finite partials (a tag: MSVC warns on constant conditions)
template <class D>
XAD_INLINE bool isNaNTangent(const D&, std::true_type)
{
    return false;
}

template <class D>
XAD_INLINE bool isNaNTangent(const D& d, std::false_type)
{
    return isNaNTangent(d);
}

template <class P, class D>
XAD_INLINE typename std::enable_if<float_or_double<D>::value, D>::type tangentTerm(
    const P& partial, const D& tangent)
{
    return tangent == D() ? D() : D(partial * tangent);
}

template <class P, class D>
XAD_INLINE auto tangentTerm(const P& partial, const D& tangent) ->
    typename std::enable_if<!float_or_double<D>::value, decltype(partial * tangent)>::type
{
    return partial * tangent;
}

template <class P, class T, std::size_t N>
XAD_INLINE Vec<T, N> tangentTerm(const P& partial, const Vec<T, N>& tangent)
{
    Vec<T, N> r(tangent);
    for (std::size_t i = 0; i < N; ++i) r[i] = tangentTerm(partial, tangent[i]);
    return r;
}

// the tangent of v = a / b: (ta - v * tb) / b, where 1 / b * ta would be 0 * inf for b = 0
template <class S, class D>
XAD_INLINE D quotientTangent(const S& b, const S& v, const D& ta, const D& tb)
{
    return D((ta - v * tb) / b);
}

template <class S, class T, std::size_t N>
XAD_INLINE Vec<T, N> quotientTangent(const S& b, const S& v, const Vec<T, N>& ta,
                                     const Vec<T, N>& tb)
{
    Vec<T, N> r(ta);
    for (std::size_t i = 0; i < N; ++i) r[i] = quotientTangent(b, v, ta[i], tb[i]);
    return r;
}

}  // namespace detail

template <class T, std::size_t N>
struct DerivativesTraits
{
    using type = Vec<T, N>;
};

template <class T>
struct DerivativesTraits<T, 1>
{
    using type = T;
};

}  // namespace xad

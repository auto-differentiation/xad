/*******************************************************************************

   Overloaded operators for binary functions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2023 Xcelerit Computing Ltd.

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

#include <XAD/BinaryExpr.hpp>
#include <XAD/BinaryFunctors.hpp>
#include <XAD/BinaryMathFunctors.hpp>

#include <XAD/BinaryOperatorMacros.hpp>
#include <XAD/Macros.hpp>

namespace xad
{

XAD_BINARY_OPERATOR(operator+, add_op)
XAD_BINARY_OPERATOR(operator*, prod_op)
XAD_BINARY_OPERATOR(operator-, sub_op)
XAD_BINARY_OPERATOR(operator/, div_op)
XAD_BINARY_OPERATOR(pow, pow_op)
XAD_BINARY_OPERATOR(max, max_op)
XAD_BINARY_OPERATOR(fmax, fmax_op)
XAD_BINARY_OPERATOR(min, min_op)
XAD_BINARY_OPERATOR(fmin, fmin_op)
XAD_BINARY_OPERATOR(fmod, fmod_op)
XAD_BINARY_OPERATOR(atan2, atan2_op)
XAD_BINARY_OPERATOR(hypot, hypot_op)
XAD_BINARY_OPERATOR(smooth_abs, smooth_abs_op)
XAD_BINARY_OPERATOR(nextafter, nextafter_op)

// note - this is C++11 only
template <class T1, class T2, class T3>
XAD_INLINE auto smooth_max(const T1& x, const T2& y, const T3& c)
    -> decltype(0.5 * (x + y + smooth_abs(x - y, c)))
{
    return 0.5 * (x + y + smooth_abs(x - y, c));
}
template <class T1, class T2>
XAD_INLINE auto smooth_max(const T1& x, const T2& y) -> decltype(0.5 * (x + y + smooth_abs(x - y)))
{
    return 0.5 * (x + y + smooth_abs(x - y));
}
template <class T1, class T2, class T3>
XAD_INLINE auto smooth_min(const T1& x, const T2& y, const T3& c)
    -> decltype(0.5 * (x + y - smooth_abs(x - y, c)))
{
    return 0.5 * (x + y - smooth_abs(x - y, c));
}
template <class T1, class T2>
XAD_INLINE auto smooth_min(const T1& x, const T2& y) -> decltype(0.5 * (x + y - smooth_abs(x - y)))
{
    return 0.5 * (x + y - smooth_abs(x - y));
}

/////////// comparisons - they just return bool

#define XAD_COMPARE_OPERATOR(op)                                                                   \
    template <class Scalar, class Expr1, class Expr2>                                              \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr1>& a,                                \
                                const Expression<Scalar, Expr2>& b)                                \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE bool operator op(const typename ExprTraits<Expr>::value_type& a,                    \
                                const Expression<Scalar, Expr>& b)                                 \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr>& a,                                 \
                                const typename ExprTraits<Expr>::value_type& b)                    \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar>                                                                        \
    XAD_INLINE bool operator op(const AReal<Scalar>& a, const AReal<Scalar>& b)                    \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar>                                                                        \
    XAD_INLINE bool operator op(const FReal<Scalar>& a, const FReal<Scalar>& b)                    \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE bool operator op(typename ExprTraits<Expr>::nested_type a,                          \
                                const Expression<Scalar, Expr>& b)                                 \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr>& a,                                 \
                                typename ExprTraits<Expr>::nested_type b)                          \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }

XAD_COMPARE_OPERATOR(==)
XAD_COMPARE_OPERATOR(!=)
XAD_COMPARE_OPERATOR(<=)
XAD_COMPARE_OPERATOR(>=)
XAD_COMPARE_OPERATOR(<)
XAD_COMPARE_OPERATOR(>)

XAD_BINARY_OPERATOR(remainder, remainder_op)

// manual remquo due to additional argument
template <class Scalar, class Expr1, class Expr2>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, Expr1, Expr2> remquo(
    const Expression<Scalar, Expr1>& a, const Expression<Scalar, Expr2>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, Expr1, Expr2>(a.derived(), b.derived(),
                                                               remquo_op<Scalar>(quo));
}

template <class Scalar>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar>, ADVar<Scalar> > remquo(
    const AReal<Scalar>& a, const AReal<Scalar>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar>, ADVar<Scalar> >(
        ADVar<Scalar>(a), ADVar<Scalar>(b), remquo_op<Scalar>(quo));
}

template <class Scalar>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar>, FReal<Scalar> > remquo(
    const FReal<Scalar>& a, const FReal<Scalar>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar>, FReal<Scalar> >(
        a, b, remquo_op<Scalar>(quo));
}

template <class Scalar, class Expr>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr>
remquo(const typename ExprTraits<Expr>::value_type& a, const Expression<Scalar, Expr>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr>(
        typename wrapper_type<Scalar, Expr>::type(a), b.derived(), remquo_op<Scalar>(quo));
}

template <class Scalar, class Expr>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type>
remquo(const Expression<Scalar, Expr>& a, const typename ExprTraits<Expr>::value_type& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type>(
        a.derived(), typename wrapper_type<Scalar, Expr>::type(b), remquo_op<Scalar>(quo));
}

#undef XAD_BINARY_OPERATOR
#undef XAD_COMPARE_OPERATOR
}  // namespace xad

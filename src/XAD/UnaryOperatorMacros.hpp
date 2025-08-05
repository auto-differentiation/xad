/*******************************************************************************

   Macros for unary operators - to be included by UnaryOperators.hpp.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

namespace xad
{
template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct ADVar;
}  // namespace xad

#define XAD_UNARY_OPERATOR(_op, func)                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE UnaryExpr<Scalar, func<Scalar>, Expr, DerivativeType>(_op)(                         \
        const Expression<Scalar, Expr, DerivativeType>& a)                                         \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar>, Expr, DerivativeType>(a.derived());                 \
    }                                                                                              \
    template <class Scalar1, std::size_t N = 1>                                                    \
    XAD_INLINE UnaryExpr<Scalar1, func<Scalar1>, ADVar<Scalar1, N>,                                \
                         typename DerivativesTraits<Scalar1, N>::type>(_op)(                       \
        const AReal<Scalar1, N>& a1)                                                               \
    {                                                                                              \
        return UnaryExpr<Scalar1, func<Scalar1>, ADVar<Scalar1, N>,                                \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(a1));     \
    }

#define XAD_UNARY_BINSCAL2(_op, func)                                                              \
    template <class Scalar, class Expr, class T2, class DerivativeType>                            \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&                         \
            !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,                      \
        UnaryExpr<Scalar, func<Scalar, T2>, Expr, DerivativeType>>::                               \
        type(_op)(const Expression<Scalar, Expr, DerivativeType>& a, const T2& b)                  \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, T2>, Expr, DerivativeType>(a.derived(),              \
                                                                         func<Scalar, T2>(b));     \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Expr>::nested_type>, Expr,       \
                         DerivativeType>(_op)(const Expression<Scalar, Expr, DerivativeType>& a,   \
                                              typename ExprTraits<Expr>::nested_type b)            \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Expr>::nested_type>, Expr,       \
                         DerivativeType>(a.derived(),                                              \
                                         func<Scalar, typename ExprTraits<Expr>::nested_type>(b)); \
    }                                                                                              \
    template <class Scalar1, class T21, std::size_t N>                                             \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_arithmetic<T21>::value && std::is_fundamental<T21>::value &&                       \
            !std::is_same<T21, typename ExprTraits<Scalar1>::nested_type>::value,                  \
        UnaryExpr<Scalar1, func<Scalar1, T21>, ADVar<Scalar1, N>,                                  \
                  typename DerivativesTraits<Scalar1, N>::type>>::type(_op)(const AReal<Scalar1,   \
                                                                                        N>& a1,    \
                                                                            const T21& b1)         \
    {                                                                                              \
        return UnaryExpr<Scalar1, func<Scalar1, T21>, ADVar<Scalar1, N>,                           \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(a1),      \
                                                                       func<Scalar1, T21>(b1));    \
    }                                                                                              \
    template <class Scalar, std::size_t M>                                                         \
    XAD_INLINE UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Scalar>::nested_type>,           \
                         ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(_op)(      \
        const AReal<Scalar, M>& a, typename ExprTraits<Scalar>::nested_type b)                     \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Scalar>::nested_type>,           \
                         ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(           \
            ADVar<Scalar, M>(a), func<Scalar, typename ExprTraits<Scalar>::nested_type>(b));       \
    }

#define XAD_UNARY_BINSCAL1(_op, func)                                                              \
    template <class Scalar2, class Expr1, class T22, class Deriv>                                  \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_arithmetic<T22>::value && std::is_fundamental<T22>::value &&                       \
            !std::is_same<T22, typename ExprTraits<Expr1>::nested_type>::value,                    \
        UnaryExpr<Scalar2, func<Scalar2, T22>, Expr1,                                              \
                  Deriv>>::type(_op)(const T22& a2, const Expression<Scalar2, Expr1, Deriv>& b2)   \
    {                                                                                              \
        return UnaryExpr<Scalar2, func<Scalar2, T22>, Expr1, Deriv>(b2.derived(),                  \
                                                                    func<Scalar2, T22>(a2));       \
    }                                                                                              \
    template <class Scalar2, class Expr1, class Deriv>                                             \
    XAD_INLINE                                                                                     \
    UnaryExpr<Scalar2, func<Scalar2, typename ExprTraits<Expr1>::nested_type>, Expr1, Deriv>(_op)( \
        typename ExprTraits<Expr1>::nested_type a2, const Expression<Scalar2, Expr1, Deriv>& b2)   \
    {                                                                                              \
        return UnaryExpr<Scalar2, func<Scalar2, typename ExprTraits<Expr1>::nested_type>, Expr1,   \
                         Deriv>(b2.derived(),                                                      \
                                func<Scalar2, typename ExprTraits<Expr1>::nested_type>(a2));       \
    }                                                                                              \
    template <class Scalar3, class T23, std::size_t N>                                             \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_arithmetic<T23>::value && std::is_fundamental<T23>::value &&                       \
            !std::is_same<T23, typename ExprTraits<Scalar3>::nested_type>::value,                  \
        UnaryExpr<Scalar3, func<Scalar3, T23>, ADVar<Scalar3, N>,                                  \
                  typename DerivativesTraits<Scalar3, N>::type>>::type(_op)(const T23& a3,         \
                                                                            const AReal<Scalar3,   \
                                                                                        N>& b3)    \
    {                                                                                              \
        return UnaryExpr<Scalar3, func<Scalar3, T23>, ADVar<Scalar3, N>,                           \
                         typename DerivativesTraits<Scalar3, N>::type>(ADVar<Scalar3, N>(b3),      \
                                                                       func<Scalar3, T23>(a3));    \
    }                                                                                              \
    template <class Scalar3, std::size_t N>                                                        \
    XAD_INLINE UnaryExpr<Scalar3, func<Scalar3, typename ExprTraits<Scalar3>::nested_type>,        \
                         ADVar<Scalar3, N>, typename DerivativesTraits<Scalar3, N>::type>(_op)(    \
        typename ExprTraits<Scalar3>::nested_type a3, const AReal<Scalar3, N>& b3)                 \
    {                                                                                              \
        return UnaryExpr<Scalar3, func<Scalar3, typename ExprTraits<Scalar3>::nested_type>,        \
                         ADVar<Scalar3, N>, typename DerivativesTraits<Scalar3, N>::type>(         \
            ADVar<Scalar3, N>(b3), func<Scalar3, typename ExprTraits<Scalar3>::nested_type>(a3));  \
    }

#define XAD_UNARY_BINSCAL(_op, func)                                                               \
    XAD_UNARY_BINSCAL1(_op, func)                                                                  \
    XAD_UNARY_BINSCAL2(_op, func)

#define XAD_MAKE_UNARY_FUNC(func)                                                                  \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE UnaryExpr<Scalar, func##_op<Scalar>, Expr, DerivativeType>(func)(                   \
        const Expression<Scalar, Expr, DerivativeType>& x0)                                        \
    {                                                                                              \
        return UnaryExpr<Scalar, func##_op<Scalar>, Expr, DerivativeType>(x0.derived());           \
    }                                                                                              \
    template <class Scalar1, std::size_t N>                                                        \
    XAD_INLINE UnaryExpr<Scalar1, func##_op<Scalar1>, ADVar<Scalar1, N>,                           \
                         typename DerivativesTraits<Scalar1, N>::type>(func)(                      \
        const AReal<Scalar1, N>& x)                                                                \
    {                                                                                              \
        return UnaryExpr<Scalar1, func##_op<Scalar1>, ADVar<Scalar1, N>,                           \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(x));      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE FRealDirect<Scalar, N>(func)(const FRealDirect<Scalar, N>& x)                       \
    {                                                                                              \
        return {FReal<Scalar, N>((func)(x.base()))};                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE ARealDirect<Scalar, N>(func)(const ARealDirect<Scalar, N>& x)                       \
    {                                                                                              \
        return {AReal<Scalar, N>((func)(x.base()))};                                               \
    }

#define XAD_MAKE_FPCLASSIFY_FUNC_RET(ret, func, using)                                             \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE ret(func)(const Expression<Scalar, Expr, DerivativeType>& x)                        \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t M = 1>                                                     \
    XAD_INLINE ret(func)(const AReal<Scalar, M>& x)                                                \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE ret(func)(const FReal<Scalar, N>& x)                                                \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE ret(func)(const xad::FRealDirect<Scalar, N>& x)                                     \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE ret(func)(const xad::ARealDirect<Scalar, N>& x)                                     \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, class Expr, class Op, class DerivativeType>                            \
    XAD_INLINE ret(func)(const UnaryExpr<Scalar, Op, Expr, DerivativeType>& x)                     \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, class Op, class Expr1, class Expr2, class Deriv>                       \
    XAD_INLINE ret(func)(const BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& x)                     \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }

#define XAD_MAKE_FPCLASSIFY_FUNC(func, using) XAD_MAKE_FPCLASSIFY_FUNC_RET(bool, func, using)

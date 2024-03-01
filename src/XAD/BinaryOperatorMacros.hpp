/*******************************************************************************

   Macros used for binary operator declarations.

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

#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

namespace xad
{
template <class>
struct AReal;
template <class>
struct ADVar;
template <class>
struct FReal;

template <class Scalar, class Expr, class Enable = void>
struct wrapper_type
{
    typedef FReal<Scalar> type;
};

template <class Scalar, class Expr>
struct wrapper_type<Scalar, Expr, typename std::enable_if<ExprTraits<Expr>::isReverse>::type>
{
    typedef ADVar<Scalar> type;
};

}  // namespace xad

#define XAD_BINARY_OPERATOR(_op, func)                                                             \
    template <class Scalar, class Expr1, class Expr2>                                              \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2>(_op)(                                \
        const Expression<Scalar, Expr1>& a1, const Expression<Scalar, Expr2>& b1)                  \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2>(a1.derived(), b1.derived());         \
    }                                                                                              \
    template <class Scalar>                                                                        \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar>,                                            \
                   ADVar<Scalar>>>::type(_op)(const AReal<Scalar>& a2, const AReal<Scalar>& b2)    \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar>, ADVar<Scalar>>(ADVar<Scalar>(a2),   \
                                                                              ADVar<Scalar>(b2));  \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr>(  \
        _op)(const typename ExprTraits<Expr>::value_type& a3, const Expression<Scalar, Expr>& b3)  \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr>(  \
            typename wrapper_type<Scalar, Expr>::type(a3), b3.derived());                          \
    }                                                                                              \
    template <class Scalar, class Expr>                                                            \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type>(  \
        _op)(const Expression<Scalar, Expr>& a4, const typename ExprTraits<Expr>::value_type& b4)  \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type>(  \
            a4.derived(), typename wrapper_type<Scalar, Expr>::type(b4));                          \
    }                                                                                              \
    template <class Scalar>                                                                        \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, FReal<Scalar>,                                            \
                   FReal<Scalar>>>::type(_op)(const FReal<Scalar>& a2, const FReal<Scalar>& b2)    \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, FReal<Scalar>, FReal<Scalar>>(a2, b2);             \
    }

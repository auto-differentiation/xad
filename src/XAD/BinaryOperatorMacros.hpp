/*******************************************************************************

   Macros used for binary operator declarations.

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
#include <XAD/Vec.hpp>
#include <type_traits>

namespace xad
{
template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct ADVar;
template <class, std::size_t>
struct FReal;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct ARealDirect;

template <class Scalar, class Expr, class Enable = void>
struct wrapper_type
{
    typedef typename ExprTraits<Expr>::value_type type;
};

template <class Scalar, class Expr>
struct wrapper_type<Scalar, Expr, typename std::enable_if<ExprTraits<Expr>::isReverse>::type>
{
    typedef ADVar<Scalar, ExprTraits<Expr>::vector_size> type;
};

template <class T>
struct is_vec : std::false_type
{
};

template <class T, std::size_t N>
struct is_vec<Vec<T, N>> : std::true_type
{
};

}  // namespace xad

#define XAD_BINARY_OPERATOR(_op, func)                                                             \
    template <class Scalar, class Expr1, class Expr2, class DerivativeType>                        \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2, DerivativeType>(_op)(                \
        const Expression<Scalar, Expr1, DerivativeType>& a1,                                       \
        const Expression<Scalar, Expr2, DerivativeType>& b1)                                       \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2, DerivativeType>(a1.derived(),        \
                                                                              b1.derived());       \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar, N>, ADVar<Scalar, N>,                       \
                   typename DerivativesTraits<Scalar, N>::type>>::                                 \
        type(_op)(const AReal<Scalar, N>& a2, const AReal<Scalar, N>& b2)                          \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar, N>, ADVar<Scalar, N>,                \
                          typename DerivativesTraits<Scalar, N>::type>(ADVar<Scalar, N>(a2),       \
                                                                       ADVar<Scalar, N>(b2));      \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr,   \
                          DerivativeType>(_op)(const typename ExprTraits<Expr>::value_type& a3,    \
                                               const Expression<Scalar, Expr, DerivativeType>& b3) \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr,   \
                          DerivativeType>(typename wrapper_type<Scalar, Expr>::type(a3),           \
                                          b3.derived());                                           \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type,   \
                          DerivativeType>(_op)(const Expression<Scalar, Expr, DerivativeType>& a4, \
                                               const typename ExprTraits<Expr>::value_type& b4)    \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type,   \
                          DerivativeType>(a4.derived(),                                            \
                                          typename wrapper_type<Scalar, Expr>::type(b4));          \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>,                       \
                   typename FReal<Scalar, N>::derivative_type>>::                                  \
        type(_op)(const FReal<Scalar, N>& a2, const FReal<Scalar, N>& b2)                          \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>,                \
                          typename FReal<Scalar, N>::derivative_type>(a2, b2);                     \
    }                                                                                              \
    template <class Scalar, class = typename std::enable_if<float_or_double<Scalar>::value>::type, \
              std::size_t N>                                                                       \
    XAD_INLINE FRealDirect<Scalar, N>(_op)(const FRealDirect<Scalar, N>& a,                        \
                                           const FRealDirect<Scalar, N>& b)                        \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a.base(), b.base()))};                                      \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    XAD_INLINE FRealDirect<Scalar, N>(_op)(const FRealDirect<Scalar, N>& a, const T& b)            \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a.base(), b))};                                             \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    XAD_INLINE FRealDirect<Scalar, N>(_op)(const T& a, const FRealDirect<Scalar, N>& b)            \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a, b.base()))};                                             \
    }                                                                                              \
    template <class Scalar, class = typename std::enable_if<float_or_double<Scalar>::value>::type, \
              std::size_t N>                                                                       \
    XAD_INLINE ARealDirect<Scalar, N>(_op)(const ARealDirect<Scalar, N>& a,                        \
                                           const ARealDirect<Scalar, N>& b)                        \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a.base(), b.base()))};                                      \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    XAD_INLINE ARealDirect<Scalar, N>(_op)(const ARealDirect<Scalar, N>& a, const T& b)            \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a.base(), b))};                                             \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    XAD_INLINE ARealDirect<Scalar, N>(_op)(const T& a, const ARealDirect<Scalar, N>& b)            \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a, b.base()))};                                             \
    }

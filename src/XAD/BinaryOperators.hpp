/*******************************************************************************

   Overloaded operators for binary functions.

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

#include <XAD/BinaryExpr.hpp>
#include <XAD/BinaryFunctors.hpp>
#include <XAD/BinaryMathFunctors.hpp>

#include <XAD/BinaryOperatorMacros.hpp>
#include <XAD/Macros.hpp>

// needed here as these are not expressions and we need to specialise
#include <XAD/ARealDirect.hpp>
#include <XAD/FRealDirect.hpp>

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
XAD_INLINE auto smooth_max(const T1& x, const T2& y,
                           const T3& c) -> decltype(0.5 * (x + y + smooth_abs(x - y, c)))
{
    return 0.5 * (x + y + smooth_abs(x - y, c));
}
template <class T1, class T2>
XAD_INLINE auto smooth_max(const T1& x, const T2& y) -> decltype(0.5 * (x + y + smooth_abs(x - y)))
{
    return 0.5 * (x + y + smooth_abs(x - y));
}
template <class T1, class T2, class T3>
XAD_INLINE auto smooth_min(const T1& x, const T2& y,
                           const T3& c) -> decltype(0.5 * (x + y - smooth_abs(x - y, c)))
{
    return 0.5 * (x + y - smooth_abs(x - y, c));
}
template <class T1, class T2>
XAD_INLINE auto smooth_min(const T1& x, const T2& y) -> decltype(0.5 * (x + y - smooth_abs(x - y)))
{
    return 0.5 * (x + y - smooth_abs(x - y));
}

template <class T, class = typename std::enable_if<xad::ExprTraits<T>::isExpr>>
XAD_INLINE auto fma(const T& a, const T& b, const T& c) -> decltype(a * b + c)
{
    return a * b + c;
}

template <
    class T1, class T2, class T3,
    class = typename std::enable_if<(xad::ExprTraits<T1>::isExpr || xad::ExprTraits<T2>::isExpr ||
                                     xad::ExprTraits<T3>::isExpr)>>
XAD_INLINE auto fma(const T1& a, const T2& b, const T3& c) -> decltype(a * b + c)
{
    return a * b + c;
}

/////////// comparisons - they just return bool

#define XAD_COMPARE_OPERATOR(op)                                                                   \
    template <class Scalar, class Expr1, class Expr2, class DerivativeType>                        \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr1, DerivativeType>& a,                \
                                const Expression<Scalar, Expr2, DerivativeType>& b)                \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE bool operator op(const typename ExprTraits<Expr>::value_type& a,                    \
                                const Expression<Scalar, Expr, DerivativeType>& b)                 \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr, DerivativeType>& a,                 \
                                const typename ExprTraits<Expr>::value_type& b)                    \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t M = 1>                                                     \
    XAD_INLINE bool operator op(const AReal<Scalar, M>& a, const AReal<Scalar, M>& b)              \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const FReal<Scalar, N>& a, const FReal<Scalar, N>& b)              \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const FRealDirect<Scalar, N>& a, const FRealDirect<Scalar, N>& b)  \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const Scalar& a, const FRealDirect<Scalar, N>& b)                  \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const FRealDirect<Scalar, N>& a, const Scalar& b)                  \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const ARealDirect<Scalar, N>& a, const ARealDirect<Scalar, N>& b)  \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const Scalar& a, const ARealDirect<Scalar, N>& b)                  \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    XAD_INLINE bool operator op(const ARealDirect<Scalar, N>& a, const Scalar& b)                  \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
                                                                                                   \
    template <class Scalar, class Expr, std::size_t N>                                             \
    XAD_INLINE bool operator op(typename ExprTraits<Scalar>::nested_type a,                        \
                                const FRealDirect<Scalar, N>& b)                                   \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    XAD_INLINE bool operator op(const FRealDirect<Scalar, N>& a,                                   \
                                typename ExprTraits<Scalar>::nested_type b)                        \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    XAD_INLINE bool operator op(typename ExprTraits<Scalar>::nested_type a,                        \
                                const ARealDirect<Scalar, N>& b)                                   \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    XAD_INLINE bool operator op(const ARealDirect<Scalar, N>& a,                                   \
                                typename ExprTraits<Scalar>::nested_type b)                        \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
                                                                                                   \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE bool operator op(typename ExprTraits<Expr>::nested_type a,                          \
                                const Expression<Scalar, Expr, DerivativeType>& b)                 \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    XAD_INLINE bool operator op(const Expression<Scalar, Expr, DerivativeType>& a,                 \
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

template <class Scalar, std::size_t M>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar, M>, ADVar<Scalar, M>> remquo(
    const AReal<Scalar, M>& a, const AReal<Scalar, M>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar, M>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(a), ADVar<Scalar, M>(b), remquo_op<Scalar>(quo));
}

template <class Scalar, std::size_t N>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>> remquo(
    const FReal<Scalar, N>& a, const FReal<Scalar, N>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>>(
        a, b, remquo_op<Scalar>(quo));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> remquo(const xad::FRealDirect<T, N>& a,
                                         const xad::FRealDirect<T, N>& b, int* c)
{
    return xad::FReal<T, N>(remquo(a.base(), b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> remquo(const xad::FRealDirect<T, N>& a, const T& b, int* c)
{
    return xad::FReal<T, N>(remquo(a.base(), b, c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> remquo(const T& a, const xad::FRealDirect<T, N>& b, int* c)
{
    return xad::FReal<T, N>(remquo(a, b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> remquo(const xad::ARealDirect<T, N>& a,
                                         const xad::ARealDirect<T, N>& b, int* c)
{
    return xad::AReal<T, N>(remquo(a.base(), b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> remquo(const xad::ARealDirect<T, N>& a, const T& b, int* c)
{
    return xad::AReal<T, N>(remquo(a.base(), b, c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> remquo(const T& a, const xad::ARealDirect<T, N>& b, int* c)
{
    return xad::AReal<T, N>(remquo(a, b.base(), c));
}

template <class Scalar, class Expr>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, typename ExprTraits<Expr>::value_type, Expr>
remquo(const typename ExprTraits<Expr>::value_type& a, const Expression<Scalar, Expr>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, typename ExprTraits<Expr>::value_type, Expr>(
        typename ExprTraits<Expr>::value_type(a), b.derived(), remquo_op<Scalar>(quo));
}

template <class Scalar, class Expr>
XAD_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename ExprTraits<Expr>::value_type>
remquo(const Expression<Scalar, Expr>& a, const typename ExprTraits<Expr>::value_type& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename ExprTraits<Expr>::value_type>(
        a.derived(), typename ExprTraits<Expr>::value_type(b), remquo_op<Scalar>(quo));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> frexp(const xad::FRealDirect<T, N>& a, int* exp)
{
    return xad::FReal<T, N>(frexp(a.base(), exp));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> ldexp(const xad::FRealDirect<T, N>& a, int b)
{
    return xad::FReal<T, N>(ldexp(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> modf(const xad::FRealDirect<T, N>& a, xad::FRealDirect<T, N>* b)
{
    return xad::FReal<T, N>(modf(a.base(), &b->base()));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::FRealDirect<T, N> modf(const xad::FRealDirect<T, N>& a, T* b)
{
    return xad::FReal<T, N>(modf(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> frexp(const xad::ARealDirect<T, N>& a, int* exp)
{
    return xad::AReal<T, N>(frexp(a.base(), exp));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> ldexp(const xad::ARealDirect<T, N>& a, int b)
{
    return xad::AReal<T, N>(ldexp(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> modf(const xad::ARealDirect<T, N>& a, xad::ARealDirect<T, N>* b)
{
    return xad::AReal<T, N>(modf(a.base(), &b->base()));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
XAD_INLINE xad::ARealDirect<T, N> modf(const xad::ARealDirect<T, N>& a, T* b)
{
    return xad::AReal<T, N>(modf(a.base(), b));
}

#undef XAD_BINARY_OPERATOR
#undef XAD_COMPARE_OPERATOR
}  // namespace xad

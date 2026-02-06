/*******************************************************************************

   Overloads of operators that translate to unary functors.

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

#include <XAD/BinaryExpr.hpp>
#include <XAD/UnaryExpr.hpp>
#include <XAD/UnaryMathFunctors.hpp>

#include <XAD/Macros.hpp>
#include <XAD/UnaryOperatorMacros.hpp>

#ifdef XAD_ENABLE_JIT
#include <XAD/ABool.hpp>
#include <XAD/JITCompiler.hpp>
#endif

namespace xad
{
template <class, std::size_t>
struct FReal;
template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct ARealDirect;

// unary plus - does nothing
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE const Expression<Scalar, Expr, DerivativeType>& operator+(
    const Expression<Scalar, Expr, DerivativeType>& a)
{
    return a;
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE const ADVar<Scalar, M> operator+(const ADVar<Scalar, M>& a)
{
    return ADVar<Scalar, M>(a);
}

XAD_UNARY_OPERATOR(operator-, negate_op)
XAD_UNARY_BINSCAL(operator+, scalar_add_op)
XAD_UNARY_BINSCAL(operator*, scalar_prod_op)
XAD_UNARY_BINSCAL1(operator-, scalar_sub1_op)
XAD_UNARY_BINSCAL2(operator-, scalar_sub2_op)
XAD_UNARY_BINSCAL1(operator/, scalar_div1_op)
XAD_UNARY_BINSCAL2(operator/, scalar_div2_op)
XAD_UNARY_BINSCAL1(pow, scalar_pow1_op)
XAD_UNARY_BINSCAL2(pow, scalar_pow2_op)
XAD_UNARY_BINSCAL1(smooth_abs, scalar_smooth_abs1_op)
XAD_UNARY_BINSCAL2(smooth_abs, scalar_smooth_abs2_op)

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
smooth_abs(const Expression<Scalar, Expr, DerivativeType>& a)
{
    return smooth_abs(a, typename ExprTraits<Expr>::nested_type(0.001));
}

template <class Scalar, std::size_t M>
XAD_INLINE UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename AReal<Scalar, M>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
smooth_abs(const AReal<Scalar, M>& a)
{
    return smooth_abs(a, typename AReal<Scalar, M>::nested_type(0.001));
}

XAD_UNARY_BINSCAL1(fmod, scalar_fmod1_op)
XAD_UNARY_BINSCAL2(fmod, scalar_fmod2_op)
XAD_UNARY_BINSCAL1(atan2, scalar_atan21_op)
XAD_UNARY_BINSCAL2(atan2, scalar_atan22_op)
XAD_UNARY_BINSCAL1(nextafter, scalar_nextafter1_op)
XAD_UNARY_BINSCAL2(nextafter, scalar_nextafter2_op)
XAD_UNARY_BINSCAL1(hypot, scalar_hypot1_op)
XAD_UNARY_BINSCAL2(hypot, scalar_hypot2_op)

// pown (integral exponents)
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, Expr, DerivativeType> pown(
    const Expression<Scalar, Expr, DerivativeType>& x, int y)
{
    return pow(x, y);
}
template <class Scalar, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
pown(const AReal<Scalar, M>& x, int y)
{
    return pow(x, y);
}

// ldexp
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, Expr, DerivativeType> ldexp(
    const Expression<Scalar, Expr, DerivativeType>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, Expr, DerivativeType>(x.derived(),
                                                                     ldexp_op<Scalar>(y));
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
ldexp(const AReal<Scalar, M>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  ldexp_op<Scalar>(y));
}

// frexp
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, Expr, DerivativeType> frexp(
    const Expression<Scalar, Expr, DerivativeType>& x, int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, Expr, DerivativeType>(x.derived(),
                                                                     frexp_op<Scalar>(exp));
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
frexp(const AReal<Scalar, M>& x, int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  frexp_op<Scalar>(exp));
}

// modf - only enabled if iptr is nested type (double) or Scalar
template <class Scalar, class Expr, class T, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, Expr, DerivativeType> modf(
    const Expression<Scalar, Expr, DerivativeType>& x, T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, Expr, DerivativeType>(x.derived(),
                                                                       modf_op<Scalar, T>(iptr));
}

template <class Scalar, class T, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
modf(const AReal<Scalar, M>& x, T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  modf_op<Scalar, T>(iptr));
}

// we put max/min here explicitly, as the 2 arguments to them must match
// and we need to avoid conflicts with the standard versions

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(max)(
    Scalar a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(max)(T a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class T, class AT, std::size_t M = 1>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, M>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, M>>>::type(max)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(b3), scalar_max_op<Scalar, Scalar>(a3));
}


template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(max)(
    const Expression<Scalar, Expr, DerivativeType>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(max)(const Expression<Scalar, Expr, DerivativeType>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class T, class AT, std::size_t N>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, N>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, N>>>::type(max)(const AT& a1, T b1)
{
    return max(b1, a1);
}



template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(min)(
    Scalar a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(min)(T a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class T, class AT, std::size_t M = 1>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, M>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, M>>>::type(min)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(b3), scalar_min_op<Scalar, Scalar>(a3));
}


template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(min)(
    const Expression<Scalar, Expr, DerivativeType>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(min)(const Expression<Scalar, Expr, DerivativeType>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class T, class AT>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>, typename AT::derivatives_type>>::type(min)(const AT& a1, T b1)
{
    return min(b1, a1);
}

XAD_UNARY_BINSCAL(fmax, scalar_fmax_op)
XAD_UNARY_BINSCAL(fmin, scalar_fmin_op)

/////////// Math functions

XAD_MAKE_UNARY_FUNC(degrees)
XAD_MAKE_UNARY_FUNC(radians)
XAD_MAKE_UNARY_FUNC(cos)
XAD_MAKE_UNARY_FUNC(sin)
XAD_MAKE_UNARY_FUNC(log)
XAD_MAKE_UNARY_FUNC(log10)
XAD_MAKE_UNARY_FUNC(log2)
XAD_MAKE_UNARY_FUNC(asin)
XAD_MAKE_UNARY_FUNC(acos)
XAD_MAKE_UNARY_FUNC(atan)
XAD_MAKE_UNARY_FUNC(sinh)
XAD_MAKE_UNARY_FUNC(cosh)
XAD_MAKE_UNARY_FUNC(expm1)
XAD_MAKE_UNARY_FUNC(exp2)
XAD_MAKE_UNARY_FUNC(log1p)
XAD_MAKE_UNARY_FUNC(asinh)
XAD_MAKE_UNARY_FUNC(acosh)
XAD_MAKE_UNARY_FUNC(atanh)
XAD_MAKE_UNARY_FUNC(abs)
XAD_MAKE_UNARY_FUNC(fabs)
XAD_MAKE_UNARY_FUNC(floor)
XAD_MAKE_UNARY_FUNC(ceil)
XAD_MAKE_UNARY_FUNC(trunc)
XAD_MAKE_UNARY_FUNC(round)
XAD_MAKE_UNARY_FUNC(exp)
XAD_MAKE_UNARY_FUNC(tanh)
XAD_MAKE_UNARY_FUNC(sqrt)
XAD_MAKE_UNARY_FUNC(cbrt)
XAD_MAKE_UNARY_FUNC(tan)
XAD_MAKE_UNARY_FUNC(erf)
XAD_MAKE_UNARY_FUNC(erfc)

// no special AD treatement here, but we need the overloads

XAD_MAKE_FPCLASSIFY_FUNC(isinf, using std::isinf)
XAD_MAKE_FPCLASSIFY_FUNC(isnan, using std::isnan)
XAD_MAKE_FPCLASSIFY_FUNC(isfinite, using std::isfinite)
XAD_MAKE_FPCLASSIFY_FUNC(signbit, using std::signbit)
XAD_MAKE_FPCLASSIFY_FUNC(isnormal, using std::isnormal)
XAD_MAKE_FPCLASSIFY_FUNC(__isinf, )
XAD_MAKE_FPCLASSIFY_FUNC(__isnan, )
XAD_MAKE_FPCLASSIFY_FUNC(__isfinite, )
XAD_MAKE_FPCLASSIFY_FUNC_RET(int, fpclassify, using std::fpclassify)
XAD_MAKE_FPCLASSIFY_FUNC_RET(long, lround, using std::lround)
XAD_MAKE_FPCLASSIFY_FUNC_RET(long long, llround, using std::llround)

XAD_UNARY_BINSCAL1(remainder, scalar_remainder1_op)
XAD_UNARY_BINSCAL2(remainder, scalar_remainder2_op)

template <class Scalar, class Expr, class T2, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
        !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr, DerivativeType>>::type
remquo(const T2& a, const Expression<Scalar, Expr, DerivativeType>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr, DerivativeType>(
        b.derived(), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
remquo(typename ExprTraits<Expr>::nested_type a, const Expression<Scalar, Expr, DerivativeType>& b,
       int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>(
        b.derived(), scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>(a, quo));
}
template <class Scalar, class T2, std::size_t M = 1>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar, M>,
                                      typename DerivativesTraits<Scalar, M>::type>>::type
    remquo(const T2& a, const AReal<Scalar, M>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(b), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     AReal<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
remquo(typename ExprTraits<Scalar>::nested_type a, const AReal<Scalar, M>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     AReal<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(
        AReal<Scalar, M>(b),
        scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>(a, quo));
}

template <class Scalar, class Expr, class T2, class DerivativeType>
XAD_INLINE typename std::enable_if<
    std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
        !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr, DerivativeType>>::type
remquo(const Expression<Scalar, Expr, DerivativeType>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr, DerivativeType>(
        a.derived(), scalar_remquo2_op<Scalar, T2>(b, quo));
}
template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
remquo(const Expression<Scalar, Expr, DerivativeType>& a, typename ExprTraits<Expr>::nested_type b,
       int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>(
        a.derived(), scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>(b, quo));
}
template <class Scalar, class T2, std::size_t M = 1>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar, M>,
                                      typename DerivativesTraits<Scalar, M>::type>>::type
    remquo(const AReal<Scalar, M>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(a), scalar_remquo2_op<Scalar, T2>(b, quo));
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
remquo(const AReal<Scalar, M>& a, typename ExprTraits<Scalar>::nested_type b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(a),
        scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>(b, quo));
}

template <class Scalar, class Derived, class Deriv>
XAD_INLINE int ilogb(const Expression<Scalar, Derived, Deriv>& x)
{
    using std::ilogb;
    return ilogb(x.value());
}

template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename ExprTraits<Derived>::value_type scalbn(
    const Expression<Scalar, Derived, Deriv>& x, int exp)
{
    using std::scalbn;
    using T = typename ExprTraits<Derived>::value_type;
    return T(x) * scalbn(1.0, exp);
}

template <class Scalar, class Derived, class T2, class DerivativeType>
XAD_INLINE typename ExprTraits<Derived>::value_type copysign(
    const Expression<Scalar, Derived, DerivativeType>& x, const T2& y)
{
    using T = typename ExprTraits<Derived>::value_type;
    bool sign = signbit(y);
    if (x < 0)
    {
        if (sign)
            return T(x);
        else
            return T(-x);
    }
    else
    {
        if (sign)
            return T(-x);
        else
            return T(x);
    }
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar copysign(double x, const FRealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar copysign(float x, const FRealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar copysign(double x, const ARealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar copysign(float x, const ARealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, class Derived, class DerivativeType>
XAD_INLINE double copysign(double x, const Expression<Scalar, Derived, DerivativeType>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, class Derived, class DerivativeType>
XAD_INLINE float copysign(float x, const Expression<Scalar, Derived, DerivativeType>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

#undef XAD_UNARY_BINSCAL
#undef XAD_UNARY_BINSCAL1
#undef XAD_UNARY_BINSCAL2
#undef XAD_MAKE_UNARY_FUNC
#undef XAD_MAKE_FPCLASSIFY_FUNC
#undef XAD_MAKE_FPCLASSIFY_FUNC_RET
}  // namespace xad

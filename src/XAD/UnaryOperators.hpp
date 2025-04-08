/*******************************************************************************

   Overloads of operators that translate to unary functors.

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
#include <XAD/UnaryExpr.hpp>
#include <XAD/UnaryMathFunctors.hpp>

#include <XAD/Macros.hpp>
#include <XAD/UnaryOperatorMacros.hpp>

namespace xad
{
template <class>
struct FReal;
template <class>
struct AReal;

// unary plus - does nothing
template <class Scalar, class Expr>
XAD_INLINE const Expression<Scalar, Expr>& operator+(const Expression<Scalar, Expr>& a)
{
    return a;
}

template <class Scalar>
XAD_INLINE const ADVar<Scalar> operator+(const AReal<Scalar>& a)
{
    return ADVar<Scalar>(a);
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

template <class Scalar, class Expr>
XAD_INLINE
    UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename ExprTraits<Expr>::nested_type>, Expr>
    smooth_abs(const Expression<Scalar, Expr>& a)
{
    return smooth_abs(a, typename ExprTraits<Expr>::nested_type(0.001));
}

template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename AReal<Scalar>::nested_type>,
                     ADVar<Scalar>>
smooth_abs(const AReal<Scalar>& a)
{
    return smooth_abs(a, typename AReal<Scalar>::nested_type(0.001));
}

XAD_UNARY_BINSCAL1(fmod, scalar_fmod1_op)
XAD_UNARY_BINSCAL2(fmod, scalar_fmod2_op)
XAD_UNARY_BINSCAL1(atan2, scalar_atan21_op)
XAD_UNARY_BINSCAL2(atan2, scalar_atan22_op)
XAD_UNARY_BINSCAL1(nextafter, scalar_nextafter1_op)
XAD_UNARY_BINSCAL2(nextafter, scalar_nextafter2_op)
XAD_UNARY_BINSCAL1(hypot, scalar_hypot1_op)
XAD_UNARY_BINSCAL2(hypto, scalar_hypot2_op)

// pown (integral exponents)
template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, Expr> pown(
    const Expression<Scalar, Expr>& x, int y)
{
    return pow(x, y);
}
template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, ADVar<Scalar>> pown(
    const AReal<Scalar>& x, int y)
{
    return pow(x, y);
}

// ldexp
template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, Expr> ldexp(const Expression<Scalar, Expr>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, Expr>(x.derived(), ldexp_op<Scalar>(y));
}

template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, ADVar<Scalar>> ldexp(const AReal<Scalar>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, ADVar<Scalar>>(ADVar<Scalar>(x),
                                                              ldexp_op<Scalar>(y));
}

// frexp
template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, Expr> frexp(const Expression<Scalar, Expr>& x,
                                                           int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, Expr>(x.derived(), frexp_op<Scalar>(exp));
}

template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, ADVar<Scalar>> frexp(const AReal<Scalar>& x,
                                                                    int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, ADVar<Scalar>>(ADVar<Scalar>(x),
                                                              frexp_op<Scalar>(exp));
}

// modf - only enabled if iptr is nested type (double) or Scalar
template <class Scalar, class Expr, class T>
XAD_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, Expr> modf(const Expression<Scalar, Expr>& x,
                                                            T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, Expr>(x.derived(), modf_op<Scalar, T>(iptr));
}

template <class Scalar, class T>
XAD_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, ADVar<Scalar>> modf(const AReal<Scalar>& x,
                                                                     T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, ADVar<Scalar>>(ADVar<Scalar>(x),
                                                                modf_op<Scalar, T>(iptr));
}

// we put max/min here explicitly, as the 2 arguments to them must match
// and we need to avoid conflicts with the standard versions

template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(max)(
    Scalar a2, const Expression<Scalar, Expr>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T>
XAD_INLINE typename std::enable_if<std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
                                   UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>>::
    type(max)(T a2, const Expression<Scalar, Expr>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class T, class AT>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>>>::type(max)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, ADVar<Scalar>>(
        ADVar<Scalar>(b3), scalar_max_op<Scalar, Scalar>(a3));
}

template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(max)(
    const Expression<Scalar, Expr>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T>
XAD_INLINE typename std::enable_if<std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
                                   UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>>::
    type(max)(const Expression<Scalar, Expr>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class T, class AT>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>>>::type(max)(const AT& a1, T b1)
{
    return max(b1, a1);
}

template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(min)(
    Scalar a2, const Expression<Scalar, Expr>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T>
XAD_INLINE typename std::enable_if<std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
                                   UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>>::
    type(min)(T a2, const Expression<Scalar, Expr>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class T, class AT>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>>>::type(min)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, ADVar<Scalar>>(
        ADVar<Scalar>(b3), scalar_min_op<Scalar, Scalar>(a3));
}

template <class Scalar, class Expr>
XAD_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(min)(
    const Expression<Scalar, Expr>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T>
XAD_INLINE typename std::enable_if<std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
                                   UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>>::
    type(min)(const Expression<Scalar, Expr>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class T, class AT>
XAD_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>>>::type(min)(const AT& a1, T b1)
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

template <class Scalar, class Expr, class T2>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr>>::type
    remquo(const T2& a, const Expression<Scalar, Expr>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr>(
        b.derived(), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar, class Expr>
XAD_INLINE
    UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>, Expr>
    remquo(typename ExprTraits<Expr>::nested_type a, const Expression<Scalar, Expr>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr>(
        b.derived(), scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>(a, quo));
}
template <class Scalar, class T2>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar>>>::type
    remquo(const T2& a, const AReal<Scalar>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar>>(
        ADVar<Scalar>(b), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar>>
remquo(typename ExprTraits<Scalar>::nested_type a, const AReal<Scalar>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar>>(
        ADVar<Scalar>(b),
        scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>(a, quo));
}

template <class Scalar, class Expr, class T2>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr>>::type
    remquo(const Expression<Scalar, Expr>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr>(
        a.derived(), scalar_remquo2_op<Scalar, T2>(b, quo));
}
template <class Scalar, class Expr>
XAD_INLINE
    UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>, Expr>
    remquo(const Expression<Scalar, Expr>& a, typename ExprTraits<Expr>::nested_type b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr>(
        a.derived(), scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>(b, quo));
}
template <class Scalar, class T2>
XAD_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar>>>::type
    remquo(const AReal<Scalar>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar>>(
        ADVar<Scalar>(a), scalar_remquo2_op<Scalar, T2>(b, quo));
}

template <class Scalar>
XAD_INLINE UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar>>
remquo(const AReal<Scalar>& a, typename ExprTraits<Scalar>::nested_type b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar>>(
        ADVar<Scalar>(a),
        scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>(b, quo));
}

#if defined(_MSC_VER) && (_MSC_VER <= 1900)
// full specialisations as compilers otherwise pick up standard version
// NOTE: They are only covering the most common cases

XAD_INLINE UnaryExpr<double, scalar_remquo2_op<double, double>, ADVar<double>> remquo(
    const AReal<double>& a, double b, int* quo)
{
    return UnaryExpr<double, scalar_remquo2_op<double, double>, ADVar<double>>(
        ADVar<double>(a), scalar_remquo2_op<double, double>(b, quo));
}

XAD_INLINE UnaryExpr<double, scalar_remquo2_op<double, double>, FReal<double>> remquo(
    const FReal<double>& a, double b, int* quo)
{
    return UnaryExpr<double, scalar_remquo2_op<double, double>, FReal<double>>(
        a, scalar_remquo2_op<double, double>(b, quo));
}

XAD_INLINE UnaryExpr<double, scalar_remquo1_op<double, double>, ADVar<double>> remquo(
    double a, const AReal<double>& b, int* quo)
{
    return UnaryExpr<double, scalar_remquo1_op<double, double>, ADVar<double>>(
        ADVar<double>(b), scalar_remquo1_op<double, double>(a, quo));
}

XAD_INLINE UnaryExpr<double, scalar_remquo1_op<double, double>, FReal<double>> remquo(
    double a, const FReal<double>& b, int* quo)
{
    return UnaryExpr<double, scalar_remquo1_op<double, double>, FReal<double>>(
        b, scalar_remquo1_op<double, double>(a, quo));
}
#endif

template <class Scalar, class Derived>
XAD_INLINE int ilogb(const Expression<Scalar, Derived>& x)
{
    using std::ilogb;
    return ilogb(x.value());
}

template <class Scalar, class Derived>
XAD_INLINE typename ExprTraits<Derived>::value_type scalbn(const Expression<Scalar, Derived>& x,
                                                           int exp)
{
    using std::scalbn;
    using T = typename ExprTraits<Derived>::value_type;
    return T(x * scalbn(1.0, exp));
}

#ifndef _WIN32
template <class Scalar, class Derived, class T2>
XAD_INLINE typename ExprTraits<Derived>::value_type copysign(const Expression<Scalar, Derived>& x,
                                                             const T2& y)
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

template <class Scalar, class Derived>
XAD_INLINE double copysign(double x, const Expression<Scalar, Derived>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, class Derived>
XAD_INLINE float copysign(float x, const Expression<Scalar, Derived>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}
#endif


#undef XAD_UNARY_BINSCAL
#undef XAD_UNARY_BINSCAL1
#undef XAD_UNARY_BINSCAL2
#undef XAD_MAKE_UNARY_FUNC
#undef XAD_MAKE_FPCLASSIFY_FUNC
#undef XAD_MAKE_FPCLASSIFY_FUNC_RET
} // namespace xad

#ifdef _WIN32

#include <cmath>

namespace std {
    inline xad::AReal<double> copysign(const xad::AReal<double>& x, const xad::AReal<double>& y) noexcept {
        return ::xad::copysign(x, y);
    }
}
#endif
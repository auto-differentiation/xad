/*******************************************************************************

   Unary functors for math functions.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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
#include <XAD/MathFunctions.hpp>
#include <XAD/UnaryFunctors.hpp>
#include <type_traits>


namespace xad
{
// degrees and radians are mapped to products
template <class Scalar>
struct degrees_op : scalar_prod_op<Scalar, Scalar>
{
    XAD_INLINE degrees_op()
        : scalar_prod_op<Scalar, Scalar>(Scalar(57.2957795130823208767981548141051703324054725))
    {
    }
};

template <class Scalar>
struct radians_op : scalar_prod_op<Scalar, Scalar>
{
    XAD_INLINE radians_op()
        : scalar_prod_op<Scalar, Scalar>(Scalar(0.0174532925199432957692369076848861271344287))
    {
    }
};

#define XAD_MAKE_UNARY_FUNCTOR(func, der)                                                          \
    template <class Scalar>                                                                        \
    struct func##_op                                                                               \
    {                                                                                              \
        XAD_INLINE Scalar operator()(const Scalar& a) const { return func(a); }                    \
        XAD_INLINE Scalar derivative(const Scalar& a) const                                        \
        {                                                                                          \
            XAD_UNUSED_VARIABLE(a);                                                                \
            return der;                                                                            \
        }                                                                                          \
    };

XAD_MAKE_UNARY_FUNCTOR(cos, -sin(a))
XAD_MAKE_UNARY_FUNCTOR(sin, cos(a));
XAD_MAKE_UNARY_FUNCTOR(log, Scalar(1) / a)
XAD_MAKE_UNARY_FUNCTOR(log10, Scalar(0.434294481903251827651128918916605082294397) / a)
XAD_MAKE_UNARY_FUNCTOR(log2, Scalar(1.442695040888963407359924681001892137426646) / a)
XAD_MAKE_UNARY_FUNCTOR(asin, Scalar(1) / sqrt(Scalar(1) - a * a))
XAD_MAKE_UNARY_FUNCTOR(acos, -Scalar(1) / sqrt(Scalar(1) - a * a))
XAD_MAKE_UNARY_FUNCTOR(atan, Scalar(1) / (Scalar(1) + a * a))
XAD_MAKE_UNARY_FUNCTOR(sinh, cosh(a))
XAD_MAKE_UNARY_FUNCTOR(cosh, sinh(a))
XAD_MAKE_UNARY_FUNCTOR(expm1, exp(a))
XAD_MAKE_UNARY_FUNCTOR(exp2, Scalar(0.6931471805599453094172321214581765680755001) * exp2(a))
XAD_MAKE_UNARY_FUNCTOR(log1p, Scalar(1) / (Scalar(1) + a))
XAD_MAKE_UNARY_FUNCTOR(asinh, Scalar(1) / sqrt(a * a + Scalar(1)))
XAD_MAKE_UNARY_FUNCTOR(acosh, Scalar(1) / sqrt(a * a - Scalar(1)))
XAD_MAKE_UNARY_FUNCTOR(atanh, Scalar(1) / (Scalar(1) - a * a))

XAD_MAKE_UNARY_FUNCTOR(erf, Scalar(1.1283791670955125738961589031215451716881013) * exp(-a * a))
XAD_MAKE_UNARY_FUNCTOR(erfc, Scalar(-1.1283791670955125738961589031215451716881013) * exp(-a * a))
XAD_MAKE_UNARY_FUNCTOR(abs, (a > Scalar()) - (a < Scalar()))
XAD_MAKE_UNARY_FUNCTOR(floor, Scalar())
XAD_MAKE_UNARY_FUNCTOR(ceil, Scalar())
XAD_MAKE_UNARY_FUNCTOR(trunc, Scalar())
XAD_MAKE_UNARY_FUNCTOR(round, Scalar())

// expressed in terms of result
#define XAD_MAKE_UNARY_FUNCTOR_RES(func, der)                                                      \
    template <class Scalar>                                                                        \
    struct func##_op                                                                               \
    {                                                                                              \
        XAD_INLINE Scalar operator()(const Scalar& a) const { return func(a); }                    \
        XAD_INLINE Scalar derivative(const Scalar& a, const Scalar& v) const                       \
        {                                                                                          \
            XAD_UNUSED_VARIABLE(a);                                                                \
            return der;                                                                            \
        }                                                                                          \
    };                                                                                             \
    template <class Scalar2>                                                                       \
    struct OperatorTraits<func##_op<Scalar2> >                                                     \
    {                                                                                              \
        enum                                                                                       \
        {                                                                                          \
            useResultBasedDerivatives = 1                                                          \
        };                                                                                         \
    };

XAD_MAKE_UNARY_FUNCTOR_RES(exp, v)
XAD_MAKE_UNARY_FUNCTOR_RES(tanh, Scalar(1) - v * v)
XAD_MAKE_UNARY_FUNCTOR_RES(sqrt, Scalar(0.5) / v)
XAD_MAKE_UNARY_FUNCTOR_RES(cbrt, Scalar(1) / Scalar(3) / (v * v))

// tangent
template <class Scalar>
struct tan_op
{
    XAD_INLINE Scalar operator()(const Scalar& a) const { return tan(a); }
    XAD_INLINE Scalar derivative(const Scalar& a) const
    {
        Scalar tmp = Scalar(1) / cos(a);
        return tmp * tmp;
    }
};

// same as abs
template <class Scalar>
struct fabs_op : abs_op<Scalar>
{
};

#define XAD_MAKE_UNARY_BINFUNCTOR(func, dera, derb)                                                \
    template <class Scalar, class T2>                                                              \
    struct scalar_##func##2_op                                                                     \
    {                                                                                              \
        XAD_INLINE explicit scalar_##func##2_op(const T2& b_t) : b(b_t) {}                         \
        XAD_INLINE Scalar operator()(const Scalar& a) const { return func(a, b); }                 \
        XAD_INLINE Scalar derivative(const Scalar& a, const Scalar& v) const                       \
        {                                                                                          \
            XAD_UNUSED_VARIABLE(a);                                                                \
            XAD_UNUSED_VARIABLE(v);                                                                \
            return dera;                                                                           \
        }                                                                                          \
        T2 b;                                                                                      \
    };                                                                                             \
    template <class Scalar, class T2>                                                              \
    struct scalar_##func##1_op                                                                     \
    {                                                                                              \
        XAD_INLINE explicit scalar_##func##1_op(const T2& b_t) : b(b_t) {}                         \
        XAD_INLINE Scalar operator()(const Scalar& a) const { return func(b, a); }                 \
        XAD_INLINE Scalar derivative(const Scalar& a, const Scalar& v) const                       \
        {                                                                                          \
            XAD_UNUSED_VARIABLE(a);                                                                \
            XAD_UNUSED_VARIABLE(v);                                                                \
            return derb;                                                                           \
        }                                                                                          \
        T2 b;                                                                                      \
    };                                                                                             \
    template <class Scalar, class T2>                                                              \
        struct OperatorTraits<scalar_##func##1_op < Scalar, T2> >                                  \
    {                                                                                              \
        enum                                                                                       \
        {                                                                                          \
            useResultBasedDerivatives = 1                                                          \
        };                                                                                         \
    };                                                                                             \
    template <class Scalar, class T2>                                                              \
        struct OperatorTraits<scalar_##func##2_op < Scalar, T2> >                                  \
    {                                                                                              \
        enum                                                                                       \
        {                                                                                          \
            useResultBasedDerivatives = 1                                                          \
        };                                                                                         \
    };

XAD_MAKE_UNARY_BINFUNCTOR(pow, Scalar(b) * Scalar(pow(a, b - T2(1))), log(b) * v)
XAD_MAKE_UNARY_BINFUNCTOR(fmod, Scalar(1), -floor(Scalar(b) / a))
XAD_MAKE_UNARY_BINFUNCTOR(atan2, Scalar(b) / (a * a + Scalar(b * b)),
                          -Scalar(b) / (a * a + Scalar(b * b)))
XAD_MAKE_UNARY_BINFUNCTOR(nextafter, Scalar(1), Scalar(0))

template <class Scalar>
struct ldexp_op
{
    XAD_INLINE explicit ldexp_op(int exp) : exp_(exp) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return ldexp(a, exp_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1 << exp_); }
    int exp_;
};

template <class Scalar>
struct frexp_op
{
    XAD_INLINE explicit frexp_op(int* exp) : exp_(exp) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return frexp(a, exp_); }
    XAD_INLINE Scalar derivative(const Scalar&) const
    {
        // we assume that operator() and derivative are called in succession and in
        // this order, which is always the case in XAD. Therefore we can read the
        // value in *exp_ here to calculate the derivative
        return Scalar(1.0) / Scalar(1 << *exp_);
    }
    int* exp_;
};

namespace detail
{
template <class Scalar, class T, bool Enable>
struct modf_helper
{
    XAD_INLINE static Scalar apply(const Scalar& a, T* iptr) { return modf(a, iptr); }
};

template <class Scalar, class T>
struct modf_helper<Scalar, T, true>
{
    XAD_INLINE static Scalar apply(const Scalar& a, T* iptr)
    {
        typename ExprTraits<T>::nested_type iptr_t;
        Scalar ret = modf(a, &iptr_t);
        *iptr = iptr_t;
        return ret;
    }
};
}  // namespace detail

template <class Scalar, class T>
struct modf_op
{
    XAD_INLINE explicit modf_op(T* iptr) : iptr_(iptr) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const
    {
        return detail::modf_helper<Scalar, T, ExprTraits<T>::isExpr>::apply(a, iptr_);
    }
    XAD_INLINE Scalar derivative(const Scalar&) const
    {
        // as the return value is the fractional part, this is equivalent to
        // a - *iptr, and since iptr can be considered a constant, the derivative
        // is just 1 in all cases
        return Scalar(1.0);
    }
    T* iptr_;
};

template <class Scalar, class T2>
struct scalar_smooth_abs2_op
{
    XAD_INLINE explicit scalar_smooth_abs2_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const
    {
        if (abs(a) > b_)
            return abs(a);
        if (a < Scalar())
        {
            return a * a * (Scalar(2) / Scalar(b_) + a / Scalar(b_ * b_));
        }
        else
        {
            return a * a * (Scalar(2) / Scalar(b_) - a / Scalar(b_ * b_));
        }
    }
    XAD_INLINE Scalar derivative(const Scalar& x) const
    {
        if (x > b_)
            return Scalar(1);
        else if (x < -b_)
            return Scalar(-1);
        else if (x < Scalar())
        {
            return x / Scalar(b_ * b_) * (Scalar(3) * x + Scalar(4) * b_);
        }
        else
            return -x / Scalar(b_ * b_) * (Scalar(3) * x - Scalar(4) * b_);
    }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_smooth_abs1_op
{
    XAD_INLINE explicit scalar_smooth_abs1_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const
    {
        if (abs(b_) > a)
            return abs(b_);
        if (b_ < Scalar())
        {
            return b_ * b_ * (Scalar(2) / a + Scalar(b_) / (a * a));
        }
        else
        {
            return b_ * b_ * (Scalar(2) / a - Scalar(b_) / (a * a));
        }
    }
    XAD_INLINE Scalar derivative(const Scalar& c) const
    {
        if (b_ > c || b_ < -c)
            return Scalar();
        else if (b_ < Scalar())
        {
            return -Scalar(2) * Scalar(b_ * b_) * (c + Scalar(b_)) / (c * c * c);
        }
        else
        {
            return -Scalar(2) * Scalar(b_ * b_) * (c - Scalar(b_)) / (c * c * c);
        }
    }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_max_op
{
    XAD_INLINE explicit scalar_max_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return max(a, b_); }
    XAD_INLINE Scalar derivative(const Scalar& a) const
    {
        // (1 + ((a-b)>0 - (a-b)<0)) / 2
        return (Scalar(1) + (Scalar((a - b_) > Scalar()) - Scalar((a - b_) < Scalar()))) /
               Scalar(2);
    }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_fmax_op : scalar_max_op<Scalar, T2>
{
    XAD_INLINE explicit scalar_fmax_op(const T2& b) : scalar_max_op<Scalar, T2>(b) {}
};

template <class Scalar, class T2>
struct scalar_min_op
{
    XAD_INLINE explicit scalar_min_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return min(a, b_); }
    XAD_INLINE Scalar derivative(const Scalar& a) const
    {
        // (1 - ((a-b)>0 - (a-b)<0)) / 2
        return (Scalar(1) - (Scalar((a - b_) > Scalar()) - Scalar((a - b_) < Scalar()))) /
               Scalar(2);
    }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_fmin_op : scalar_min_op<Scalar, T2>
{
    XAD_INLINE explicit scalar_fmin_op(const T2& b) : scalar_min_op<Scalar, T2>(b) {}
};

template <class Scalar, class T2>
struct scalar_remainder1_op
{
    XAD_INLINE explicit scalar_remainder1_op(const T2& b) : b_(b) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return remainder(b_, a); }
    XAD_INLINE Scalar derivative(const Scalar& a) const
    {
        // function is rare enough that there's no need to optimize this better
        int n_;
        using std::remquo;
        (void)remquo(b_, a, &n_);
        return Scalar(-n_);
    }
    T2 b_;
};

template <class Scalar, class T2>
struct scalar_remainder2_op
{
    XAD_INLINE explicit scalar_remainder2_op(const T2& b) : b_(b) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return remainder(a, b_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    T2 b_;
};

template <class Scalar, class T2>
struct scalar_remquo1_op
{
    XAD_INLINE scalar_remquo1_op(const T2& b, int* quo) : b_(b), quo_(quo), q_() {}
    XAD_INLINE Scalar operator()(const Scalar& a) const
    {
        using std::remquo;
        Scalar v = remquo(b_, a, &q_);
        *quo_ = q_;
        return v;
    }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(-q_); }
    T2 b_;
    int* quo_;
    mutable int q_;
};

template <class Scalar, class T2>
struct scalar_remquo2_op
{
    XAD_INLINE scalar_remquo2_op(const T2& b, int* quo) : b_(b), quo_(quo) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { using std::remquo; return remquo(a, b_, quo_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    T2 b_;
    int* quo_;
};

}  // namespace xad

// FrExp needs special treatment - it's not really unary, but operand is not
// an AD type

/*
 * TODO:
hypot(T x, T y)
fma
logb
tgamma
lgamma

scalbln
rint
lrint
llrint
nearbyint
copysign
nan
nexttoward
fdim
isnormal

isgreater
isgreaterequal
isless
islessequal
islessgreater
isunordered

CHECK:
- do we need to capture the exponent in frexp somewhere?

OTHERS:
cospi
sine
sinpi
tanpi
sincos(T x, T*sinx, T* cosx)
acospi
asinpi
atanpi
atan2pi

clamp(T x, float minval, float maxval)
sign
step(T edge, T x)

exp10
powr
rootn( T x, int y)
rsqrt

*/

#undef XAD_MAKE_UNARY_FUNCTOR_RES
#undef XAD_MAKE_UNARY_FUNCTOR
#undef XAD_MAKE_UNARY_BINFUNCTOR

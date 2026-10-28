/*******************************************************************************

   Functors capturing unary expressions.

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
#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>

namespace xad
{

template <class Scalar>
struct negate_op
{
    XAD_INLINE Scalar operator()(const Scalar& a) const { return -a; }
    XAD_INLINE Scalar derivative(const Scalar&) const { return -Scalar(1); }
};

// binary operations with a scalar are actually unary functors

template <class Scalar, class T2>
struct scalar_add_op
{
    XAD_INLINE explicit scalar_add_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return a + b_; }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    Scalar b_;
};

// the factor stays passive: an active one has a zero tangent, which times an infinite a gives NaN
template <class Scalar, class T2>
struct scalar_prod_op
{
    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    XAD_INLINE explicit scalar_prod_op(const T2& b) : b_(nested_type(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return Scalar(a * b_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(b_); }
    nested_type b_;
};

template <class Scalar, class T2>
struct scalar_sub1_op
{
    XAD_INLINE explicit scalar_sub1_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return b_ - a; }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(-1); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_sub2_op
{
    XAD_INLINE explicit scalar_sub2_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return a - Scalar(b_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_div1_op
{
    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    XAD_INLINE explicit scalar_div1_op(const T2& b) : b_(nested_type(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return b_ / a; }
    // -b / a^2 as -(b / a) * (1 / a), as a * a can overflow or underflow
    XAD_INLINE Scalar derivative(const Scalar& a, const Scalar& v) const
    {
        return -v * (Scalar(1) / a);
    }
    template <class D>
    XAD_INLINE D tangent(const Scalar& a, const Scalar& v, const D& t) const
    {
        return detail::quotientTangent(a, v, D(), t);
    }
    nested_type b_;
};

template <class Scalar, class T2>
struct OperatorTraits<scalar_div1_op<Scalar, T2> >
{
    enum
    {
        useResultBasedDerivatives = 1
    };
};

template <class Scalar, class T2>
struct scalar_div2_op
{
    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    XAD_INLINE explicit scalar_div2_op(const T2& b) : b_(nested_type(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return a / b_; }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(nested_type(1) / b_); }
    nested_type b_;
};

namespace detail
{
template <class Scalar>
struct HasFinitePartials<negate_op<Scalar> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasFinitePartials<scalar_add_op<Scalar, T2> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasFinitePartials<scalar_prod_op<Scalar, T2> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasFinitePartials<scalar_sub1_op<Scalar, T2> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasFinitePartials<scalar_sub2_op<Scalar, T2> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasFinitePartials<scalar_div2_op<Scalar, T2> > : std::true_type
{
};
template <class Scalar, class T2>
struct HasTangentRule<scalar_div1_op<Scalar, T2> > : std::true_type
{
};
}  // namespace detail
}  // namespace xad

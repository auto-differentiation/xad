/*******************************************************************************

   Functors for binary arithmetic operators.

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
struct add_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(1); }
};

template <class Scalar>
struct prod_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a * b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar&) const { return a; }
};

template <class Scalar>
struct sub_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a - b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-1); }
};

namespace detail
{
template <class Scalar>
struct HasFinitePartials<add_op<Scalar> > : std::true_type
{
};
template <class Scalar>
struct HasFinitePartials<prod_op<Scalar> > : std::true_type
{
};
template <class Scalar>
struct HasFinitePartials<sub_op<Scalar> > : std::true_type
{
};
}  // namespace detail

template <class Scalar>
struct div_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a / b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return Scalar(1) / b; }

    // -a / b^2 as -(a / b) / b, without waiting for a / b; b * b can overflow or underflow
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        const Scalar r = Scalar(1) / b;
        return -(a * r) * r;
    }

    template <class D>
    XAD_INLINE D tangent(const Scalar&, const Scalar& b, const Scalar& v, const D& ta,
                         const D& tb) const
    {
        return detail::quotientTangent(b, v, ta, tb);
    }
};

namespace detail
{
template <class Scalar>
struct HasTangentRule<div_op<Scalar> > : std::true_type
{
};
}  // namespace detail
}  // namespace xad

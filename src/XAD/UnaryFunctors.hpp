/*******************************************************************************

   Functors capturing unary expressions.

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

template <class Scalar, class T2>
struct scalar_prod_op
{
    XAD_INLINE explicit scalar_prod_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return Scalar(a * b_); }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(b_); }
    Scalar b_;
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
    XAD_INLINE explicit scalar_div1_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return b_ / a; }
    XAD_INLINE Scalar derivative(const Scalar& a) const { return -b_ / (a * a); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_div2_op
{
    XAD_INLINE explicit scalar_div2_op(const T2& b) : b_(Scalar(b)) {}
    XAD_INLINE Scalar operator()(const Scalar& a) const { return a / b_; }
    XAD_INLINE Scalar derivative(const Scalar&) const { return Scalar(1) / b_; }
    Scalar b_;
};
}  // namespace xad

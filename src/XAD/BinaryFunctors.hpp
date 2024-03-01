/*******************************************************************************

   Functors for binary arithmetic operators.

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

template <class Scalar>
struct div_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a / b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return Scalar(1) / b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -a / (b * b); }
};
}  // namespace xad

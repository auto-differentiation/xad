/*******************************************************************************

   Implementation template for binary derivatives, specialising if 2nd parameter
   is not needed.

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
#include <XAD/Macros.hpp>

namespace xad
{
namespace detail
{

template <bool>
struct BinaryDerivativeImpl
{
    template <class Op, class Scalar>
    static XAD_INLINE Scalar derivative_a(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar&)
    {
        return op.derivative_a(a, b);
    }

    template <class Op, class Scalar>
    static XAD_INLINE Scalar derivative_b(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar&)
    {
        return op.derivative_b(a, b);
    }
};

template <>
struct BinaryDerivativeImpl<true>
{
    template <class Op, class Scalar>
    static XAD_INLINE Scalar derivative_a(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar& c)
    {
        return op.derivative_a(a, b, c);
    }

    template <class Op, class Scalar>
    static XAD_INLINE Scalar derivative_b(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar& c)
    {
        return op.derivative_b(a, b, c);
    }
};
}  // namespace detail
}  // namespace xad

/*******************************************************************************

   Declare expression types.

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

// keeps information about multipliers and slots for an expression
// locally on stack
template <typename TapeType, int N>
struct DerivInfo
{
    unsigned index = 0;
    typename TapeType::value_type multipliers[N];
    typename TapeType::slot_type slots[N];
};

/// Represents a generic expression, for the Scalar base type.
///
/// It uses the CTRP pattern, where derived classes register themselves with
/// the base class in the second template parameter
template <class Scalar, class Derived, class DerivativeType = Scalar>
struct Expression
{
    /// get a reference to the derived object
    XAD_INLINE const Derived& derived() const { return static_cast<const Derived&>(*this); }

    /// get the value
    XAD_INLINE Scalar value() const { return derived().value(); }

    XAD_INLINE Scalar getValue() const { return value(); }

#ifdef XAD_ALLOW_INT_CONVERSION
    XAD_INLINE explicit operator char() const { return static_cast<char>(getValue()); }
    XAD_INLINE explicit operator unsigned char() const
    {
        return static_cast<unsigned char>(getValue());
    }
    XAD_INLINE explicit operator signed char() const
    {
        return static_cast<signed char>(getValue());
    }
    XAD_INLINE explicit operator short() const { return static_cast<short>(getValue()); }
    XAD_INLINE explicit operator unsigned short() const
    {
        return static_cast<unsigned short>(getValue());
    }
    XAD_INLINE explicit operator int() const { return static_cast<int>(getValue()); }
    XAD_INLINE explicit operator unsigned int() const
    {
        return static_cast<unsigned int>(getValue());
    }
    XAD_INLINE explicit operator long() const { return static_cast<long>(getValue()); }
    XAD_INLINE explicit operator unsigned long() const
    {
        return static_cast<unsigned long>(getValue());
    }
    XAD_INLINE explicit operator long long() const { return static_cast<long long>(getValue()); }
    XAD_INLINE explicit operator unsigned long long() const
    {
        return static_cast<unsigned long long>(getValue());
    }
#endif

    // convert to boolean
    XAD_INLINE explicit operator bool() const { return value() != Scalar(0); }

    /// calculate the derivatives, given a tape object
    template <class Tape, int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s) const
    {
        derived().calc_derivatives(info, s, Scalar(1));
    }

    /// calculate the derivatives, given tape and multiplier
    template <class Tape, int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s,
                                     const Scalar& multiplier) const
    {
        derived().calc_derivatives(info, s, multiplier);
    }

    XAD_INLINE bool shouldRecord() const { return derived().shouldRecord(); }

    XAD_INLINE DerivativeType derivative() const { return derived().derivative(); }
};

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE Scalar value(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.value();
}

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE DerivativeType derivative(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.derivative();
}
}  // namespace xad

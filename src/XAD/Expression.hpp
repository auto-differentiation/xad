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
#include <cstddef>

namespace xad
{
    template<typename T, size_t Size>
    struct DerivInfo {
        size_t index = 0;
        T multipliers[Size];
        unsigned int slots[Size]; 
    };
/// Represents a generic expression, for the Scalar base type.
///
/// It uses the CTRP pattern, where derived classes register themselves with
/// the base class in the second template parameter
template <class Scalar, class Derived>
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
    template <class Tape, size_t Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Scalar, Size> &info, Tape& s) const
    {
        derived().calc_derivatives(info, s, Scalar(1));
    }

    /// calculate the derivatives, given tape and multiplier
    template <class Tape, size_t Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Scalar, Size> &info, Tape& s, const Scalar& multiplier) const
    {
        derived().calc_derivatives(info, s, multiplier);
    }

    template <typename Slot, size_t Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Scalar, Size> &info, Slot* slot, Scalar* muls, int& n, const Scalar& mul) const
    {
        derived().calc_derivatives(info, slot, muls, n, mul);
    }

    template <typename It1, typename It2, size_t Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Scalar, Size> &info, It1& sit, It2& mit, const Scalar& mul) const
    {
        derived().calc_derivatives(info, sit, mit, mul);
    }

    XAD_INLINE bool shouldRecord() const { return derived().shouldRecord(); }

    XAD_INLINE Scalar derivative() const { return derived().derivative(); }
};

template <class Scalar, class Expr>
XAD_INLINE Scalar value(const Expression<Scalar, Expr>& expr)
{
    return expr.value();
}

template <class Scalar, class Expr>
XAD_INLINE Scalar derivative(const Expression<Scalar, Expr>& expr)
{
    return expr.derivative();
}
}  // namespace xad

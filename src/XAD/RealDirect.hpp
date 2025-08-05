/*******************************************************************************

   Base class for direct mode.

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
#include <XAD/MathFunctions.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

namespace xad
{

template <class BaseType, class Derived>
struct RealDirect
{
    typedef typename BaseType::value_type Scalar;
    typedef Derived derived_type;

    constexpr XAD_INLINE RealDirect() = default;
    constexpr XAD_INLINE RealDirect(RealDirect &&o) noexcept = default;
    constexpr XAD_INLINE RealDirect(const RealDirect &o) = default;
    RealDirect &operator=(const RealDirect &val) = default;
    RealDirect &operator=(RealDirect &&val) = default;

    constexpr XAD_INLINE RealDirect(const BaseType &rhs) : base_(rhs) {}

    XAD_INLINE ~RealDirect() = default;

    template <class T>
    RealDirect(const T &val) : base_(val)
    {
    }

    XAD_INLINE BaseType &base() { return base_; }
    constexpr XAD_INLINE const BaseType &base() const { return base_; }

    XAD_INLINE RealDirect &operator+=(const RealDirect &x)
    {
        base_ += x.base();
        return *this;
    }
    XAD_INLINE RealDirect &operator-=(const RealDirect &x)
    {
        base_ -= x.base();
        return *this;
    }

    XAD_INLINE RealDirect &operator*=(const RealDirect &x)
    {
        base_ *= x.base();
        return *this;
    }

    XAD_INLINE RealDirect &operator/=(const RealDirect &x)
    {
        base_ /= x.base();
        return *this;
    }

    XAD_INLINE const Scalar &value() const { return base_.value(); }

    XAD_INLINE const typename BaseType::derivative_type &derivative() const
    {
        return base_.derivative();
    }

    XAD_INLINE Scalar &value() { return base_.value(); }

    XAD_INLINE typename BaseType::derivative_type &derivative() { return base_.derivative(); }

    XAD_INLINE void setDerivative(const typename BaseType::derivative_type &a) { derivative() = a; }

    XAD_INLINE typename BaseType::derivative_type getDerivative() const { return derivative(); }

    constexpr XAD_INLINE Scalar getValue() const { return value(); }

  private:
    BaseType base_;
};

template <class Base, class Derived>
XAD_INLINE Derived operator-(const RealDirect<Base, Derived> &a)
{

    return {Base(-a.base())};
}

}  // namespace xad

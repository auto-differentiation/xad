/*******************************************************************************

   Base class for direct mode.

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
#include <XAD/MathFunctions.hpp>
#include <XAD/Traits.hpp>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace xad
{

template <class BaseType, class Derived>
struct RealDirect
{
    typedef typename BaseType::value_type Scalar;
    typedef Derived derived_type;
    typedef Scalar value_type;
    typedef typename BaseType::nested_type nested_type;
    typedef typename BaseType::derivative_type derivative_type;

    constexpr XAD_INLINE RealDirect() = default;
    constexpr XAD_INLINE RealDirect(RealDirect &&o) noexcept = default;
    constexpr XAD_INLINE RealDirect(const RealDirect &o) = default;
    XAD_INLINE RealDirect &operator=(const RealDirect &val) = default;
    XAD_INLINE RealDirect &operator=(RealDirect &&val) = default;

    constexpr XAD_INLINE RealDirect(const BaseType &rhs) : base_(rhs) {}
    // operations return a temporary: take over its slot rather than record a copy
    constexpr XAD_INLINE RealDirect(BaseType &&rhs) noexcept : base_(std::move(rhs)) {}

    XAD_INLINE ~RealDirect() = default;

    // constrained so that std::is_constructible / std::is_convertible give the right answers
    template <class T, class = typename std::enable_if<
                           std::is_constructible<BaseType, const T &>::value>::type>
    RealDirect(const T &val) : base_(val)
    {
    }

    XAD_INLINE BaseType &base() { return base_; }
    constexpr XAD_INLINE const BaseType &base() const { return base_; }

    XAD_INLINE Derived &derived() { return static_cast<Derived &>(*this); }
    XAD_INLINE const Derived &derived() const { return static_cast<const Derived &>(*this); }

    XAD_INLINE Derived &operator+=(const RealDirect &x)
    {
        base_ += x.base();
        return derived();
    }
    XAD_INLINE Derived &operator-=(const RealDirect &x)
    {
        base_ -= x.base();
        return derived();
    }

    XAD_INLINE Derived &operator*=(const RealDirect &x)
    {
        base_ *= x.base();
        return derived();
    }

    XAD_INLINE Derived &operator/=(const RealDirect &x)
    {
        base_ /= x.base();
        return derived();
    }

    XAD_INLINE Derived &operator++()
    {
        base_ = base_ + nested_type(1);
        return derived();
    }

    XAD_INLINE Derived operator++(int)
    {
        Derived ret = derived();
        base_ = base_ + nested_type(1);
        return ret;
    }

    XAD_INLINE Derived &operator--()
    {
        base_ = base_ - nested_type(1);
        return derived();
    }

    XAD_INLINE Derived operator--(int)
    {
        Derived ret = derived();
        base_ = base_ - nested_type(1);
        return ret;
    }

#ifdef XAD_ALLOW_INT_CONVERSION
    // same explicit conversions as AReal / FReal (see Expression.hpp)
    XAD_INLINE explicit operator char() const { return static_cast<char>(base_); }
    XAD_INLINE explicit operator unsigned char() const { return static_cast<unsigned char>(base_); }
    XAD_INLINE explicit operator signed char() const { return static_cast<signed char>(base_); }
    XAD_INLINE explicit operator short() const { return static_cast<short>(base_); }
    XAD_INLINE explicit operator unsigned short() const
    {
        return static_cast<unsigned short>(base_);
    }
    XAD_INLINE explicit operator int() const { return static_cast<int>(base_); }
    XAD_INLINE explicit operator unsigned int() const { return static_cast<unsigned int>(base_); }
    XAD_INLINE explicit operator long() const { return static_cast<long>(base_); }
    XAD_INLINE explicit operator unsigned long() const { return static_cast<unsigned long>(base_); }
    XAD_INLINE explicit operator long long() const { return static_cast<long long>(base_); }
    XAD_INLINE explicit operator unsigned long long() const
    {
        return static_cast<unsigned long long>(base_);
    }
#endif

    XAD_INLINE explicit operator bool() const { return static_cast<bool>(base_); }

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

template <class Base, class Derived>
XAD_INLINE Derived operator+(const RealDirect<Base, Derived> &a)
{
    return a.derived();
}

template <class C, class T, class Base, class Derived>
XAD_INLINE std::basic_ostream<C, T> &operator<<(std::basic_ostream<C, T> &os,
                                                const RealDirect<Base, Derived> &x)
{
    return os << x.base();
}

template <class C, class T, class Base, class Derived>
XAD_INLINE std::basic_istream<C, T> &operator>>(std::basic_istream<C, T> &is,
                                                RealDirect<Base, Derived> &x)
{
    return is >> x.base();
}

}  // namespace xad

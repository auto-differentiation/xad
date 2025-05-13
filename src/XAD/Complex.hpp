/*******************************************************************************

   An AD-enabled equivalent of std::complex.

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
#include <XAD/BinaryOperators.hpp>
#include <XAD/Expression.hpp>
#include <XAD/Literals.hpp>
#include <XAD/Traits.hpp>
#include <XAD/UnaryOperators.hpp>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>

// Note: The seemingly high number of overloads in this file, containing partially
// duplicated code, is necessary to make sure all supported compilers pick the overloads
// from here instead of the the std::complex versions provided by the standard library.

namespace xad
{

namespace detail
{
template <class T>
class complex_impl
{
  public:
    typedef T value_type;

    explicit complex_impl(const T& areal = T(), const T& aimag = T()) : real_(areal), imag_(aimag)
    {
    }

    template <class X>
    explicit complex_impl(const std::complex<X>& o) : real_(o.real()), imag_(o.imag())
    {
    }

    template <class X>
    XAD_INLINE complex_impl& operator=(const X& other)
    {
        real_ = other;
        imag_ = T();
        return *this;
    }

    template <class X>
    XAD_INLINE complex_impl& operator=(const std::complex<X>& other)
    {
        real_ = other.real();
        imag_ = other.imag();
        return *this;
    }

    XAD_INLINE T& real() { return real_; }
    XAD_INLINE T& imag() { return imag_; }
    XAD_INLINE const T& real() const { return real_; }
    XAD_INLINE const T& imag() const { return imag_; }

    XAD_INLINE void real(const T& value) { real_ = value; }
    XAD_INLINE void imag(const T& value) { imag_ = value; }

    void setDerivative(
        typename ExprTraits<T>::nested_type rd,
        typename ExprTraits<T>::nested_type id = typename ExprTraits<T>::nested_type())
    {
        derivative(real_) = rd;
        derivative(imag_) = id;
    }

    XAD_INLINE void setAdjoint(
        typename ExprTraits<T>::nested_type rd,
        typename ExprTraits<T>::nested_type id = typename ExprTraits<T>::nested_type())
    {
        this->setDerivative(rd, id);
    }

    XAD_INLINE std::complex<typename ExprTraits<T>::nested_type> getDerivative() const
    {
        return std::complex<typename ExprTraits<T>::nested_type>(derivative(real_),
                                                                 derivative(imag_));
    }

    XAD_INLINE std::complex<typename ExprTraits<T>::nested_type> getAdjoint() const
    {
        return this->getDerivative();
    }

  private:
    T real_, imag_;
};

}  // namespace detail

}  // namespace xad

namespace std
{

template <class Scalar, class T>
class complex<xad::ADTypeBase<Scalar, T>> : public xad::detail::complex_impl<T>
{
  public:
    typedef xad::detail::complex_impl<T> base;
    typedef T value_type;

    complex(const T& areal = T(), const T& aimag = T()) : base(areal, aimag) {}

    XAD_INLINE complex<T>& derived() { return static_cast<complex<T>&>(*this); }

    XAD_INLINE complex<T>& operator+=(const T& other)
    {
        base::real() += other;
        return derived();
    }

    template <class X>
    XAD_INLINE complex<T>& operator+=(const std::complex<X>& other)
    {
        base::real() += other.real();
        base::imag() += other.imag();
        return derived();
    }

    XAD_INLINE complex<T>& operator-=(const T& other)
    {
        base::real() -= other;
        return derived();
    }

    template <class X>
    XAD_INLINE complex<T>& operator-=(const std::complex<X>& other)
    {
        base::real() -= other.real();
        base::imag() -= other.imag();
        return derived();
    }

    XAD_INLINE complex<T>& operator*=(const T& other)
    {
        base::real() *= other;
        base::imag() *= other;
        return derived();
    }

    template <class X>
    complex<T>& operator*=(const std::complex<X>& other)
    {
        T real_t = base::real() * other.real() - base::imag() * other.imag();
        base::imag(base::real() * other.imag() + other.real() * base::imag());
        base::real(real_t);
        return derived();
    }

    XAD_INLINE complex<T>& operator/=(const T& other)
    {
        base::real() /= other;
        base::imag() /= other;
        return derived();
    }

    template <class X>
    complex<T>& operator/=(const std::complex<X>& other)
    {
        T den = T(other.real()) * T(other.real()) + T(other.imag()) * T(other.imag());
        T real_t = ((base::real() * T(other.real())) + (base::imag() * T(other.imag()))) / den;
        base::imag((base::imag() * T(other.real()) - base::real() * T(other.imag())) / den);
        base::real(real_t);
        return derived();
    }
};

template <class T>
class complex<xad::AReal<T>> : public complex<xad::ADTypeBase<T, xad::AReal<T>>>
{
  public:
    typedef complex<xad::ADTypeBase<T, xad::AReal<T>>> base;

    // inheriting template constructors doesn't work in all compilers

    XAD_INLINE complex(const xad::AReal<T>& areal = xad::AReal<T>(),
                       const xad::AReal<T>& aimag = xad::AReal<T>())
        : base(areal, aimag)
    {
    }

    template <class X>
    XAD_INLINE complex(const X& areal,
                       typename std::enable_if<!xad::ExprTraits<X>::isExpr>::type* = nullptr)
        : base(xad::AReal<T>(areal), xad::AReal<T>())
    {
    }

    template <class X>
    XAD_INLINE complex(  // cppcheck-suppress noExplicitConstructor
        const X& areal,
        typename std::enable_if<xad::ExprTraits<X>::isExpr &&
                                xad::ExprTraits<X>::direction == xad::DIR_REVERSE>::type* = nullptr)
        : base(xad::AReal<T>(areal), xad::AReal<T>())
    {
    }

    template <class X>
    XAD_INLINE complex(const complex<X>& o) : base(xad::AReal<T>(o.real()), xad::AReal<T>(o.imag()))
    {
    }

    using base::operator+=;
    using base::operator-=;
    using base::operator*=;
    using base::operator/=;
};

template <class T>
class complex<xad::FReal<T>> : public complex<xad::ADTypeBase<T, xad::FReal<T>>>
{
  public:
    typedef complex<xad::ADTypeBase<T, xad::FReal<T>>> base;

    // inheriting template constructors doesn't work in all compilers

    XAD_INLINE complex(
        const xad::FReal<T>& areal = xad::FReal<T>(),  // cppcheck-suppress noExplicitConstructor
        const xad::FReal<T>& aimag = xad::FReal<T>())
        : base(areal, aimag)
    {
    }

    template <class X>
    XAD_INLINE complex(const X& areal,  // cppcheck-suppress noExplicitConstructor
                       typename std::enable_if<!xad::ExprTraits<X>::isExpr>::type* = nullptr)
        : base(xad::FReal<T>(areal), xad::FReal<T>())
    {
    }

    template <class X>
    XAD_INLINE complex(  // cppcheck-suppress noExplicitConstructor
        const X& areal,
        typename std::enable_if<xad::ExprTraits<X>::isExpr &&
                                xad::ExprTraits<X>::direction == xad::DIR_FORWARD>::type* = nullptr)
        : base(xad::FReal<T>(areal), xad::FReal<T>())
    {
    }

    template <class X>
    XAD_INLINE complex(const complex<X>& o)
        : base(xad::FReal<T>(o.real()),
               xad::FReal<T>(o.imag()))  // cppcheck-suppress noExplicitConstructor
    {
    }

    using base::operator+=;
    using base::operator-=;
    using base::operator*=;
    using base::operator/=;
};

}  // namespace std

namespace xad
{

// read access to value and derivatives
template <class T>
XAD_INLINE std::complex<T> derivative(const std::complex<AReal<T>>& z)
{
    return std::complex<T>(derivative(z.real()), derivative(z.imag()));
}

template <class T>
XAD_INLINE std::complex<T> derivative(const std::complex<FReal<T>>& z)
{
    return std::complex<T>(derivative(z.real()), derivative(z.imag()));
}

template <class T>
XAD_INLINE std::complex<T> value(const std::complex<AReal<T>>& z)
{
    return std::complex<T>(value(z.real()), value(z.imag()));
}

template <class T>
XAD_INLINE std::complex<T> value(const std::complex<FReal<T>>& z)
{
    return std::complex<T>(value(z.real()), value(z.imag()));
}

template <class T>
XAD_INLINE std::complex<T> value(const std::complex<T>& z)
{
    return z;
}

namespace detail
{

// declare first, implementation at the bottom of this file,
// to avoid issues with order of declaration (functions called in bodies
// not defined yet)

template <class T>
XAD_INLINE T abs_impl(const std::complex<T>& x);

template <class T>
XAD_INLINE std::complex<T> exp_impl(const std::complex<T>& z);

template <class T1, class T2>
XAD_INLINE std::complex<typename xad::ExprTraits<T1>::value_type> polar_impl(const T1& r,
                                                                             const T2& theta);

template <class T>
XAD_INLINE std::complex<T> sqrt_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> sinh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> cosh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> tanh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> asinh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> acosh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> atanh_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> sin_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> cos_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> tan_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> asin_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> acos_impl(const std::complex<T>& z);

template <class T>
XAD_INLINE std::complex<T> atan_impl(const std::complex<T>& z);

// note that this captures the AReal and FReal base types too
template <class Scalar, class Derived>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type arg_impl(
    const xad::Expression<Scalar, Derived>& x);

template <class T>
XAD_INLINE T arg_impl(const std::complex<T>& z);

#if (defined(_MSC_VER) && (_MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) &&             \
    !defined(__clang__)
template <class Scalar, class Derived>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type proj_impl(
    const xad::Expression<Scalar, Derived>& x);
#else
template <class Scalar, class Derived>
XAD_INLINE std::complex<typename xad::ExprTraits<Derived>::value_type> proj_impl(
    const xad::Expression<Scalar, Derived>& x);
#endif

template <class T>
XAD_INLINE T norm_impl(const std::complex<T>& x);

}  // namespace detail

}  // namespace xad

namespace std
{

// access to real / imag
template <class Scalar, class T>
XAD_INLINE T real(const std::complex<xad::ADTypeBase<Scalar, T>>& z)
{
    return z.derived().real();
}

template <class Scalar, class T>
XAD_INLINE T& real(std::complex<xad::ADTypeBase<Scalar, T>>& z)
{
    return z.derived().real();
}

template <class Scalar, class Expr>
XAD_INLINE typename xad::ExprTraits<Expr>::value_type real(
    const xad::Expression<Scalar, Expr>& other)
{
    return other.derived();
}

template <class Scalar, class T>
XAD_INLINE T imag(const std::complex<xad::ADTypeBase<Scalar, T>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class T>
XAD_INLINE T& imag(std::complex<xad::ADTypeBase<Scalar, T>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class Expr>
XAD_INLINE typename xad::ExprTraits<Expr>::value_type imag(const xad::Expression<Scalar, Expr>&)
{
    return typename xad::ExprTraits<Expr>::value_type(0);
}

///////////////////////// operators
template <class T>
XAD_INLINE const std::complex<xad::AReal<T>>& operator+(const std::complex<xad::AReal<T>>& x)
{
    return x;
}

template <class T>
XAD_INLINE const std::complex<xad::FReal<T>>& operator+(const std::complex<xad::FReal<T>>& x)
{
    return x;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(const std::complex<xad::AReal<T>>& x)
{
    return std::complex<xad::AReal<T>>(-x.real(), -x.imag());
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const std::complex<xad::FReal<T>>& x)
{
    return std::complex<xad::FReal<T>>(-x.real(), -x.imag());
}

// operator== - lots of variants here, I'm sure this could be done cleaner...

template <class T>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T>>& lhs,
                           const std::complex<xad::AReal<T>>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <class T>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T>>& lhs,
                           const std::complex<xad::FReal<T>>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <class T, class Expr>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T>>& lhs,
                           const xad::Expression<T, Expr>& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, class Expr>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T>>& lhs,
                           const xad::Expression<T, Expr>& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T>>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T>>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, class Expr>
XAD_INLINE bool operator==(const xad::Expression<T, Expr>& rhs,
                           const std::complex<xad::AReal<T>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, class Expr>
XAD_INLINE bool operator==(const xad::Expression<T, Expr>& rhs,
                           const std::complex<xad::FReal<T>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T>
XAD_INLINE bool operator==(const T& rhs, const std::complex<xad::AReal<T>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T>
XAD_INLINE bool operator==(const T& rhs, const std::complex<xad::FReal<T>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// operator !=

template <class T>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T>>& lhs,
                           const std::complex<xad::AReal<T>>& rhs)
{
    return !(lhs == rhs);
}

template <class T>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T>>& lhs,
                           const std::complex<xad::FReal<T>>& rhs)
{
    return !(lhs == rhs);
}

template <class T, class Expr>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T>>& lhs,
                           const xad::Expression<T, Expr>& rhs)
{
    return !(lhs == rhs);
}

template <class T, class Expr>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T>>& lhs,
                           const xad::Expression<T, Expr>& rhs)
{
    return !(lhs == rhs);
}

template <class T>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T>>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

template <class T>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T>>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

template <class T, class Expr>
XAD_INLINE bool operator!=(const xad::Expression<T, Expr>& rhs,
                           const std::complex<xad::AReal<T>>& lhs)
{
    return !(lhs == rhs);
}

template <class T, class Expr>
XAD_INLINE bool operator!=(const xad::Expression<T, Expr>& rhs,
                           const std::complex<xad::FReal<T>>& lhs)
{
    return !(lhs == rhs);
}

template <class T>
XAD_INLINE bool operator!=(const T& rhs, const std::complex<xad::AReal<T>>& lhs)
{
    return !(lhs == rhs);
}

template <class T>
XAD_INLINE bool operator!=(const T& rhs, const std::complex<xad::FReal<T>>& lhs)
{
    return !(lhs == rhs);
}

// operator+

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<xad::AReal<T>>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<xad::AReal<T>> lhs,
                                                 const xad::AReal<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<T> lhs, const xad::AReal<T>& rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z += rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<xad::AReal<T>> lhs, const T& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator+(std::complex<xad::AReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator+(
    const std::complex<T>& lhs, const xad::Expression<T, Expr>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(const std::complex<T>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(const xad::AReal<T>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator+(const T& rhs, std::complex<xad::AReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator+(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator+(
    const xad::Expression<T, Expr>& rhs, const std::complex<T>& lhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const std::complex<T>& lhs,
                                                 std::complex<xad::FReal<T>> rhs)
{
    rhs += lhs;
    return rhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(std::complex<xad::FReal<T>> lhs,
                                                 const xad::FReal<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const std::complex<T>& lhs,
                                                 const xad::FReal<T>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z += rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(std::complex<xad::FReal<T>> lhs, const T& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator+(std::complex<xad::FReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const xad::FReal<T>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const xad::FReal<T>& rhs,
                                                 const std::complex<T>& lhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z += rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const T& rhs, std::complex<xad::FReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator+(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    lhs += rhs;
    return lhs;
}

// operator-

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<xad::AReal<T>>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(const std::complex<T>& lhs,
                                                 std::complex<xad::AReal<T>> rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z -= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<xad::AReal<T>> lhs,
                                                 const xad::AReal<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<T> lhs, const xad::AReal<T>& rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z -= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<xad::AReal<T>> lhs, const T& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator-(std::complex<xad::AReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator-(
    const std::complex<T>& lhs, const xad::Expression<T, Expr>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator-(
    const xad::Expression<T, Expr>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(const xad::AReal<T>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator-(const T& rhs, std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator-(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const std::complex<T>& lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z -= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T>> z = rhs;
    lhs -= z;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<xad::FReal<T>> lhs,
                                                 const xad::FReal<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<T> lhs, const xad::FReal<T>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z -= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<xad::FReal<T>> lhs, const T& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator-(std::complex<xad::FReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const xad::FReal<T>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const xad::FReal<T>& rhs, std::complex<T> lhs)
{
    return std::complex<xad::FReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const T& rhs, std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator-(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs - lhs.real(), -lhs.imag());
}

// operator*
template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<xad::AReal<T>>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(const std::complex<T>& lhs,
                                                 const std::complex<xad::AReal<T>>& rhs)
{
    return rhs * lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(std::complex<xad::AReal<T>> lhs,
                                                 const xad::AReal<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(const std::complex<T>& lhs,
                                                 const xad::AReal<T>& rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z *= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(std::complex<xad::AReal<T>> lhs, const T& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator*(std::complex<xad::AReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator*(
    const std::complex<T>& lhs, const xad::Expression<T, Expr>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator*(
    const xad::Expression<T, Expr>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(const xad::AReal<T>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator*(const T& rhs, std::complex<xad::AReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator*(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(const std::complex<T>& lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z *= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T>> z = rhs;
    lhs *= z;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<xad::FReal<T>> lhs,
                                                 const xad::FReal<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<T> lhs, const xad::FReal<T>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z *= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<xad::FReal<T>> lhs, const T& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator*(std::complex<xad::FReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(const xad::FReal<T>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(const xad::FReal<T>& rhs, std::complex<T> lhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z *= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator*(const T& rhs, std::complex<xad::FReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator*(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    lhs *= rhs;
    return lhs;
}

// operator/

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<xad::AReal<T>>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<xad::AReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(const std::complex<T>& lhs,
                                                 std::complex<xad::AReal<T>> rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z /= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<xad::AReal<T>> lhs,
                                                 const xad::AReal<T>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<T> lhs, const xad::AReal<T>& rhs)
{
    std::complex<xad::AReal<T>> z = lhs;
    z /= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<xad::AReal<T>> lhs, const T& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator/(std::complex<xad::AReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator/(
    const std::complex<T>& lhs, const xad::Expression<T, Expr>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator/(
    const xad::Expression<T, Expr>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(const xad::AReal<T>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs) / lhs;
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> operator/(const T& rhs, std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs) / lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::AReal<T>> operator/(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::AReal<T>> lhs)
{
    return std::complex<xad::AReal<T>>(rhs) / lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(const std::complex<T>& lhs,
                                                 const std::complex<xad::FReal<T>>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z /= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<xad::FReal<T>> lhs,
                                                 const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T>> z = rhs;
    lhs /= z;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<xad::FReal<T>> lhs,
                                                 const xad::FReal<T>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<T> lhs, const xad::FReal<T>& rhs)
{
    std::complex<xad::FReal<T>> z = lhs;
    z /= rhs;
    return z;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<xad::FReal<T>> lhs, const T& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator/(std::complex<xad::FReal<T>> lhs,
                                                 const xad::Expression<T, Expr>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(const xad::FReal<T>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs) / lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(const xad::FReal<T>& rhs, std::complex<T> lhs)
{
    return std::complex<xad::FReal<T>>(rhs) / lhs;
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> operator/(const T& rhs, std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs) / lhs;
}

template <class T, class Expr>
XAD_INLINE std::complex<xad::FReal<T>> operator/(const xad::Expression<T, Expr>& rhs,
                                                 std::complex<xad::FReal<T>> lhs)
{
    return std::complex<xad::FReal<T>>(rhs) / lhs;
}
/////////////////////// math functions

template <class T>
XAD_INLINE xad::AReal<T> arg(const complex<xad::AReal<T>>& x)
{
    return xad::detail::arg_impl(x);
}

template <class T>
XAD_INLINE xad::FReal<T> arg(const complex<xad::FReal<T>>& x)
{
    return xad::detail::arg_impl(x);
}

template <class Scalar, class Derived>
typename xad::ExprTraits<Derived>::value_type arg(const xad::Expression<Scalar, Derived>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T>
typename xad::AReal<T> arg(const xad::AReal<T>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T>
typename xad::FReal<T> arg(const xad::FReal<T>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T, class Scalar>
XAD_INLINE T norm(const complex<xad::ADTypeBase<Scalar, T>>& x)
{
    return ::xad::detail::norm_impl(x);
}

template <class T>
XAD_INLINE xad::AReal<T> norm(const complex<xad::AReal<T>>& x)
{
    return ::xad::detail::norm_impl(x);
}

template <class T>
XAD_INLINE xad::FReal<T> norm(const complex<xad::FReal<T>>& x)
{
    return ::xad::detail::norm_impl(x);
}

// appleclang15 needs this overload for type paramed norm
#if defined(__APPLE__) && defined(__clang__) && defined(__apple_build_version__) &&                \
    (__apple_build_version__ >= 15000000)
template <class T>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T>::isExpr, T>::type norm(complex<T>& x)
{
    return ::xad::detail::norm_impl(x);
}
#endif

// return the expression type from multiplying x*x without actually evaluating it
template <class Scalar, class Derived>
XAD_INLINE auto norm(const xad::Expression<Scalar, Derived>& x) -> decltype(x * x)
{
    return x * x;
}

template <class T>
XAD_INLINE xad::AReal<T> abs(const complex<xad::AReal<T>>& x)
{
    return xad::detail::abs_impl(x);
}

template <class T>
XAD_INLINE xad::FReal<T> abs(const complex<xad::FReal<T>>& x)
{
    return xad::detail::abs_impl(x);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> conj(const complex<xad::AReal<T>>& z)
{
    complex<xad::AReal<T>> ret(z.real(), -z.imag());
    return ret;
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> conj(const complex<xad::FReal<T>>& z)
{
    complex<xad::FReal<T>> ret(z.real(), -z.imag());
    return ret;
}

#if ((defined(_MSC_VER) && (_MSC_VER < 1920)) || (defined(__GNUC__) && __GNUC__ < 7)) &&           \
    !defined(__clang__)
template <class Scalar, class Derived>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type conj(
    const xad::Expression<Scalar, Derived>& x)
{
    return typename xad::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived>
XAD_INLINE complex<typename xad::ExprTraits<Derived>::value_type> conj(
    const xad::Expression<Scalar, Derived>& x)
{
    return complex<typename xad::ExprTraits<Derived>::value_type>(x);
}
#endif

template <class T>
XAD_INLINE complex<xad::AReal<T>> proj(const std::complex<xad::AReal<T>>& z)
{
    if (xad::isinf(z.real()) || xad::isinf(z.imag()))
    {
        typedef typename xad::ExprTraits<T>::nested_type type;
        const type infty = std::numeric_limits<type>::infinity();
        if (xad::signbit(z.imag()))
            return complex<xad::AReal<T>>(infty, -0.0);
        else
            return complex<xad::AReal<T>>(infty, 0.0);
    }
    else
        return z;
}

template <class T>
complex<xad::FReal<T>> proj(const std::complex<xad::FReal<T>>& z)
{
    if (xad::isinf(z.real()) || xad::isinf(z.imag()))
    {
        typedef typename xad::ExprTraits<T>::nested_type type;
        const type infty = std::numeric_limits<type>::infinity();
        if (xad::signbit(z.imag()))
            return complex<xad::FReal<T>>(infty, -0.0);
        else
            return complex<xad::FReal<T>>(infty, 0.0);
    }
    else
        return z;
}

template <class Scalar, class Derived>
XAD_INLINE auto proj(const xad::Expression<Scalar, Derived>& x)
    -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

template <class T>
XAD_INLINE auto proj(const xad::AReal<T>& x) -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

template <class T>
XAD_INLINE auto proj(const xad::FReal<T>& x) -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

// T and expr
// expr and T
// different expr (derived1, derived2 - returns scalar)

template <class T>
XAD_INLINE complex<xad::AReal<T>> polar(const xad::AReal<T>& r,
                                        const xad::AReal<T>& theta = xad::AReal<T>())
{
    return xad::detail::polar_impl(r, theta);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> polar(const xad::FReal<T>& r,
                                        const xad::FReal<T>& theta = xad::FReal<T>())
{
    return xad::detail::polar_impl(r, theta);
}

template <class Scalar, class Expr>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr>& r, const xad::Expression<Scalar, Expr>& theta)
{
    typedef typename xad::ExprTraits<Expr>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

#if defined(_MSC_VER) && _MSC_VER < 1920
// VS 2017 needs loads of specialisations to resolve the right overload and avoid calling the
// std::version

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<xad::AReal<Scalar>> polar(const xad::UnaryExpr<Scalar, Op, Expr>& r,
                                             const xad::AReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar>(r), theta);
}

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<xad::AReal<Scalar>> polar(const xad::AReal<Scalar>& r,
                                             const xad::UnaryExpr<Scalar, Op, Expr>& theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar>(theta));
}

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<xad::FReal<Scalar>> polar(const xad::UnaryExpr<Scalar, Op, Expr>& r,
                                             const xad::FReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar>(r), theta);
}

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<xad::FReal<Scalar>> polar(const xad::FReal<Scalar>& r,
                                             const xad::UnaryExpr<Scalar, Op, Expr>& theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar>(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<xad::AReal<Scalar>> polar(const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& r,
                                             const xad::AReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar>(r), theta);
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<xad::AReal<Scalar>> polar(const xad::AReal<Scalar>& r,
                                             const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar>(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<xad::FReal<Scalar>> polar(const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& r,
                                             const xad::FReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar>(r), theta);
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<xad::FReal<Scalar>> polar(const xad::FReal<Scalar>& r,
                                             const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar>(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op3, Expr3>& r,
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2>& r,
    const xad::UnaryExpr<Scalar, Op3, Expr3>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Op2, class Expr2>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op1, Expr1>& r, const xad::UnaryExpr<Scalar, Op2, Expr2>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Expr4>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op3, Expr3, Expr4>& r,
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    double r, const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr1>::value_type(r),
                                   typename xad::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op, Expr1, Expr2>& r, double theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr1>::value_type(r),
                                   typename xad::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    double r, const xad::UnaryExpr<Scalar, Op, Expr>& theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r),
                                   typename xad::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar, class Op, class Expr>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op, Expr>& r, double theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r),
                                   typename xad::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar>
XAD_INLINE complex<xad::AReal<Scalar>> polar(const xad::AReal<Scalar>& r, double theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar>(theta));
}

template <class Scalar>
XAD_INLINE complex<xad::FReal<Scalar>> polar(const xad::FReal<Scalar>& r, double theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar>(theta));
}

template <class Scalar>
XAD_INLINE complex<xad::AReal<Scalar>> polar(double r, const xad::AReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar>(r), theta);
}

template <class Scalar>
XAD_INLINE complex<xad::FReal<Scalar>> polar(double r, const xad::FReal<Scalar>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar>(r), theta);
}

#endif

// 2 different expression types passed:
// we only enable this function if the underlying value_type of both expressions
// is the same
template <class Scalar, class Expr1, class Expr2>
XAD_INLINE typename std::enable_if<std::is_same<typename xad::ExprTraits<Expr1>::value_type,
                                                typename xad::ExprTraits<Expr2>::value_type>::value,
                                   complex<typename xad::ExprTraits<Expr1>::value_type>>::type
polar(const xad::Expression<Scalar, Expr1>& r, const xad::Expression<Scalar, Expr2>& theta = 0)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

// T, expr - only enabled if T is scalar
template <class Scalar, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> polar(
    Scalar r, const xad::Expression<Scalar, Expr>& theta = 0)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r), theta.derived());
}

// expr, T - only enabled if T is scalar
template <class Scalar, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr>& r, Scalar theta = Scalar())
{
    return xad::detail::polar_impl(r.derived(), typename xad::ExprTraits<Expr>::value_type(theta));
}

// just one expr, as second parameter is optional
template <class Scalar, class Expr>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr>& r)
{
    return complex<typename xad::ExprTraits<Expr>::value_type>(r);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> exp(const complex<xad::AReal<T>>& z)
{
    return xad::detail::exp_impl(z);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> exp(const complex<xad::FReal<T>>& z)
{
    return xad::detail::exp_impl(z);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> log(const complex<xad::AReal<T>>& z)
{
    return complex<xad::AReal<T>>(log(xad::detail::abs_impl(z)), xad::detail::arg_impl(z));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> log(const complex<xad::FReal<T>>& z)
{
    return complex<xad::FReal<T>>(log(xad::detail::abs_impl(z)), xad::detail::arg_impl(z));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> log10(const complex<xad::AReal<T>>& z)
{
    // log(z) * 1/log(10)
    return log(z) * T(0.43429448190325182765112891891661);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> log10(const complex<xad::FReal<T>>& z)
{
    // log(z) * 1/log(10)
    return log(z) * T(0.43429448190325182765112891891661);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x,
                                      const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, const xad::AReal<T>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, const complex<T>& y)
{
    return pow(x, complex<xad::AReal<T>>(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<T>& x, const complex<xad::AReal<T>>& y)
{
    return pow(complex<xad::AReal<T>>(x), y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, const T& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, class T2>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::AReal<T>>>::type pow(
    const complex<xad::AReal<T>>& x, const T2& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, int y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, unsigned y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, unsigned short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, unsigned long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const complex<xad::AReal<T>>& x, unsigned long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const xad::AReal<T>& x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(const T& x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(int x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(short x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(long x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(long long x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(unsigned x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(unsigned short x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(unsigned long x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::AReal<T>> pow(unsigned long long x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, class T2>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::AReal<T>>>::type pow(
    const T2& x, const complex<xad::AReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x,
                                      const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, const complex<T>& y)
{
    return pow(x, complex<xad::FReal<T>>(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<T>& x, const complex<xad::FReal<T>>& y)
{
    return pow(complex<xad::FReal<T>>(x), y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, const xad::FReal<T>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, const T& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, int y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, unsigned y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, unsigned short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, unsigned long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const complex<xad::FReal<T>>& x, unsigned long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, class T2>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::FReal<T>>>::type pow(
    const complex<xad::FReal<T>>& x, const T2& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const xad::FReal<T>& x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(const T& x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(int x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(short x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(long x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(long long x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(unsigned long x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(unsigned x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(unsigned long long x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T>
XAD_INLINE complex<xad::FReal<T>> pow(unsigned short x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, class T2>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::FReal<T>>>::type pow(
    const T2& x, const complex<xad::FReal<T>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> sqrt(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::sqrt_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> sqrt(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::sqrt_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> sin(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::sin_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> sin(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::sin_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> cos(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::cos_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> cos(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::cos_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> tan(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::tan_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> tan(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::tan_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> asin(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::asin_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> asin(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::asin_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> acos(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::acos_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> acos(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::acos_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> atan(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::atan_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> atan(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::atan_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> sinh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::sinh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> sinh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::sinh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> cosh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::cosh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> cosh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::cosh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> tanh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::tanh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> tanh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::tanh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> asinh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::asinh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> asinh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::asinh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> acosh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::acosh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> acosh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::acosh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::AReal<T>> atanh(const std::complex<xad::AReal<T>>& z)
{
    return ::xad::detail::atanh_impl(z);
}

template <class T>
XAD_INLINE std::complex<xad::FReal<T>> atanh(const std::complex<xad::FReal<T>>& z)
{
    return ::xad::detail::atanh_impl(z);
}

}  // namespace std

namespace xad
{
namespace detail
{

template <class T>
XAD_INLINE T norm_impl(const std::complex<T>& x)
{
    return x.real() * x.real() + x.imag() * x.imag();
}

template <class T>
XAD_INLINE T abs_impl(const std::complex<T>& x)
{
    using std::sqrt;
    if (xad::isinf(x.real()) || xad::isinf(x.imag()))
        return std::numeric_limits<double>::infinity();
    return xad::hypot(x.real(), x.imag());
}

template <class T>
XAD_INLINE std::complex<T> exp_impl(const std::complex<T>& z)
{
    using std::cos;
    using std::exp;
    using std::sin;
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (xad::isinf(z.real()))
    {
        if (z.real() > 0.0)
        {
            if (z.imag() == 0.0)
                return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
            if (xad::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                       std::numeric_limits<nested>::quiet_NaN());
            if (xad::isnan(z.imag()))
                return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                       std::numeric_limits<nested>::quiet_NaN());
        }
        else
        {
            if (xad::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(0.0, 0.0);
            if (xad::isnan(z.imag()))
                return std::complex<T>(0.0, 0.0);
        }
    }
    else if (xad::isnan(z.real()))
    {
        if (z.imag() == 0.0 && !xad::signbit(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
        else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    T e = ::xad::exp(z.real());
    return std::complex<T>(e * cos(z.imag()), e * sin(z.imag()));
}

template <class T1, class T2>
XAD_INLINE std::complex<typename xad::ExprTraits<T1>::value_type> polar_impl(const T1& r,
                                                                             const T2& theta)
{
    using std::cos;
    using std::sin;
    typedef typename xad::ExprTraits<T1>::value_type base_type;
    return std::complex<base_type>(base_type(r * cos(theta)), base_type(r * sin(theta)));
}

template <class T>
XAD_INLINE std::complex<T> sqrt_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (xad::isinf(z.real()) && z.real() < 0.0)
    {
        if (xad::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, std::numeric_limits<nested>::infinity());
        if (xad::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::infinity());
    }
    if (xad::isinf(z.real()) && z.real() > 0.0)
    {
        if (xad::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);

        if (xad::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    if (xad::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::infinity());

    return ::xad::detail::polar_impl(sqrt(abs(z)), arg(z) * T(0.5));
}

template <class T>
XAD_INLINE std::complex<T> sinh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    auto cls = xad::fpclassify(z.real());
    if (cls == FP_INFINITE && xad::isinf(z.imag()) && z.real() > 0.0 && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());
    if (cls == FP_NAN && z.imag() == 0.0 && !xad::signbit(z.imag()))
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    if (cls == FP_ZERO && !xad::signbit(z.real()))
    {
        if ((xad::isinf(z.imag()) && z.imag() > 0.0) || xad::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
    }
    return (exp(z) - exp(-z)) / T(2.0);
}

template <class T>
XAD_INLINE std::complex<T> cosh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    auto cls = xad::fpclassify(z.real());
    if (cls == FP_INFINITE && xad::isinf(z.imag()) && z.real() > 0.0 && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());
    if (cls == FP_NAN && z.imag() == 0.0 && !xad::signbit(z.imag()))
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    if (cls == FP_ZERO && !xad::signbit(z.real()))
    {
        if ((xad::isinf(z.imag()) && z.imag() > 0.0) || xad::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    }
    return (exp(z) + exp(-z)) / T(2.0);
}

template <class T>
XAD_INLINE std::complex<T> tanh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (z.real() == 0.0)
    {
        if (xad::isinf(z.imag()) && z.imag() > 0.0)
        {
#if defined(__APPLE__) || (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
#endif
        }
        if (xad::isnan(z.imag()))
        {
#if defined(__APPLE__) | (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
#endif
        }
    }
    if (xad::isinf(z.real()) && z.real() > 0.0 && (z.imag() > 0.0 || xad ::isnan(z.imag())))
        return std::complex<T>(1.0, 0.0);
    if (xad::isnan(z.real()) && z.imag() == 0.0)
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    return sinh(z) / cosh(z);
}

template <class T>
XAD_INLINE std::complex<T> asinh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (xad::isinf(z.real()) && z.real() > 0.0)
    {
        if (xad::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.25);
        if (xad::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        if (z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
    }
    if (xad::isnan(z.real()))
    {
        if (xad::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        if (z.imag() == 0.0 && !xad::signbit(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    }
    if (xad::isinf(z.imag()) && z.imag() > 0.0 && xad::isfinite(z.real()) && z.real() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               3.141592653589793238462643383279502884197169399 * 0.5);
    return log(z + sqrt(T(1.0) + (z * z)));
}

template <class T>
XAD_INLINE std::complex<T> acosh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (xad::isinf(z.imag()) && z.imag() > 0.0)
    {
        if (xad::isfinite(z.real()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.5);
        if (xad::isinf(z.real()) && z.real() < 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.75);
    }
    if (xad::isnan(z.imag()))
    {
        if (z.real() == 0.0)
#if defined(__APPLE__) | (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   3.141592653589793238462643383279502884197169399 * 0.5);
#endif
        else if (xad::isinf(z.real()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    if (xad::isinf(z.real()) && xad::isfinite(z.imag()) && z.imag() > 0.0)
    {
        if (z.real() < 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399);
        else
            return std::complex<T>(std::numeric_limits<nested>::infinity(), +0.0);
    }
    if (xad::isnan(z.real()) && xad::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());

    return log(z + sqrt(z + T(1.0)) * sqrt(z - T(1.0)));
}

template <class T>
XAD_INLINE std::complex<T> atanh_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (xad::isinf(z.real()) && z.real() > 0.0)
    {
        if (xad::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
        if (xad::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
        if (xad::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    }
    if (xad::isnan(z.real()) && xad::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    if (z.real() == 1.0 && z.imag() == 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
    if (z.real() > 0.0 && xad::isfinite(z.real()) && xad::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    if (z.real() == 0.0)
    {
        if (z.imag() == 0.0)
            return std::complex<T>(0.0, 0.0);
        if (xad::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
    }
    return (log(T(1.0) + z) - log(T(1.0) - z)) / T(2.0);
}

template <class T>
XAD_INLINE std::complex<T> sin_impl(const std::complex<T>& z)
{
    // -i * sinh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> sinhiz = sinh(iz);
    return std::complex<T>(sinhiz.imag(), -sinhiz.real());
}

template <class T>
XAD_INLINE std::complex<T> cos_impl(const std::complex<T>& z)
{
    // cosh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    return cosh(iz);
}

template <class T>
XAD_INLINE std::complex<T> tan_impl(const std::complex<T>& z)
{
    // -i * tanh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> tanhiz = tanh(iz);
    return std::complex<T>(tanhiz.imag(), -tanhiz.real());
}

template <class T>
XAD_INLINE std::complex<T> asin_impl(const std::complex<T>& z)
{
    // -i * asinh(i*z);
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> asinhiz = asinh(iz);
    return std::complex<T>(asinhiz.imag(), -asinhiz.real());
}

template <class T>
XAD_INLINE std::complex<T> acos_impl(const std::complex<T>& z)
{
    typedef typename xad::ExprTraits<T>::nested_type nested;
    if (z.real() == 0.0)
    {
        if (z.imag() == 0.0 && !xad::signbit(z.imag()))
            return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5, -0.0);
        if (xad::isnan(z.imag()))
            return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5,
                                   -std::numeric_limits<nested>::quiet_NaN());
    }
    if (xad::isfinite(z.real()) && xad::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5,
                               -std::numeric_limits<nested>::infinity());
    if (xad::isinf(z.real()))
    {
        if (z.real() < 0.0)
        {

            if (xad::isfinite(z.imag()) && z.imag() >= 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399,
                                       -std::numeric_limits<nested>::infinity());
            if (xad::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.75,
                                       -std::numeric_limits<nested>::infinity());
        }
        else
        {
            if (xad::isfinite(z.imag()) && z.imag() >= 0.0)
                return std::complex<T>(+0.0, -std::numeric_limits<nested>::infinity());
            if (xad::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.25,
                                       -std::numeric_limits<nested>::infinity());
        }
        if (xad::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::infinity());
    }
    if (xad::isnan(z.real()))
    {
        if (xad::isfinite(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
        else if (xad::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   -std::numeric_limits<nested>::infinity());
    }

    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> lnizsqrt = log(iz + sqrt(T(1.0) - (z * z)));
    std::complex<T> ilnizsqrt(-lnizsqrt.imag(), lnizsqrt.real());
    return T(3.141592653589793238462643383279502884197169399 * 0.5) + ilnizsqrt;
}

template <class T>
XAD_INLINE std::complex<T> atan_impl(const std::complex<T>& z)
{
    // -i * atanh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> atanhiz = atanh(iz);
    return std::complex<T>(atanhiz.imag(), -atanhiz.real());
}

template <class T>
XAD_INLINE T arg_impl(const std::complex<T>& z)
{
    using std::atan2;
    return atan2(z.imag(), z.real());
}

template <class Scalar, class Derived>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type arg_impl(
    const xad::Expression<Scalar, Derived>& x)
{
    using std::atan2;

    // as this function returns constants only depending on > or < 0,
    // where derivatives are 0 anyway, we can return scalars converted to the
    // underlying expression type
    typedef typename xad::ExprTraits<Derived>::value_type ret_type;
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    (void)x;  // silence unused warning
    return ret_type();
#else
    if (x > 0.0)
        return ret_type();
    else if (x < 0.0)
        return ret_type(3.141592653589793238462643383279502884197169399);  // PI
    else
        return atan2(ret_type(), ret_type(x));  // for correct handling of +/- zero
#endif
}

#if (defined(_MSC_VER) && (_MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) &&             \
    !defined(__clang__)
template <class Scalar, class Derived>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type proj_impl(
    const xad::Expression<Scalar, Derived>& x)
{
    return typename xad::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived>
XAD_INLINE std::complex<typename xad::ExprTraits<Derived>::value_type> proj_impl(
    const xad::Expression<Scalar, Derived>& x)
{
    if (xad::isinf(x))
        return std::complex<typename xad::ExprTraits<Derived>::value_type>(
            std::numeric_limits<typename xad::ExprTraits<Derived>::nested_type>::infinity());
    else
        return std::complex<typename xad::ExprTraits<Derived>::value_type>(x);
}
#endif

}  // namespace detail
}  // namespace xad

/*******************************************************************************

   An AD-enabled equivalent of std::complex.

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

template <class Scalar, class T, class Deriv>
class complex<xad::ADTypeBase<Scalar, T, Deriv>> : public xad::detail::complex_impl<T>
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

template <class T, std::size_t N>
class complex<xad::AReal<T, N>>
    : public complex<
          xad::ADTypeBase<T, xad::AReal<T, N>, typename xad::DerivativesTraits<T, N>::type>>
{
  public:
    typedef complex<
        xad::ADTypeBase<T, xad::AReal<T, N>, typename xad::DerivativesTraits<T, N>::type>>
        base;

    // inheriting template constructors doesn't work in all compilers

    XAD_INLINE complex(const xad::AReal<T, N>& areal = xad::AReal<T, N>(),
                       const xad::AReal<T, N>& aimag = xad::AReal<T, N>())
        : base(areal, aimag)
    {
    }

    template <class X>
    XAD_INLINE complex(const X& areal,
                       typename std::enable_if<!xad::ExprTraits<X>::isExpr>::type* = nullptr)
        : base(xad::AReal<T, N>(areal), xad::AReal<T, N>())
    {
    }

    template <class X>
    XAD_INLINE complex(  // cppcheck-suppress noExplicitConstructor
        const X& areal,
        typename std::enable_if<xad::ExprTraits<X>::isExpr &&
                                xad::ExprTraits<X>::direction == xad::DIR_REVERSE>::type* = nullptr)
        : base(xad::AReal<T, N>(areal), xad::AReal<T, N>())
    {
    }

    template <class X>
    XAD_INLINE complex(const complex<X>& o)
        : base(xad::AReal<T, N>(o.real()), xad::AReal<T, N>(o.imag()))
    {
    }

    using base::operator+=;
    using base::operator-=;
    using base::operator*=;
    using base::operator/=;
};

template <class T, std::size_t N>
class complex<xad::FReal<T, N>>
    : public complex<
          xad::ADTypeBase<T, xad::FReal<T, N>, typename xad::FRealTraits<T, N>::derivative_type>>
{
  public:
    typedef complex<
        xad::ADTypeBase<T, xad::FReal<T, N>, typename xad::FRealTraits<T, N>::derivative_type>>
        base;

    // inheriting template constructors doesn't work in all compilers

    XAD_INLINE complex(const xad::FReal<T, N>& areal =
                           xad::FReal<T, N>(),  // cppcheck-suppress noExplicitConstructor
                       const xad::FReal<T, N>& aimag = xad::FReal<T, N>())
        : base(areal, aimag)
    {
    }

    template <class X>
    XAD_INLINE complex(const X& areal,  // cppcheck-suppress noExplicitConstructor
                       typename std::enable_if<!xad::ExprTraits<X>::isExpr>::type* = nullptr)
        : base(xad::FReal<T, N>(areal), xad::FReal<T, N>())
    {
    }

    template <class X>
    XAD_INLINE complex(  // cppcheck-suppress noExplicitConstructor
        const X& areal,
        typename std::enable_if<xad::ExprTraits<X>::isExpr &&
                                xad::ExprTraits<X>::direction == xad::DIR_FORWARD>::type* = nullptr)
        : base(xad::FReal<T, N>(areal), xad::FReal<T, N>())
    {
    }

    template <class X>
    XAD_INLINE complex(const complex<X>& o)
        : base(xad::FReal<T, N>(o.real()),
               xad::FReal<T, N>(o.imag()))  // cppcheck-suppress noExplicitConstructor
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

// read access to value and derivatives, only for scalars
template <class T, std::size_t N>
XAD_INLINE std::complex<T> derivative(const std::complex<AReal<T, N>>& z)
{
    static_assert(N == 1,
                  "Global derivative function is only defined for scalar derivatives - use "
                  "derivative(z.real()) instead");
    return std::complex<T>(derivative(z.real()), derivative(z.imag()));
}

template <class T, std::size_t N>
XAD_INLINE std::complex<T> derivative(const std::complex<FReal<T, N>>& z)
{
    static_assert(N == 1,
                  "Global derivative function is only defined for scalar derivatives - use "
                  "derivative(z.real()) instead");
    return std::complex<T>(derivative(z.real()), derivative(z.imag()));
}

template <class T, std::size_t N>
XAD_INLINE std::complex<T> value(const std::complex<AReal<T, N>>& z)
{
    return std::complex<T>(value(z.real()), value(z.imag()));
}

template <class T, std::size_t N>
XAD_INLINE std::complex<T> value(const std::complex<FReal<T, N>>& z)
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
template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type arg_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x);

template <class T>
XAD_INLINE T arg_impl(const std::complex<T>& z);

#if (defined(_MSC_VER) && (_MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) &&             \
    !defined(__clang__)
template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type proj_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x);
#else
template <class Scalar, class Derived, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Derived>::value_type> proj_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x);
#endif

template <class T>
XAD_INLINE T norm_impl(const std::complex<T>& x);

}  // namespace detail

}  // namespace xad

namespace std
{

// access to real / imag
template <class Scalar, class T, class Deriv>
XAD_INLINE T real(const std::complex<xad::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().real();
}

template <class Scalar, class T, class Deriv>
XAD_INLINE T& real(std::complex<xad::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().real();
}

template <class Scalar, class Expr>
XAD_INLINE typename xad::ExprTraits<Expr>::value_type real(
    const xad::Expression<Scalar, Expr>& other)
{
    return other.derived();
}

template <class Scalar, class T, class Deriv>
XAD_INLINE T imag(const std::complex<xad::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class T, class Deriv>
XAD_INLINE T& imag(std::complex<xad::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class Expr, class Deriv>
XAD_INLINE typename xad::ExprTraits<Expr>::value_type imag(
    const xad::Expression<Scalar, Expr, Deriv>&)
{
    return typename xad::ExprTraits<Expr>::value_type(0);
}

///////////////////////// operators
template <class T, std::size_t N>
XAD_INLINE const std::complex<xad::AReal<T, N>>& operator+(const std::complex<xad::AReal<T, N>>& x)
{
    return x;
}

template <class T, std::size_t N>
XAD_INLINE const std::complex<xad::FReal<T, N>>& operator+(const std::complex<xad::FReal<T, N>>& x)
{
    return x;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(const std::complex<xad::AReal<T, N>>& x)
{
    return std::complex<xad::AReal<T, N>>(-x.real(), -x.imag());
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const std::complex<xad::FReal<T, N>>& x)
{
    return std::complex<xad::FReal<T, N>>(-x.real(), -x.imag());
}

// operator== - lots of variants here, I'm sure this could be done cleaner...

template <class T, std::size_t N>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T, N>>& lhs,
                           const std::complex<xad::AReal<T, N>>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <class T, std::size_t N>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T, N>>& lhs,
                           const std::complex<xad::FReal<T, N>>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <class T, class Expr, std::size_t N, class Deriv>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T, N>>& lhs,
                           const xad::Expression<T, Expr, Deriv>& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, class Expr, class Deriv, std::size_t N>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T, N>>& lhs,
                           const xad::Expression<T, Expr, Deriv>& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N>
XAD_INLINE bool operator==(const std::complex<xad::AReal<T, N>>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N>
XAD_INLINE bool operator==(const std::complex<xad::FReal<T, N>>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator==(const xad::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<xad::AReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator==(const xad::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<xad::FReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N>
XAD_INLINE bool operator==(const T& rhs, const std::complex<xad::AReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <class T, std::size_t N>
XAD_INLINE bool operator==(const T& rhs, const std::complex<xad::FReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// operator !=

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T, N>>& lhs,
                           const std::complex<xad::AReal<T, N>>& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T, N>>& lhs,
                           const std::complex<xad::FReal<T, N>>& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T, N>>& lhs,
                           const xad::Expression<T, Expr, Deriv>& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T, N>>& lhs,
                           const xad::Expression<T, Expr, Deriv>& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const std::complex<xad::AReal<T, N>>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const std::complex<xad::FReal<T, N>>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator!=(const xad::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<xad::AReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE bool operator!=(const xad::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<xad::FReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const T& rhs, const std::complex<xad::AReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

template <class T, std::size_t N>
XAD_INLINE bool operator!=(const T& rhs, const std::complex<xad::FReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

// operator+

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<xad::AReal<T, N>>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<T> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<xad::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Deriv, class Expr>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator+(
    const std::complex<T>& lhs, const xad::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(const std::complex<T>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(const xad::AReal<T, N>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(const T& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator+(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator+(
    const xad::Expression<T, Expr, Deriv>& rhs, const std::complex<T>& lhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const std::complex<T>& lhs,
                                                    std::complex<xad::FReal<T, N>> rhs)
{
    rhs += lhs;
    return rhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const std::complex<T>& lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(std::complex<xad::FReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const xad::FReal<T, N>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const xad::FReal<T, N>& rhs,
                                                    const std::complex<T>& lhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const T& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator+(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

// operator-

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<xad::AReal<T, N>>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(const std::complex<T>& lhs,
                                                    std::complex<xad::AReal<T, N>> rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<T> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<xad::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator-(
    const std::complex<T>& lhs, const xad::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator-(
    const xad::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(const xad::AReal<T, N>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(const T& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator-(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const std::complex<T>& lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T, N>> z = rhs;
    lhs -= z;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<T> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<xad::FReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const xad::FReal<T, N>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const xad::FReal<T, N>& rhs,
                                                    std::complex<T> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const T& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator-(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

// operator*
template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<xad::AReal<T, N>>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(const std::complex<T>& lhs,
                                                    const std::complex<xad::AReal<T, N>>& rhs)
{
    return rhs * lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(const std::complex<T>& lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(std::complex<xad::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator*(
    const std::complex<T>& lhs, const xad::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator*(
    const xad::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(const xad::AReal<T, N>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(const T& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator*(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(const std::complex<T>& lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T, N>> z = rhs;
    lhs *= z;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<T> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<xad::FReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(const xad::FReal<T, N>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(const xad::FReal<T, N>& rhs,
                                                    std::complex<T> lhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(const T& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator*(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

// operator/

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<xad::AReal<T, N>>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<xad::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(const std::complex<T>& lhs,
                                                    std::complex<xad::AReal<T, N>> rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<T> lhs,
                                                    const xad::AReal<T, N>& rhs)
{
    std::complex<xad::AReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<xad::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(std::complex<xad::AReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator/(
    const std::complex<T>& lhs, const xad::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> operator/(
    const xad::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename xad::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(const xad::AReal<T, N>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(const T& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::AReal<T, N>> operator/(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::AReal<T, N>> lhs)
{
    return std::complex<xad::AReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(const std::complex<T>& lhs,
                                                    const std::complex<xad::FReal<T, N>>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<xad::FReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    std::complex<xad::FReal<T, N>> z = rhs;
    lhs /= z;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<T> lhs,
                                                    const xad::FReal<T, N>& rhs)
{
    std::complex<xad::FReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<xad::FReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(std::complex<xad::FReal<T, N>> lhs,
                                                    const xad::Expression<T, Expr, Deriv>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(const xad::FReal<T, N>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(const xad::FReal<T, N>& rhs,
                                                    std::complex<T> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(const T& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
XAD_INLINE std::complex<xad::FReal<T, N>> operator/(const xad::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<xad::FReal<T, N>> lhs)
{
    return std::complex<xad::FReal<T, N>>(rhs) / lhs;
}
/////////////////////// math functions

template <class T, std::size_t N>
XAD_INLINE xad::AReal<T, N> arg(const complex<xad::AReal<T, N>>& x)
{
    return xad::detail::arg_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE xad::FReal<T, N> arg(const complex<xad::FReal<T, N>>& x)
{
    return xad::detail::arg_impl(x);
}

template <class Scalar, class Derived, class Deriv>
typename xad::ExprTraits<Derived>::value_type arg(const xad::Expression<Scalar, Derived, Deriv>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T, std::size_t N>
typename xad::AReal<T, N> arg(const xad::AReal<T, N>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T, std::size_t N>
typename xad::FReal<T, N> arg(const xad::FReal<T, N>& x)
{
    return ::xad::detail::arg_impl(x);
}

template <class T, class Scalar, class Deriv>
XAD_INLINE T norm(const complex<xad::ADTypeBase<Scalar, T, Deriv>>& x)
{
    return ::xad::detail::norm_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE xad::AReal<T, N> norm(const complex<xad::AReal<T, N>>& x)
{
    return ::xad::detail::norm_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE xad::FReal<T, N> norm(const complex<xad::FReal<T, N>>& x)
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
template <class Scalar, class Derived, class Deriv>
XAD_INLINE auto norm(const xad::Expression<Scalar, Derived, Deriv>& x) -> decltype(x * x)
{
    return x * x;
}

template <class T, std::size_t N>
XAD_INLINE xad::AReal<T, N> abs(const complex<xad::AReal<T, N>>& x)
{
    return xad::detail::abs_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE xad::FReal<T, N> abs(const complex<xad::FReal<T, N>>& x)
{
    return xad::detail::abs_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> conj(const complex<xad::AReal<T, N>>& z)
{
    complex<xad::AReal<T, N>> ret(z.real(), -z.imag());
    return ret;
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> conj(const complex<xad::FReal<T, N>>& z)
{
    complex<xad::FReal<T, N>> ret(z.real(), -z.imag());
    return ret;
}

#if ((defined(_MSC_VER) && (_MSC_VER < 1920)) || (defined(__GNUC__) && __GNUC__ < 7)) &&           \
    !defined(__clang__)
template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type conj(
    const xad::Expression<Scalar, Derived, Deriv>& x)
{
    return typename xad::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Derived>::value_type> conj(
    const xad::Expression<Scalar, Derived, Deriv>& x)
{
    return complex<typename xad::ExprTraits<Derived>::value_type>(x);
}
#endif

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> proj(const std::complex<xad::AReal<T, N>>& z)
{
    if (xad::isinf(z.real()) || xad::isinf(z.imag()))
    {
        typedef typename xad::ExprTraits<T>::nested_type type;
        const type infty = std::numeric_limits<type>::infinity();
        if (xad::signbit(z.imag()))
            return complex<xad::AReal<T, N>>(infty, -0.0);
        else
            return complex<xad::AReal<T, N>>(infty, 0.0);
    }
    else
        return z;
}

template <class T, std::size_t N>
complex<xad::FReal<T, N>> proj(const std::complex<xad::FReal<T, N>>& z)
{
    if (xad::isinf(z.real()) || xad::isinf(z.imag()))
    {
        typedef typename xad::ExprTraits<T>::nested_type type;
        const type infty = std::numeric_limits<type>::infinity();
        if (xad::signbit(z.imag()))
            return complex<xad::FReal<T, N>>(infty, -0.0);
        else
            return complex<xad::FReal<T, N>>(infty, 0.0);
    }
    else
        return z;
}

template <class Scalar, class Derived, class Deriv>
XAD_INLINE auto proj(const xad::Expression<Scalar, Derived, Deriv>& x)
    -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE auto proj(const xad::AReal<T, N>& x) -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

template <class T, std::size_t N>
XAD_INLINE auto proj(const xad::FReal<T, N>& x) -> decltype(::xad::detail::proj_impl(x))
{
    return ::xad::detail::proj_impl(x);
}

// T and expr
// expr and T
// different expr (derived1, derived2 - returns scalar)

template <class T, std::size_t N = 1>
XAD_INLINE complex<xad::AReal<T, N>> polar(const xad::AReal<T, N>& r,
                                           const xad::AReal<T, N>& theta = xad::AReal<T, N>())
{
    return xad::detail::polar_impl(r, theta);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> polar(const xad::FReal<T, N>& r,
                                           const xad::FReal<T, N>& theta = xad::FReal<T, N>())
{
    return xad::detail::polar_impl(r, theta);
}

template <class Scalar, class Expr, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr, Deriv>& r,
    const xad::Expression<Scalar, Expr, Deriv>& theta)
{
    typedef typename xad::ExprTraits<Expr>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

#if defined(_MSC_VER) && _MSC_VER < 1920
// VS 2017 needs loads of specialisations to resolve the right overload and avoid calling the
// std::version

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N = 1>
XAD_INLINE complex<xad::AReal<Scalar, N>> polar(const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& r,
                                                const xad::AReal<Scalar, N>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar, N>(r), theta);
}

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N = 1>
XAD_INLINE complex<xad::AReal<Scalar, N>> polar(
    const xad::AReal<Scalar, N>& r, const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar, N>(theta));
}

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& r,
                                                const xad::FReal<Scalar, N>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar, N>(r), theta);
}

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(
    const xad::FReal<Scalar, N>& r, const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar, N>(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2, std::size_t M = 1>
XAD_INLINE complex<xad::AReal<Scalar, M>> polar(
    const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, typename DerivativesTraits<Scalar, M>::type>& r,
    const xad::AReal<Scalar, M>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar, M>(r), theta);
}

template <class Scalar, class Op, class Expr1, class Expr2, std::size_t M = 1>
XAD_INLINE complex<xad::AReal<Scalar, M>> polar(
    const xad::AReal<Scalar, M>& r,
    const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, typename DerivativesTraits<Scalar, M>::type>&
        theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar, M>(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2, class Deriv, std::size_t N>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(
    const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& r, const xad::FReal<Scalar, N>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar, N>(r), theta);
}

template <class Scalar, class Op, class Expr1, class Expr2, class Deriv, std::size_t N>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(
    const xad::FReal<Scalar, N>& r, const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar, N>(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op3, Expr3, Deriv>& r,
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& r,
    const xad::UnaryExpr<Scalar, Op3, Expr3, Deriv>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Op2, class Expr2, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op1, Expr1, Deriv>& r,
    const xad::UnaryExpr<Scalar, Op2, Expr2, Deriv>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Expr4,
          class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op3, Expr3, Expr4, Deriv>& r,
    const xad::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& theta)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    double r, const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr1>::value_type(r),
                                   typename xad::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr1>::value_type> polar(
    const xad::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& r, double theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr1>::value_type(r),
                                   typename xad::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    double r, const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r),
                                   typename xad::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar, class Op, class Expr, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::UnaryExpr<Scalar, Op, Expr, Deriv>& r, double theta)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r),
                                   typename xad::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE complex<xad::AReal<Scalar, M>> polar(const xad::AReal<Scalar, M>& r, double theta)
{
    return xad::detail::polar_impl(r, xad::AReal<Scalar, M>(theta));
}

template <class Scalar, std::size_t N = 1>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(const xad::FReal<Scalar, N>& r, double theta)
{
    return xad::detail::polar_impl(r, xad::FReal<Scalar, N>(theta));
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE complex<xad::AReal<Scalar, M>> polar(double r, const xad::AReal<Scalar, M>& theta)
{
    return xad::detail::polar_impl(xad::AReal<Scalar, M>(r), theta);
}

template <class Scalar, std::size_t N = 1>
XAD_INLINE complex<xad::FReal<Scalar, N>> polar(double r, const xad::FReal<Scalar, N>& theta)
{
    return xad::detail::polar_impl(xad::FReal<Scalar, N>(r), theta);
}

#endif

// 2 different expression types passed:
// we only enable this function if the underlying value_type of both expressions
// is the same
template <class Scalar, class Expr1, class Expr2, class Deriv>
XAD_INLINE typename std::enable_if<std::is_same<typename xad::ExprTraits<Expr1>::value_type,
                                                typename xad::ExprTraits<Expr2>::value_type>::value,
                                   complex<typename xad::ExprTraits<Expr1>::value_type>>::type
polar(const xad::Expression<Scalar, Expr1, Deriv>& r,
      const xad::Expression<Scalar, Expr2, Deriv>& theta = 0)
{
    typedef typename xad::ExprTraits<Expr1>::value_type type;
    return xad::detail::polar_impl(type(r), type(theta));
}

// T, expr - only enabled if T is scalar
template <class Scalar, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> polar(
    Scalar r, const xad::Expression<Scalar, Expr, Deriv>& theta = 0)
{
    return xad::detail::polar_impl(typename xad::ExprTraits<Expr>::value_type(r), theta.derived());
}

// expr, T - only enabled if T is scalar
template <class Scalar, class Expr, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr, Deriv>& r, Scalar theta = Scalar())
{
    return xad::detail::polar_impl(r.derived(), typename xad::ExprTraits<Expr>::value_type(theta));
}

// just one expr, as second parameter is optional
template <class Scalar, class Expr, class Deriv>
XAD_INLINE complex<typename xad::ExprTraits<Expr>::value_type> polar(
    const xad::Expression<Scalar, Expr, Deriv>& r)
{
    return complex<typename xad::ExprTraits<Expr>::value_type>(r);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> exp(const complex<xad::AReal<T, N>>& z)
{
    return xad::detail::exp_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> exp(const complex<xad::FReal<T, N>>& z)
{
    return xad::detail::exp_impl(z);
}

template <class T, std::size_t N = 1>
XAD_INLINE complex<xad::AReal<T, N>> log(const complex<xad::AReal<T, N>>& z)
{
    return complex<xad::AReal<T, N>>(log(xad::detail::abs_impl(z)), xad::detail::arg_impl(z));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> log(const complex<xad::FReal<T, N>>& z)
{
    return complex<xad::FReal<T, N>>(log(xad::detail::abs_impl(z)), xad::detail::arg_impl(z));
}

template <class T, std::size_t N = 1>
XAD_INLINE complex<xad::AReal<T, N>> log10(const complex<xad::AReal<T, N>>& z)
{
    // log(z) * 1/log(10)
    return log(z) * T(0.43429448190325182765112891891661);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> log10(const complex<xad::FReal<T, N>>& z)
{
    // log(z) * 1/log(10)
    return log(z) * T(0.43429448190325182765112891891661);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x,
                                         const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x,
                                         const xad::AReal<T, N>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, const complex<T>& y)
{
    return pow(x, complex<xad::AReal<T, N>>(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<T>& x, const complex<xad::AReal<T, N>>& y)
{
    return pow(complex<xad::AReal<T, N>>(x), y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, const T& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, class T2, std::size_t N>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::AReal<T, N>>>::type
pow(const complex<xad::AReal<T, N>>& x, const T2& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, int y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, unsigned y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, unsigned short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, unsigned long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const complex<xad::AReal<T, N>>& x, unsigned long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const xad::AReal<T, N>& x,
                                         const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(const T& x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(int x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(short x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(long x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(long long x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(unsigned x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(unsigned short x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(unsigned long x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::AReal<T, N>> pow(unsigned long long x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, class T2, std::size_t N>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::AReal<T, N>>>::type
pow(const T2& x, const complex<xad::AReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x,
                                         const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, const complex<T>& y)
{
    return pow(x, complex<xad::FReal<T, N>>(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<T>& x, const complex<xad::FReal<T, N>>& y)
{
    return pow(complex<xad::FReal<T, N>>(x), y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x,
                                         const xad::FReal<T, N>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, const T& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, int y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, unsigned y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, unsigned short y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, unsigned long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const complex<xad::FReal<T, N>>& x, unsigned long long y)
{
    return xad::detail::exp_impl(log(x) * T(y));
}

template <class T, class T2, std::size_t N>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::FReal<T, N>>>::type
pow(const complex<xad::FReal<T, N>>& x, const T2& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const xad::FReal<T, N>& x,
                                         const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(const T& x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(int x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(short x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(long x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(long long x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(unsigned long x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(unsigned x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(unsigned long long x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
XAD_INLINE complex<xad::FReal<T, N>> pow(unsigned short x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(T(x)) * y);
}

template <class T, class T2, std::size_t N>
XAD_INLINE typename std::enable_if<xad::ExprTraits<T2>::isExpr, complex<xad::FReal<T, N>>>::type
pow(const T2& x, const complex<xad::FReal<T, N>>& y)
{
    return xad::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> sqrt(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::sqrt_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> sqrt(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::sqrt_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> sin(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::sin_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> sin(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::sin_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> cos(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::cos_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> cos(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::cos_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> tan(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::tan_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> tan(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::tan_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> asin(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::asin_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> asin(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::asin_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> acos(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::acos_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> acos(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::acos_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> atan(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::atan_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> atan(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::atan_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> sinh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::sinh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> sinh(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::sinh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> cosh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::cosh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> cosh(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::cosh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> tanh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::tanh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> tanh(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::tanh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> asinh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::asinh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> asinh(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::asinh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> acosh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::acosh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> acosh(const std::complex<xad::FReal<T, N>>& z)
{
    return ::xad::detail::acosh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::AReal<T, N>> atanh(const std::complex<xad::AReal<T, N>>& z)
{
    return ::xad::detail::atanh_impl(z);
}

template <class T, std::size_t N>
XAD_INLINE std::complex<xad::FReal<T, N>> atanh(const std::complex<xad::FReal<T, N>>& z)
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

template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type arg_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x)
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
template <class Scalar, class Derived, class Deriv>
XAD_INLINE typename xad::ExprTraits<Derived>::value_type proj_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x)
{
    return typename xad::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived, class Deriv>
XAD_INLINE std::complex<typename xad::ExprTraits<Derived>::value_type> proj_impl(
    const xad::Expression<Scalar, Derived, Deriv>& x)
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

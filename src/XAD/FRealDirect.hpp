/*******************************************************************************

   Direct Forward mode - without expression templates.

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
#include <XAD/Traits.hpp>

namespace xad
{

template <class Scalar, std::size_t N = 1>
struct FRealDirect : public RealDirect<FReal<Scalar, N>, FRealDirect<Scalar, N>>
{
    typedef FReal<Scalar, N> base_type;
    typedef typename base_type::derivative_type DerivativeType;

    typedef RealDirect<FReal<Scalar, N>, FRealDirect<Scalar, N>> base_class;
    using base_class::base_class;
    constexpr XAD_INLINE FRealDirect() : base_class() {}

    template <class T>
    FRealDirect(const T &val, const T &der) : base_class(val)
    {
        this->base().setDerivative(der);
    }

    XAD_INLINE DerivativeType &derivative() { return this->base().derivative(); }
    XAD_INLINE DerivativeType getDerivative() const { return this->base().derivative(); }
};

template <class Scalar, std::size_t N>
XAD_INLINE const Scalar &value(const FRealDirect<Scalar, N> &x)
{
    return x.value();
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar &value(FRealDirect<Scalar, N> &x)
{
    return x.value();
}

template <class Scalar, std::size_t N>
XAD_INLINE const typename FRealDirect<Scalar, N>::DerivativeType &derivative(
    const FRealDirect<Scalar, N> &x)
{
    return x.derivative();
}

template <class Scalar, std::size_t N>
XAD_INLINE typename FRealDirect<Scalar, N>::DerivativeType &derivative(FRealDirect<Scalar, N> &x)
{
    return x.derivative();
}

typedef FRealDirect<double> FDD;
typedef FRealDirect<float> FDF;
}  // namespace xad

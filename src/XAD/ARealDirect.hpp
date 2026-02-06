/*******************************************************************************

   Direct Adjoint mode, without expression templates.

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
#include <XAD/RealDirect.hpp>
#include <XAD/Tape.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

namespace xad
{

template <class Scalar, std::size_t N = 1>
struct ARealDirect : public RealDirect<AReal<Scalar, N>, ARealDirect<Scalar, N>>
{
    typedef Tape<Scalar, N> tape_type;
    typedef AReal<Scalar, N> base_type;
    typedef typename base_type::derivative_type derivative_type;
    typedef typename tape_type::slot_type slot_type;

    typedef RealDirect<AReal<Scalar, N>, ARealDirect<Scalar, N>> base_class;
    using base_class::base_class;

    XAD_INLINE void setDerivative(const derivative_type &a) { this->base().setDerivative(a); }

    XAD_INLINE void setAdjoint(const derivative_type &a) { this->base().setDerivative(a); }

    XAD_INLINE slot_type getSlot() const { return this->base().getSlot(); }

    XAD_INLINE bool shouldRecord() { return this->base().shouldRecord(); }
};

template <class T, std::size_t N>
XAD_INLINE const T &value(const ARealDirect<T, N> &x)
{
    return x.base().value();
}

template <class T, std::size_t N>
XAD_INLINE T &value(ARealDirect<T, N> &x)
{
    return x.base().value();
}

template <class T, std::size_t N>
XAD_INLINE const typename ARealDirect<T, N>::derivative_type &derivative(const ARealDirect<T, N> &x)
{
    return x.base().derivative();
}

template <class T, std::size_t N>
XAD_INLINE typename ARealDirect<T, N>::derivative_type &derivative(ARealDirect<T, N> &x)
{
    return x.base().derivative();
}

typedef ARealDirect<double> ADD;
typedef ARealDirect<float> ADF;

}  // namespace xad

/*******************************************************************************

   Unary expressions.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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

#include <XAD/Expression.hpp>
#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>


namespace xad
{
namespace detail
{
template <bool>
struct UnaryDerivativeImpl
{
    template <class Op, class Scalar>
    XAD_INLINE static Scalar derivative(const Op& op, const Scalar& a, const Scalar&)
    {
        return op.derivative(a);
    }
};

template <>
struct UnaryDerivativeImpl<true>
{
    template <class Op, class Scalar>
    XAD_INLINE static Scalar derivative(const Op& op, const Scalar& a, const Scalar& v)
    {
        return op.derivative(a, v);
    }
};
}  // namespace detail

template <class, class>
struct Expression;

/// Base class of all unary expressions
template <class Scalar, class Op, class Expr>
struct UnaryExpr : Expression<Scalar, UnaryExpr<Scalar, Op, Expr> >
{
    typedef detail::UnaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    XAD_INLINE explicit UnaryExpr(Expr a, Op op = Op()) : a_(a), op_(op), v_(op_(a_.value())) {}
    XAD_INLINE Scalar value() const { return v_; }
    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape& s, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(s, mul * der_impl::template derivative(op_, value(a_), v_));
    }
    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape& s) const
    {
        using xad::value;
        a_.calc_derivatives(s, der_impl::template derivative(op_, value(a_), v_));
    }

    template <typename Slot>
    XAD_INLINE void calc_derivatives(Slot* slot, Scalar* muls, int& n, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(slot, muls, n, mul * der_impl::template derivative(op_, value(a_), v_));
    }
    template <typename It1, typename It2>
    XAD_INLINE void calc_derivatives(It1& sit, It2& mit, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(sit, mit, mul * der_impl::template derivative(op_, value(a_), v_));
    }

    XAD_INLINE bool shouldRecord() const { return a_.shouldRecord(); }

    XAD_INLINE Scalar derivative() const
    {
        using xad::derivative;
        using xad::value;
        return der_impl::template derivative(op_, value(a_), v_) * derivative(a_);
    }

  private:
    Expr a_;
    Op op_;
    Scalar v_;
};

template <class Scalar, class Op, class Expr>
struct ExprTraits<UnaryExpr<Scalar, Op, Expr> >
{
    static const bool isExpr = true;
    static const int numVariables = ExprTraits<Expr>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction = ExprTraits<typename ExprTraits<Expr>::value_type>::direction;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename ExprTraits<Expr>::value_type value_type;
};
}  // namespace xad

/*******************************************************************************

   Binary expression template.

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

#include <XAD/BinaryDerivativeImpl.hpp>
#include <XAD/Expression.hpp>
#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

namespace xad
{

template <class Scalar, class Op, class Expr1, class Expr2>
struct BinaryExpr : Expression<Scalar, BinaryExpr<Scalar, Op, Expr1, Expr2> >
{
    typedef detail::BinaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    XAD_INLINE BinaryExpr(const Expr1& a, const Expr2& b, Op op = Op())
        : a_(a), b_(b), op_(op), v_(op_(a_.value(), b_.value()))
    {
    }
    XAD_INLINE Scalar value() const { return v_; }

    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape& s, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(s,
                            mul * der_impl::template derivative_a(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(s,
                            mul * der_impl::template derivative_b(op_, value(a_), value(b_), v_));
    }
    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape& s) const
    {
        using xad::value;
        a_.calc_derivatives(s, der_impl::template derivative_a(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(s, der_impl::template derivative_b(op_, value(a_), value(b_), v_));
    }

    template <typename Slot>
    XAD_INLINE void calc_derivatives(Slot* slot, Scalar* muls, int& n, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(slot, muls, n,
                            mul * der_impl::template derivative_a(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(slot, muls, n,
                            mul * der_impl::template derivative_b(op_, value(a_), value(b_), v_));
    }

    template <typename It1, typename It2>
    XAD_INLINE void calc_derivatives(It1& sit, It2& mit, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(sit, mit,
                            mul * der_impl::template derivative_a(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(sit, mit,
                            mul * der_impl::template derivative_b(op_, value(a_), value(b_), v_));
    }

    XAD_INLINE Scalar derivative() const
    {
        using xad::derivative;
        using xad::value;
        return der_impl::template derivative_a(op_, value(a_), value(b_), v_) * derivative(a_) +
               der_impl::template derivative_b(op_, value(a_), value(b_), v_) * derivative(b_);
    }

    XAD_INLINE bool shouldRecord() const { return a_.shouldRecord() || b_.shouldRecord(); }

  private:
    Expr1 a_;
    Expr2 b_;
    Op op_;
    Scalar v_;
};

template <class Scalar, class Op, class Expr1, class Expr2>
struct ExprTraits<BinaryExpr<Scalar, Op, Expr1, Expr2> >
{
    static const bool isExpr = true;
    static const int numVariables =
        ExprTraits<Expr1>::numVariables + ExprTraits<Expr2>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr1>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr1>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction =
        ExprTraits<typename ExprTraits<Expr1>::value_type>::direction;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename ExprTraits<Expr1>::value_type value_type;
    typedef Scalar scalar_type;
    // make sure that both sides of the binary expression have the same value_type
    // This should always be the case, as expressions with scalars are producing UnaryExpr
    // objects, and mixing different AReal/FReal types in a single expression is invalid
    static_assert(std::is_same<typename ExprTraits<Expr1>::value_type,
                               typename ExprTraits<Expr2>::value_type>::value,
                  "both expressions must be the same underlying type");
};
}  // namespace xad

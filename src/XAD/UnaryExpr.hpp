/*******************************************************************************

   Unary expressions.

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

#include <XAD/Config.hpp>
#include <XAD/Expression.hpp>
#ifdef XAD_ENABLE_JIT
#include <XAD/JITExprTraits.hpp>
#endif
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

template <class, class, class>
struct Expression;

/// Base class of all unary expressions
template <class Scalar, class Op, class Expr, class DerivativeType = Scalar>
struct UnaryExpr : Expression<Scalar, UnaryExpr<Scalar, Op, Expr, DerivativeType>, DerivativeType>
{
    typedef detail::UnaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    XAD_INLINE explicit UnaryExpr(const Expr& a, Op op = Op()) : a_(a), op_(op), v_(op_(a_.value()))
    {
    }
    XAD_INLINE Scalar value() const { return v_; }
    template <class Tape, int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s, const Scalar& mul) const
    {
        using xad::value;
        a_.calc_derivatives(info, s, mul * der_impl::template derivative<>(op_, a_.value(), v_));
    }
    template <class Tape, int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s) const
    {
        using xad::value;
        a_.calc_derivatives(info, s, der_impl::template derivative<>(op_, value(a_), v_));
    }

    XAD_INLINE bool shouldRecord() const { return a_.shouldRecord(); }

#ifdef XAD_ENABLE_JIT
    uint32_t recordJIT(JITGraph& graph) const
    {
        // Check ldexp first, then scalar constant, then simple unary
        return recordJITDispatch(graph);
    }

  private:
    uint32_t recordJITDispatch(JITGraph& graph) const
    {
        if (IsLdexpOp<Op>::value)
            return recordJITLdexp(graph);
        else if (HasScalarConstant<Op>::value)
            return recordJITScalar(graph);
        else
            return recordJITSimple(graph);
    }

    uint32_t recordJITSimple(JITGraph& graph) const
    {
        uint32_t slotA = a_.recordJIT(graph);
        constexpr JITOpCode opcode = JITOpCodeFor<Op>::value;
        return graph.addNode(opcode, slotA);
    }

    uint32_t recordJITScalar(JITGraph& graph) const
    {
        uint32_t slotA = a_.recordJIT(graph);
        uint32_t slotB = recordJITConstant(graph, getScalarConstant(op_));
        constexpr JITOpCode opcode = JITOpCodeFor<Op>::value;
        // For scalar_sub1, scalar_div1, scalar_pow1: scalar is first operand
        if (IsScalarFirstOp<Op>::value)
            return graph.addNode(opcode, slotB, slotA);
        else
            return graph.addNode(opcode, slotA, slotB);
    }

    uint32_t recordJITLdexp(JITGraph& graph) const
    {
        uint32_t slotA = a_.recordJIT(graph);
        constexpr JITOpCode opcode = JITOpCodeFor<Op>::value;
        double exp_val = getLdexpExponent(op_);
        return graph.addNode(opcode, slotA, 0, 0, exp_val);  // Store exponent in immediate
    }

  public:
#endif
    XAD_INLINE DerivativeType derivative() const
    {
        using xad::derivative;
        using xad::value;
        return der_impl::template derivative<>(op_, value(a_), v_) * derivative(a_);
    }

  private:
    Expr a_;
    Op op_;
    Scalar v_;
};

template <class Scalar, class Op, class Expr, class DerivativeType>
struct ExprTraits<UnaryExpr<Scalar, Op, Expr, DerivativeType> >
{
    static const bool isExpr = true;
    static const int numVariables = ExprTraits<Expr>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction = ExprTraits<typename ExprTraits<Expr>::value_type>::direction;
    static const std::size_t vector_size =
        ExprTraits<typename ExprTraits<Expr>::value_type>::vector_size;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename ExprTraits<Expr>::value_type value_type;
};
}  // namespace xad

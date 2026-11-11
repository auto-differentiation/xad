/*******************************************************************************

   Binary expression template.

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

#include <XAD/Config.hpp>
#include <XAD/BinaryDerivativeImpl.hpp>
#include <XAD/Expression.hpp>
#ifdef XAD_ENABLE_JIT
#include <XAD/JITExprTraits.hpp>
#endif
#include <XAD/Macros.hpp>
#include <XAD/Traits.hpp>
#include <type_traits>

namespace xad
{

// A partial of a constant zero operand inlined here is infinite by design (a zero tangent masks
// it); MSVC reports that as C4723, an error with /WX.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4723)
#endif
template <class Scalar, class Op, class Expr1, class Expr2, class DerivativeType = Scalar>
struct BinaryExpr
    : Expression<Scalar, BinaryExpr<Scalar, Op, Expr1, Expr2, DerivativeType>, DerivativeType>
{
    typedef detail::BinaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    // Eigen's numext math functions (numext::sqrt, ...) evaluate this expression as its active type
    typedef typename ExprTraits<Expr1>::value_type
        Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl;
    XAD_INLINE BinaryExpr(const Expr1& a, const Expr2& b, Op op = Op())
        : a_(a), b_(b), op_(op), v_(op_(a_.value(), b_.value()))
    {
    }
#ifdef XAD_BINARY_EXPR_COPY_MEMBERWISE
    XAD_FORCE_INLINE BinaryExpr(const BinaryExpr& o) : a_(o.a_), b_(o.b_), op_(o.op_), v_(o.v_) {}
#endif
    XAD_INLINE Scalar value() const { return v_; }

    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s, const Scalar& mul) const
    {
        calcA(info, s, mul, detail::UnitPartialA<Op>());
        calcB(info, s, mul, detail::UnitPartialB<Op>());
    }
    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s) const
    {
        calcA(info, s, detail::unit_multiplier(), detail::UnitPartialA<Op>());
        calcB(info, s, detail::unit_multiplier(), detail::UnitPartialB<Op>());
    }
    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s, detail::minus_unit m) const
    {
        calcA(info, s, m, detail::UnitPartialA<Op>());
        calcB(info, s, m, detail::UnitPartialB<Op>());
    }

    XAD_INLINE DerivativeType derivative() const
    {
        return derivative(std::integral_constant<bool, detail::HasTangentRule<Op>::value>());
    }

    XAD_INLINE bool shouldRecord() const { return a_.shouldRecord() || b_.shouldRecord(); }

#ifdef XAD_ENABLE_JIT
    bool shouldRecordJIT() const { return a_.shouldRecordJIT() || b_.shouldRecordJIT(); }

    uint32_t recordJIT(JITGraph& graph) const
    {
        static_assert(static_cast<uint16_t>(JITOpCodeFor<Op>::value) != 0xFFFF,
                      "JIT opcode mapping missing for binary operator");
        uint32_t slotA = a_.recordJIT(graph);
        uint32_t slotB = b_.recordJIT(graph);
        return recordJITBinaryNode<Op>(graph, slotA, slotB);
    }
#endif

  private:
    // an operand's partials with the incoming multiplier m: m passed on (or its sign flipped) where
    // the op's partial is +1 (-1), else m times the partial
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcA(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, 1>) const
    {
        detail::calcWith(a_, info, s, m);
    }
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcA(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, -1>) const
    {
        detail::calcWith(a_, info, s, detail::flipSign(m));
    }
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcA(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, 0>) const
    {
        using xad::value;
        detail::calcWith(
            a_, info, s,
            detail::times(m, der_impl::template derivative_a<>(op_, value(a_), value(b_), v_)));
    }
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcB(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, 1>) const
    {
        detail::calcWith(b_, info, s, m);
    }
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcB(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, -1>) const
    {
        detail::calcWith(b_, info, s, detail::flipSign(m));
    }
    template <class Info, class Tape, class M>
    XAD_FORCE_INLINE void calcB(Info& info, Tape& s, const M& m,
                                std::integral_constant<int, 0>) const
    {
        using xad::value;
        detail::calcWith(
            b_, info, s,
            detail::times(m, der_impl::template derivative_b<>(op_, value(a_), value(b_), v_)));
    }

    XAD_INLINE DerivativeType derivative(std::true_type) const
    {
        using xad::derivative;
        using xad::value;
        return op_.tangent(value(a_), value(b_), v_, derivative(a_), derivative(b_));
    }

    XAD_INLINE DerivativeType derivative(std::false_type) const
    {
        using xad::derivative;
        using xad::value;
        const Scalar& pa = der_impl::template derivative_a<>(op_, value(a_), value(b_), v_);
        const Scalar& pb = der_impl::template derivative_b<>(op_, value(a_), value(b_), v_);
        const DerivativeType& ta = derivative(a_);
        const DerivativeType& tb = derivative(b_);
#ifdef XAD_BRANCH_FREE_TANGENTS
        return detail::tangentTerm(pa, ta, detail::HasFinitePartials<Op>()) +
               detail::tangentTerm(pb, tb, detail::HasFinitePartials<Op>());
#else
        DerivativeType d = pa * ta + pb * tb;
        if (XAD_VERY_UNLIKELY(detail::isNaNTangent(d, detail::HasFinitePartials<Op>())))
            d = detail::tangentTerm(pa, ta) + detail::tangentTerm(pb, tb);
        return d;
#endif
    }

    typename ExprStorage<Expr1>::type a_;
    typename ExprStorage<Expr2>::type b_;
    Op op_;
    Scalar v_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <class Scalar, class Op, class Expr1, class Expr2, class DerivativeType>
struct ExprTraits<BinaryExpr<Scalar, Op, Expr1, Expr2, DerivativeType>>
{
    static const bool isExpr = true;
    static const int numVariables =
        ExprTraits<Expr1>::numVariables + ExprTraits<Expr2>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr1>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr1>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction =
        ExprTraits<typename ExprTraits<Expr1>::value_type>::direction;
    static const std::size_t vector_size =
        ExprTraits<typename ExprTraits<Expr1>::value_type>::vector_size;

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

namespace Eigen
{
template <typename T>
struct NumTraits;

// Eigen's numext functions (numext::abs, ...) take their result type from NumTraits: that of the
// active type, not of the expression
template <class Scalar, class Op, class Expr1, class Expr2, class DerivativeType>
struct NumTraits<xad::BinaryExpr<Scalar, Op, Expr1, Expr2, DerivativeType>>
    : NumTraits<typename xad::ExprTraits<Expr1>::value_type>
{
};
}  // namespace Eigen

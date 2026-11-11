/*******************************************************************************

   Literal AD types for all modes.

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
#include <XAD/Expression.hpp>
#ifdef XAD_ENABLE_JIT
#include <XAD/JITCompiler.hpp>
#include <XAD/JITExprTraits.hpp>
#endif
#include <XAD/Macros.hpp>
#include <XAD/Tape.hpp>
#include <XAD/Traits.hpp>

#include <XAD/Vec.hpp>
#include <algorithm>
#include <iosfwd>
#include <utility>

namespace xad
{
template <class, std::size_t>
class Tape;
template <class, std::size_t>
struct FReal;

template <class Scalar, std::size_t N>
struct FRealTraits
{
    using type = FReal<Scalar, N>;
    using derivative_type = Vec<Scalar, N>;
};

template <class Scalar>
struct FRealTraits<Scalar, 1>
{
    using type = FReal<Scalar, 1>;
    using derivative_type = Scalar;
};

template <class Scalar, class Derived, class DerivativeType = Scalar>
struct ADTypeBase : public Expression<Scalar, Derived, DerivativeType>
{
    typedef typename ExprTraits<Derived>::value_type value_type;
    typedef typename ExprTraits<Derived>::nested_type nested_type;

    static_assert(std::is_floating_point<nested_type>::value,
                  "Active AD types only work with floating point");

    constexpr explicit XAD_INLINE ADTypeBase(Scalar val = Scalar()) : a_(val) {}
    constexpr XAD_INLINE ADTypeBase(ADTypeBase&& o) noexcept = default;
    constexpr XAD_INLINE ADTypeBase(const ADTypeBase& o) = default;
    XAD_INLINE ADTypeBase& operator=(ADTypeBase&& o) noexcept = default;
    XAD_INLINE ADTypeBase& operator=(const ADTypeBase& o) = default;
    XAD_INLINE ~ADTypeBase() = default;

    constexpr XAD_INLINE Scalar getValue() const { return value(); }
    XAD_INLINE const Scalar& value() const { return a_; }
    XAD_INLINE Scalar& value() { return a_; }

    template <class E>
    XAD_INLINE Derived& operator+=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() + x);
    }
    template <class E>
    XAD_INLINE Derived& operator-=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() - x);
    }
    template <class E>
    XAD_INLINE Derived& operator*=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() * x);
    }
    template <class E>
    XAD_INLINE Derived& operator/=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() / x);
    }

    // ints and passive floats, e.g. a double for AReal<FReal<double>>, where Scalar and value_type
    // would both need a conversion
    template <class I>
    using passive_result =
        typename std::enable_if<std::is_arithmetic<I>::value && !ExprTraits<I>::isExpr,
                                Derived>::type;

    XAD_INLINE Derived& operator+=(Scalar x)
    {
        a_ += x;
        return derived();
    }
    template <class I>
    XAD_INLINE passive_result<I>& operator+=(I x)
    {
        return derived() += Scalar(x);
    }
    XAD_INLINE Derived& operator-=(Scalar rhs)
    {
        a_ -= rhs;
        return derived();
    }
    template <class I>
    XAD_INLINE passive_result<I>& operator-=(I x)
    {
        return derived() -= Scalar(x);
    }
    XAD_INLINE Derived& operator*=(Scalar x) { return derived() = (derived() * x); }
    template <class I>
    XAD_INLINE passive_result<I>& operator*=(I x)
    {
        return derived() = (derived() * nested_type(x));
    }
    XAD_INLINE Derived& operator/=(Scalar x) { return derived() = (derived() / x); }
    template <class I>
    XAD_INLINE passive_result<I>& operator/=(I x)
    {
        return derived() = (derived() / nested_type(x));
    }
    XAD_INLINE Derived& operator+=(const value_type& x) { return derived() = derived() + x; }
    XAD_INLINE Derived& operator-=(const value_type& x) { return derived() = derived() - x; }
    XAD_INLINE Derived& operator*=(const value_type& x) { return derived() = derived() * x; }
    XAD_INLINE Derived& operator/=(const value_type& x) { return derived() = derived() / x; }
    XAD_INLINE Derived& operator++() { return derived() = (derived() + Scalar(1)); }
    XAD_INLINE Derived operator++(int)
    {
        auto tmp = derived();
        derived() = (derived() + Scalar(1));
        return tmp;
    }
    XAD_INLINE Derived& operator--() { return derived() = (derived() - Scalar(1)); }
    XAD_INLINE Derived operator--(int)
    {
        auto tmp = derived();
        derived() = (derived() - Scalar(1));
        return tmp;
    }

  private:
    XAD_INLINE Derived& derived() { return static_cast<Derived&>(*this); }
    XAD_INLINE const Derived& derived() const { return static_cast<const Derived&>(*this); }

  protected:
    Scalar a_;
};

template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct ADVar;

template <class Scalar, std::size_t M>
struct ExprTraits<AReal<Scalar, M>>
{
    static const bool isExpr = true;
    static const int numVariables = 1;
    static const bool isForward = false;
    static const bool isReverse = true;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_REVERSE;
    static const std::size_t vector_size = M;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef AReal<Scalar, M> value_type;
    typedef Scalar scalar_type;
};

template <class Scalar, std::size_t M>
struct ExprTraits<ADVar<Scalar, M>> : public ExprTraits<AReal<Scalar, M>>
{
};

namespace detail
{
struct areal_assign;
}

template <class Scalar, std::size_t N = 1>
struct AReal
    : public ADTypeBase<Scalar, AReal<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
{
    typedef Tape<Scalar, N> tape_type;
    typedef ADTypeBase<Scalar, AReal<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
        base_type;
    typedef typename tape_type::slot_type slot_type;
    typedef Scalar value_type;
    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename DerivativesTraits<Scalar, N>::type derivative_type;
#ifdef XAD_ENABLE_JIT
    typedef JITCompiler<nested_type, N> jit_type;
#endif

    XAD_INLINE AReal(nested_type val = nested_type()) : base_type(val), slot_(INVALID_SLOT) {}

    // explicit conversion from int (also used by static_cast) to avoid warnings
    template <class U>
    explicit AReal(U val, typename std::enable_if<std::is_integral<U>::value>::type* = 0)
        : base_type(static_cast<nested_type>(val)), slot_(INVALID_SLOT)
    {
    }

    XAD_RECORDING_INLINE AReal(const AReal& o) : base_type(), slot_(INVALID_SLOT)
    {
        if (auto* s = tape_type::getActive())
        {
#ifdef XAD_LINEAR_SLOTS
            // linear slots: a slot is written by one statement only, so a copy shares it
            XAD_UNUSED_VARIABLE(s);
            slot_ = o.slot_;
            copyTape(o);
#elif defined(XAD_COPY_ELISION)
            if (o.isOn(s))
                shareSlotOf(s, o);  // no statement: a statement writing either detaches it first
#ifdef XAD_TAPE_CHECKS
            else if (o.shouldRecord())  // another tape's variable: a constant here
            {
                registerOn(s);
                s->pushLhs(slot_);
            }
#endif
#else
            if (o.shouldRecord())
            {
                registerOn(s);
                // read before the partial is written: the tape's stores may alias slot_
                const slot_type slot = slot_;
                pushAll<1>(s, o);
                s->pushLhs(slot);
            }
#endif
        }
#ifdef XAD_ENABLE_JIT
        else if (getActiveJit() != nullptr)
        {
            if (o.shouldRecordJIT())
            {
                // Copy the slot directly - preserves JIT dependency chain
                slot_ = o.slot_;
            }
        }
#endif
        this->a_ = o.getValue();
    }

    static constexpr slot_type INVALID_SLOT = tape_type::INVALID_SLOT;
#ifdef XAD_COPY_ELISION
    XAD_INLINE slot_type getSlot() const
    {
        if (XAD_UNLIKELY(shouldRecord() && (slot_ & SHARED_BIT)))
            return tapeSlot();
        return slot_;
    }
#else
    XAD_INLINE slot_type getSlot() const { return slot_; }
#endif
    XAD_INLINE tape_type* getTape() const { return tape_type::getActive(); }

    XAD_INLINE AReal(AReal&& o) noexcept : base_type(static_cast<base_type&&>(o)), slot_(o.slot_)
    {
        o.slot_ = INVALID_SLOT;
        swapTape(o);
    }

    XAD_INLINE AReal& operator=(AReal&& o) noexcept
    {
        static_cast<base_type&>(*this) = static_cast<base_type&&>(o);
        // object moved from still gets destructor called, so this makes sure that
        // the old slot we had gets destructed
        std::swap(slot_, o.slot_);
        swapTape(o);
        return *this;
    }

#ifdef XAD_COPY_ELISION
    // forced: gcc stopped inlining it when it got the shared-slot branch, costing a call per
    // passive temporary
    XAD_FORCE_INLINE ~AReal()
    {
        if (auto tape = tape_type::getActive())
        {
            if (XAD_LIKELY(ownsSlot(tape)))
                tape->unregisterVariable(slot_);
            else if (XAD_UNLIKELY(slot_ != INVALID_SLOT))  // shared, another tape's or JIT
                releaseOther(tape);
        }
    }
#else
    XAD_INLINE ~AReal()
    {
#ifndef XAD_LINEAR_SLOTS  // linear slots: a copy may still use the slot, so it is never freed
        if (auto tape = tape_type::getActive())
            if (isOn(tape))
                tape->unregisterVariable(slot_);
#endif
    }
#endif

    XAD_FORCE_INLINE AReal& operator=(const AReal& o);

    XAD_INLINE AReal& operator=(nested_type x)
    {
        this->a_ = x;
        auto tape = tape_type::getActive();
        if (tape && isOn(tape))
#ifdef XAD_LINEAR_SLOTS
        {
            // linear slots: copies may share the old slot, so the constant gets a new one;
            // the variable stays active, so an input registered before stays an input
            registerOn(tape);
            tape->pushLhs(slot_);
        }
#elif defined(XAD_COPY_ELISION)
        {
            if (!ownsSlot(tape))
                claimSlot(tape);  // copies may share the slot
            tape->pushLhs(slot_);
        }
#else
            tape->pushLhs(slot_);
#endif
#ifdef XAD_ENABLE_JIT
        else
            dropJitNode(std::integral_constant<bool, jit_supported>());
#endif
        return *this;
    }

    template <class Expr>
    XAD_FORCE_INLINE AReal(const Expression<Scalar, Expr, derivative_type>&
                               expr);  // cppcheck-suppress noExplicitConstructor

    template <class Expr>
    XAD_INLINE AReal& operator=(const Expression<Scalar, Expr, derivative_type>& expr)
    {
        return assignExpression(expr);
    }

  private:
    friend struct detail::areal_assign;

    // operator= from an expression, always inlined where std::complex<AReal> assigns its parts
    template <class Expr>
    XAD_FORCE_INLINE AReal& assignExpression(const Expression<Scalar, Expr, derivative_type>& expr);

  public:
#ifdef XAD_ENABLE_JIT
    using base_type::operator+=;
    using base_type::operator-=;

    XAD_INLINE AReal& operator+=(Scalar x)
    {
        return addScalar(x, std::integral_constant<bool, jit_supported>());
    }
    XAD_INLINE AReal& operator-=(Scalar x)
    {
        return subScalar(x, std::integral_constant<bool, jit_supported>());
    }
#endif

    XAD_INLINE void setDerivative(derivative_type a) { derivative() = a; }
    XAD_INLINE void setAdjoint(derivative_type a) { setDerivative(a); }
    XAD_INLINE derivative_type getAdjoint() const { return getDerivative(); }

#ifdef XAD_ENABLE_JIT
  private:
    // JIT is intentionally limited to scalar, first-order mode only:
    // - vector mode (N>1) is not supported
    // - higher-order AD (Scalar != nested_type) is not supported
    enum
    {
        jit_supported = std::is_same<Scalar, nested_type>::value && (N == 1)
    };

    static XAD_INLINE jit_type* getActiveJitImpl(std::true_type) { return jit_type::getActive(); }
    static XAD_INLINE jit_type* getActiveJitImpl(std::false_type) { return nullptr; }
    static XAD_INLINE jit_type* getActiveJit()
    {
        return getActiveJitImpl(std::integral_constant<bool, jit_supported>());
    }

    // Scalar shifts and assignments change the graph, but not on a registered input: forward()
    // reads its value, so they set the value to replay. A copy sharing the input's node is not
    // read by forward(), so they change its graph.
    XAD_INLINE bool jitRecordsScalarOp() const
    {
        jit_type* j = jit_type::getActive();
        if (!j || !shouldRecordJIT() || tape_type::getActive())
            return false;
        const JITGraph& g = j->getGraph();
        const uint32_t node = jitNode(slot_);
        return node >= g.nodeCount() || !g.isInput(node) || !j->isRegisteredInput(*this, node);
    }

    // a shift needs no tape entry, but the JIT graph must record it
    XAD_INLINE AReal& addScalar(Scalar x, std::true_type)
    {
        return jitRecordsScalarOp() ? (*this = *this + x) : base_type::operator+=(x);
    }
    XAD_INLINE AReal& addScalar(Scalar x, std::false_type) { return base_type::operator+=(x); }
    XAD_INLINE AReal& subScalar(Scalar x, std::true_type)
    {
        return jitRecordsScalarOp() ? (*this = *this - x) : base_type::operator-=(x);
    }
    XAD_INLINE AReal& subScalar(Scalar x, std::false_type) { return base_type::operator-=(x); }

    // a constant now, as in the JIT's copy assignment
    XAD_INLINE void dropJitNode(std::true_type)
    {
        if (jitRecordsScalarOp())
            slot_ = INVALID_SLOT;
    }
    XAD_INLINE void dropJitNode(std::false_type) {}

    // out of line: inlined into the force-inlined constructor, it slowed down JIT recording
    template <class Expr>
    XAD_NEVER_INLINE void tryRecordJitFromExprCtor(
        const Expression<Scalar, Expr, derivative_type>& expr, std::true_type)
    {
        if (auto* j = jit_type::getActive())
        {
            const Expr& e = static_cast<const Expr&>(expr);
            if (e.shouldRecordJIT())
                slot_ = jitSlot(e.recordJIT(j->getGraph()));
        }
    }

    template <class Expr>
    XAD_INLINE void tryRecordJitFromExprCtor(const Expression<Scalar, Expr, derivative_type>&,
                                             std::false_type)
    {
    }

    template <class Expr>
    XAD_INLINE void tryRecordJitFromExprAssign(const Expression<Scalar, Expr, derivative_type>& expr,
                                               std::true_type)
    {
        if (auto* j = jit_type::getActive())
        {
            const Expr& e = static_cast<const Expr&>(expr);
            if (e.shouldRecordJIT() || this->shouldRecordJIT())
                slot_ = jitSlot(e.recordJIT(j->getGraph()));
        }
    }

    template <class Expr>
    XAD_INLINE void tryRecordJitFromExprAssign(const Expression<Scalar, Expr, derivative_type>&,
                                               std::false_type)
    {
    }

  private:
    // Only enable the JIT derivative path when it is type-correct.
    // For higher-order AD (Scalar != nested_type) we must not even instantiate code that
    // would return JITCompiler<nested_type,N>::derivative_type as derivative_type.
    XAD_INLINE const derivative_type* tryGetJitDerivativePtr(std::true_type) const
    {
        auto j = getActiveJit();
        if (!j)
            return nullptr;
        if (!shouldRecordJIT())
        {
            static const derivative_type zero = derivative_type();
            return &zero;
        }
        return &j->derivative(slot_);
    }

    XAD_INLINE const derivative_type* tryGetJitDerivativePtr(std::false_type) const
    {
        return nullptr;
    }

    XAD_INLINE derivative_type* tryGetJitDerivativePtr(std::true_type)
    {
        auto j = getActiveJit();
        if (!j)
            return nullptr;
        if (shouldRecord())
        {
            // a tape variable is a constant here: its adjoint reads 0 and ignores writes
            static XAD_THREAD_LOCAL derivative_type ignored;
            ignored = derivative_type();
            return &ignored;
        }
        // a constant node: a slot no later node takes
        if (!shouldRecordJIT())
            slot_ = jitSlot(recordJIT(j->getGraph()));
        return &j->derivative(slot_);
    }

    XAD_INLINE derivative_type* tryGetJitDerivativePtr(std::false_type)
    {
        return nullptr;
    }

  public:
#endif

    template <class Info>
    XAD_FORCE_INLINE void pushRhs(Info& info, const Scalar& mul, slot_type slot) const
    {
        info.add(mul, slot);
    }

    template <class Info>
    XAD_FORCE_INLINE void calc_derivatives(Info& info, tape_type& s, const Scalar& mul) const
    {
        if (isOn(&s))
#ifdef XAD_COPY_ELISION
            pushRhs(info, mul, tapeSlot());
#else
            pushRhs(info, mul, slot_);
#endif
    }

    template <class Info>
    XAD_FORCE_INLINE void calc_derivatives(Info& info, tape_type& s) const
    {
        if (isOn(&s))
#ifdef XAD_COPY_ELISION
            pushRhs(info, Scalar(1), tapeSlot());
#else
            pushRhs(info, Scalar(1), slot_);
#endif
    }

    XAD_INLINE derivative_type getDerivative() const { return derivative(); }

    XAD_INLINE const derivative_type& derivative() const
    {
        auto t = tape_type::getActive();
        if (!t)
        {
#ifdef XAD_ENABLE_JIT
            if (const derivative_type* d =
                    tryGetJitDerivativePtr(std::integral_constant<bool, jit_supported>()))
                return *d;
#endif
            throw NoTapeException();
        }
        if (!isOn(t))
        {
            // we return a dummy const ref if not registered on tape - always zero
            static const derivative_type zero = derivative_type();
            return zero;
        }
#ifdef XAD_COPY_ELISION
        return t->derivative(tapeSlot());
#else
        return t->derivative(slot_);
#endif
    }

    XAD_INLINE derivative_type& derivative()
    {
        auto t = tape_type::getActive();
        if (!t)
        {
#ifdef XAD_ENABLE_JIT
            if (derivative_type* d =
                    tryGetJitDerivativePtr(std::integral_constant<bool, jit_supported>()))
                return *d;
#endif
            throw NoTapeException();
        }
        // register ourselves if not already done
        if (!isOn(t))
        {
            registerOn(t);
            t->pushLhs(slot_);
        }
#ifdef XAD_COPY_ELISION
        return t->derivative(tapeSlot());
#else
        return t->derivative(slot_);
#endif
    }
#ifdef XAD_ENABLE_JIT
    // JIT variables are passive on tapes; the hint keeps GCC's code layout of an INVALID_SLOT check
    XAD_INLINE bool shouldRecord() const { return XAD_LIKELY(slot_ < FIRST_JIT_SLOT); }
    XAD_INLINE bool shouldRecordJIT() const { return isJitSlot(slot_); }
#else
    XAD_INLINE bool shouldRecord() const { return slot_ != INVALID_SLOT; }
#endif

#ifdef XAD_ENABLE_JIT
    uint32_t recordJIT(JITGraph& graph) const
    {
        if (shouldRecordJIT())
            return jitNode(slot_);
        // Not registered - treat as constant (handles nested AD types)
        return recordJITConstant(graph, getNestedDoubleValue(this->a_));
    }
#endif

  private:
    XAD_RECORDING_OUTLINE void recordCopy(tape_type* s, const AReal& o);

#ifndef XAD_SKIP_ZERO_PARTIALS
    // Stage the partials in stack arrays, then append them to the tape in one
    // batch. The arrays are locals, so the compiler can keep them in registers.
    template <int Size, typename Expr>
    XAD_FORCE_INLINE void pushStaged(tape_type* t, const Expr& expr) const
    {
        DerivInfo<tape_type, Size> info;
        expr.calc_derivatives(info, *t);
        t->pushAll(info.multipliers, info.slots, info.index);
    }

    // Write the partials into the tape as they are computed, saving the copy.
    // Falls back to staging when the run would cross a chunk boundary, and for
    // value types that are not trivially copyable.
    template <int Size, typename Expr>
    XAD_FORCE_INLINE void pushAll(tape_type* t, const Expr& expr) const
    {
        if (!std::is_trivially_copyable<typename tape_type::value_type>::value)
        {
            pushStaged<Size>(t, expr);
            return;
        }
#ifdef XAD_REDUCED_MEMORY
        auto run = t->tryReserveOperationRun(Size);
        if (XAD_VERY_UNLIKELY(run.multipliers == nullptr))
        {
            pushStaged<Size>(t, expr);
            return;
        }
        DerivInfoDirect<tape_type> info;
        info.multipliers = run.multipliers;
        info.slots = run.slots;
        info.pos = run.pos;
        expr.calc_derivatives(info, *t);
        t->commitOperationRun(info.pos);
#else
        DerivInfoDirect<tape_type> info;
        info.dst = t->tryReserveOperations(Size);
        if (XAD_VERY_UNLIKELY(info.dst == nullptr))
        {
            pushStaged<Size>(t, expr);
            return;
        }
        expr.calc_derivatives(info, *t);
        t->commitOperations(info.index);
#endif
    }
#else
    // as above; false if there were partials and all were zero, which are not recorded
    template <int Size, typename Expr>
    XAD_FORCE_INLINE bool pushStaged(tape_type* t, const Expr& expr) const
    {
        DerivInfo<tape_type, Size> info;
        expr.calc_derivatives(info, *t);
        bool keep = info.dropZeros();
        t->pushAll(info.multipliers, info.slots, info.index);
        return keep;
    }

#ifndef XAD_REDUCED_MEMORY
    using staged_pair = typename DerivInfoDirect<tape_type>::pair_type;
#endif

#if defined(__GNUC__) && !defined(__clang__)
// GCC cannot see that the partials written through info fill the stack buffer(s)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    // writes all partials and drops the zeros afterwards, with one test; at a chunk boundary
    // they go to the stack and are appended out of line
    template <int Size, typename Expr>
    XAD_FORCE_INLINE bool pushAll(tape_type* t, const Expr& expr) const
    {
        if (!std::is_trivially_copyable<typename tape_type::value_type>::value)
            return pushStaged<Size>(t, expr);
#ifdef XAD_REDUCED_MEMORY
        using mul_type = typename tape_type::value_type;
        alignas(mul_type) unsigned char stagedMuls[Size * sizeof(mul_type)];
        alignas(slot_type) unsigned char stagedSlots[Size * sizeof(slot_type)];
        auto run = t->tryReserveOperationRun(Size);
        const bool direct = run.multipliers != nullptr;
        if (XAD_VERY_UNLIKELY(!direct))
        {
            run.multipliers = reinterpret_cast<mul_type*>(stagedMuls);
            run.slots = reinterpret_cast<slot_type*>(stagedSlots);
            run.pos = 0;
        }
        DerivInfoDirect<tape_type> info;
        info.multipliers = run.multipliers;
        info.slots = run.slots;
        info.pos = info.start = run.pos;
        expr.calc_derivatives(info, *t);
        bool keep = info.template dropZeros<Size>();
        if (XAD_VERY_LIKELY(direct))
            t->commitOperationRun(info.pos);
        else
            appendStaged(t, info.multipliers, info.slots, unsigned(info.pos));
        return keep;
#else
        alignas(staged_pair) unsigned char staged[Size * sizeof(staged_pair)];
        DerivInfoDirect<tape_type> info;
        info.dst = t->tryReserveOperations(Size);
        const bool direct = info.dst != nullptr;
        if (XAD_VERY_UNLIKELY(!direct))
            info.dst = reinterpret_cast<staged_pair*>(staged);
        expr.calc_derivatives(info, *t);
        bool keep = info.dropZeros();
        if (XAD_VERY_LIKELY(direct))
            t->commitOperations(info.index);
        else
            appendStaged(t, info.dst, info.index);
        return keep;
#endif
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef XAD_REDUCED_MEMORY
    static XAD_NEVER_INLINE XAD_COLD void appendStaged(tape_type* t,
                                                       const typename tape_type::value_type* muls,
                                                       const slot_type* slots, unsigned n)
    {
        t->pushAll(muls, slots, n);
    }
#else
    static XAD_NEVER_INLINE XAD_COLD void appendStaged(tape_type* t, const staged_pair* ops,
                                                       unsigned n)
    {
        for (unsigned i = 0; i < n; ++i) t->pushAll(&ops[i].first, &ops[i].second, 1);
    }
#endif
#endif

    XAD_INLINE void registerOn(tape_type* t)
    {
        slot_ = t->registerVariable();
        setTape(t);
    }
#ifdef XAD_COPY_ELISION
    // SHARED_BIT in slot_ marks a slot that copies may share; only marked variables read the
    // tape's share counts
    static constexpr slot_type SHARED_BIT = tape_type::SHARED_SLOT_BIT;
#ifdef XAD_ENABLE_JIT
    static_assert((SHARED_BIT | (tape_type::MAX_SLOTS - 1)) < FIRST_JIT_SLOT,
                  "a marked tape slot is no JIT slot");
#endif

    // the tape's slot (slot_ is on a tape)
    XAD_INLINE slot_type tapeSlot() const { return slot_ & ~SHARED_BIT; }

    // whether slot_ is on t and no other variable may share it
    XAD_INLINE bool ownsSlot(const tape_type* t) const
    {
#ifdef XAD_TAPE_CHECKS
        return slot_ < SHARED_BIT && tapeId_ == t->id_;
#else
        XAD_UNUSED_VARIABLE(t);
        return slot_ < SHARED_BIT;
#endif
    }

    XAD_INLINE void shareSlotOf(tape_type* t, const AReal& o)
    {
        const slot_type s = o.slot_ | SHARED_BIT;
        o.slot_ = s;
        slot_ = s;
        copyTape(o);
        t->shareSlot(s & ~SHARED_BIT);
    }

    // frees slot_ (on t) unless other variables share it
    XAD_INLINE void releaseSlot(tape_type* t)
    {
        if (XAD_UNLIKELY(slot_ & SHARED_BIT))
            t->unregisterShared(tapeSlot());
        else
            t->unregisterVariable(slot_);
    }

    // the destructor for a slot ownsSlot rejects: out of line, so that the destructor inlines
    XAD_NEVER_INLINE void releaseOther(tape_type* t)
    {
        if (isOn(t))
            t->unregisterShared(tapeSlot());
    }

    // before a statement writes slot_, when not ownsSlot: a slot of its own on t
    XAD_INLINE void claimSlot(tape_type* t)
    {
        if (!isOn(t))
            registerOn(t);
        else
            detachSlot(t);
    }

    // slot_ on t, shared: unmarked, a new slot if others still use it
    XAD_INLINE void detachSlot(tape_type* t)
    {
        slot_type slot = tapeSlot();
        t->detachShared(slot);
        slot_ = slot;
    }
#endif
#ifdef XAD_TAPE_CHECKS
    // whether slot_ is one of t's, which the slot alone cannot tell
    XAD_INLINE bool isOn(const tape_type* t) const { return shouldRecord() && tapeId_ == t->id_; }
    XAD_INLINE void setTape(const tape_type* t) { tapeId_ = t->id_; }
    XAD_INLINE void swapTape(AReal& o) { std::swap(tapeId_, o.tapeId_); }
#if defined(XAD_LINEAR_SLOTS) || defined(XAD_COPY_ELISION)
    XAD_INLINE void copyTape(const AReal& o) { tapeId_ = o.tapeId_; }
#endif
#else
    XAD_INLINE bool isOn(const tape_type*) const { return shouldRecord(); }
    XAD_INLINE void setTape(const tape_type*) {}
    XAD_INLINE void swapTape(AReal&) {}
#if defined(XAD_LINEAR_SLOTS) || defined(XAD_COPY_ELISION)
    XAD_INLINE void copyTape(const AReal&) {}
#endif
#endif

    template <class T, std::size_t d__cnt>
    friend class Tape;
#ifdef XAD_ENABLE_JIT
    template <class T, std::size_t d__cnt>
    friend class JITCompiler;
    template <class T, std::size_t d__cnt>
    friend class ABool;
#endif
#ifdef XAD_COPY_ELISION
    mutable typename tape_type::slot_type slot_;  // a copy marks the slot of its source shared
#else
    typename tape_type::slot_type slot_;
#endif
#ifdef XAD_TAPE_CHECKS
    unsigned int tapeId_ = 0;  // the tape of slot_
#endif
};

// this class wraps AReal<T> and makes sure that no new copies are created on
// the Tape
// when this guy is copied (unlike the AReal<T> copy)
// therefore we can use auto = ... in expressions
template <class Scalar, std::size_t N = 1>
struct ADVar
    : public Expression<Scalar, ADVar<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
{
    typedef AReal<Scalar, N> areal_type;
    typedef typename areal_type::tape_type tape_type;

    XAD_INLINE explicit ADVar(const areal_type& a) : ar_(a), shouldRecord_(a.shouldRecord()) {}
#ifdef XAD_EXPR_COPY_MEMBERWISE
    XAD_FORCE_INLINE ADVar(const ADVar& o) : ar_(o.ar_), shouldRecord_(o.shouldRecord_) {}
#endif

    XAD_INLINE Scalar getValue() const { return ar_.getValue(); }

    XAD_INLINE const Scalar& value() const { return ar_.value(); }

    template <class Info>
    XAD_INLINE void calc_derivatives(Info& info, tape_type& s, const Scalar& mul) const
    {
        ar_.calc_derivatives(info, s, mul);
    }

    template <class Info>
    XAD_INLINE void calc_derivatives(Info& info, tape_type& s) const
    {
        ar_.calc_derivatives(info, s);
    }

    XAD_INLINE const typename areal_type::derivative_type& derivative() const
    {
        return ar_.derivative();
    }

    XAD_INLINE bool shouldRecord() const { return shouldRecord_; }

#ifdef XAD_ENABLE_JIT
    bool shouldRecordJIT() const { return ar_.shouldRecordJIT(); }
    uint32_t recordJIT(JITGraph& graph) const { return ar_.recordJIT(graph); }
#endif

  private:
    areal_type const& ar_;
    bool shouldRecord_;
};

template <class Scalar, std::size_t M>
XAD_RECORDING_OUTLINE void AReal<Scalar, M>::recordCopy(tape_type* s, const AReal& o)
{
#ifdef XAD_COPY_ELISION
    // o is passive here (or another tape's): this variable stays active, holding a constant
    XAD_UNUSED_VARIABLE(o);
    if (!ownsSlot(s))
        claimSlot(s);
#else
    if (!isOn(s))
        registerOn(s);
    pushAll<1>(s, o);
#endif
    s->pushLhs(slot_);
}

template <class Scalar, std::size_t M>
XAD_FORCE_INLINE AReal<Scalar, M>& AReal<Scalar, M>::operator=(const AReal& o)
{
    if (auto* s = tape_type::getActive())
    {
#ifdef XAD_LINEAR_SLOTS
        if (o.shouldRecord())
        {
            // linear slots: share the slot of o, which no later statement writes
            slot_ = o.slot_;
            copyTape(o);
        }
        else if (isOn(s))
        {
            // a passive value: a new slot with no partials, so the variable stays active
            registerOn(s);
            s->pushLhs(slot_);
        }
#elif defined(XAD_COPY_ELISION)
        if (o.isOn(s))
        {
            // share the slot of o, freeing the one held before
            if ((slot_ | SHARED_BIT) != (o.slot_ | SHARED_BIT) || !isOn(s))
            {
                if (isOn(s))
                    releaseSlot(s);
                shareSlotOf(s, o);
            }
        }
        else if (o.shouldRecord() || this->shouldRecord())
            recordCopy(s, o);
#else
        if (o.shouldRecord() || this->shouldRecord())
            recordCopy(s, o);
#endif
    }
#ifdef XAD_ENABLE_JIT
    else if (getActiveJit() != nullptr)
    {
        if (o.shouldRecordJIT() || this->shouldRecordJIT())
        {
            // Copy the slot directly - preserves JIT dependency chain
            slot_ = o.shouldRecordJIT() ? o.slot_ : INVALID_SLOT;
        }
    }
#endif
    this->a_ = o.getValue();
    return *this;
}

template <class Scalar, std::size_t M>
template <class Expr>
XAD_FORCE_INLINE AReal<Scalar, M>::AReal(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
    : base_type(expr.getValue()), slot_(INVALID_SLOT)
{
    tape_type* s = tape_type::getActive();
    if (s)
    {
        if (expr.shouldRecord())
        {
#ifdef XAD_OUTLINE_RECORDING
            DerivInfo<tape_type, ExprTraits<Expr>::numVariables> info;
            expr.calc_derivatives(info, *s);
#ifdef XAD_SKIP_ZERO_PARTIALS
            if (!info.dropZeros())
                return;  // all partials zero: the result stays passive
#endif
            slot_ = s->recordStatement(info.multipliers, info.slots, info.index);
            setTape(s);
#elif defined(XAD_SKIP_ZERO_PARTIALS)
            // all partials zero: the result stays passive
            if (pushAll<ExprTraits<Expr>::numVariables>(s, expr))
            {
                registerOn(s);
                s->pushLhs(slot_);
            }
#else
            registerOn(s);
            pushAll<ExprTraits<Expr>::numVariables>(s, expr);
            s->pushLhs(slot_);
#endif
        }
    }
#ifdef XAD_ENABLE_JIT
    else
    {
        this->tryRecordJitFromExprCtor(expr, std::integral_constant<bool, jit_supported>());
    }
#endif
}

template <class Scalar, std::size_t M>
template <class Expr>
XAD_FORCE_INLINE AReal<Scalar, M>& AReal<Scalar, M>::assignExpression(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
{
    tape_type* s = tape_type::getActive();
    if (s)
    {
        if (expr.shouldRecord() || this->shouldRecord())
        {
#ifdef XAD_OUTLINE_RECORDING
            DerivInfo<tape_type, ExprTraits<Expr>::numVariables> info;
            expr.calc_derivatives(info, *s);
#if defined(XAD_TAPE_CHECKS) || defined(XAD_ENABLE_JIT)
            if (!isOn(s))
                slot_ = INVALID_SLOT;  // a slot of another tape or a JIT slot: register on s
#endif
#ifdef XAD_SKIP_ZERO_PARTIALS
            // all partials zero: a passive lhs stays passive, an active one gets an empty statement
            if (info.dropZeros() || isOn(s))
            {
#endif
#ifdef XAD_LINEAR_SLOTS
                // linear slots: every assignment writes a new slot, as copies may share the old one
                slot_ = s->recordStatement(info.multipliers, info.slots, info.index);
#else
#ifdef XAD_COPY_ELISION
                // the partials are taken: a new slot is not among them
                if (!ownsSlot(s) && isOn(s))
                    detachSlot(s);
#endif
                s->recordAssignment(slot_, info.multipliers, info.slots, info.index);
#endif
                setTape(s);
#ifdef XAD_SKIP_ZERO_PARTIALS
            }
#endif
#elif defined(XAD_SKIP_ZERO_PARTIALS)
            // all partials zero: a passive lhs stays passive, an active one gets an empty statement
            if (pushAll<ExprTraits<Expr>::numVariables>(s, expr) || isOn(s))
            {
#ifdef XAD_LINEAR_SLOTS
                registerOn(s);  // linear slots: a new slot, as copies may share the old one
#elif defined(XAD_COPY_ELISION)
                if (!ownsSlot(s))
                    claimSlot(s);
#else
                if (!isOn(s))
                    registerOn(s);
#endif
                s->pushLhs(slot_);
            }
#else
            pushAll<ExprTraits<Expr>::numVariables>(s, expr);
            // only register this variable after evaluating the expression, as this
            // variable might appear on the rhs of the equation too and if not yet
            // registered, it doesn't need recording of derivatives
#ifdef XAD_LINEAR_SLOTS
            registerOn(s);  // linear slots: a new slot, as copies may share the old one
#elif defined(XAD_COPY_ELISION)
            if (!ownsSlot(s))
                claimSlot(s);
#else
            if (!isOn(s))
                registerOn(s);
#endif
            s->pushLhs(slot_);
#endif
        }
    }
#ifdef XAD_ENABLE_JIT
    else
    {
        this->tryRecordJitFromExprAssign(expr, std::integral_constant<bool, jit_supported>());
    }
#endif
    this->a_ = expr.getValue();
    return *this;
}

template <class, std::size_t>
struct FReal;

template <class Scalar, std::size_t N>
struct ExprTraits<FReal<Scalar, N>>
{
    static const bool isExpr = true;
    static const int numVariables = 1;
    static const bool isForward = true;
    static const bool isReverse = false;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_FORWARD;
    static const std::size_t vector_size = 1;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef FReal<Scalar, N> value_type;
    typedef Scalar scalar_type;
};

template <class, std::size_t>
struct FRealDirect;

template <class Scalar, std::size_t N>
struct ExprTraits<FRealDirect<Scalar, N>>
{
    static const bool isExpr = false;
    static const int numVariables = 1;
    static const bool isForward = true;
    static const bool isReverse = false;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_FORWARD;
    static const std::size_t vector_size = N;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef FRealDirect<Scalar, N> value_type;
    typedef Scalar scalar_type;
};

template <class, std::size_t>
struct ARealDirect;

template <class Scalar, std::size_t N>
struct ExprTraits<ARealDirect<Scalar, N>>
{
    static const bool isExpr = false;
    static const int numVariables = 1;
    static const bool isForward = false;
    static const bool isReverse = true;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_REVERSE;
    static const std::size_t vector_size = N;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef ARealDirect<Scalar, N> value_type;
    typedef Scalar scalar_type;
};

template <class Scalar, std::size_t N = 1>
struct FReal
    : public ADTypeBase<Scalar, FReal<Scalar, N>, typename FRealTraits<Scalar, N>::derivative_type>
{
    typedef typename FRealTraits<Scalar, N>::derivative_type derivative_type;
    typedef ADTypeBase<Scalar, typename FRealTraits<Scalar, N>::type, derivative_type> base_type;
    typedef Scalar value_type;
    typedef typename ExprTraits<Scalar>::nested_type nested_type;

    constexpr XAD_INLINE FReal(nested_type val = nested_type(),
                               derivative_type der = derivative_type())
        : base_type(val), der_(der)
    {
    }

    // explicit conversion from int (also used by static_cast) to avoid warnings
    template <class U>
    constexpr XAD_INLINE explicit FReal(
        U val, typename std::enable_if<std::is_integral<U>::value>::type* = 0)
        : base_type(static_cast<nested_type>(val)), der_()
    {
    }

    constexpr XAD_INLINE FReal(const FReal& o) : base_type(o), der_(o.der_) {}
    constexpr XAD_INLINE FReal(FReal&& o) noexcept = default;
    XAD_INLINE FReal& operator=(const FReal& o) = default;
    XAD_INLINE FReal& operator=(FReal&& o) noexcept = default;

    XAD_INLINE FReal& operator=(nested_type x)
    {
        this->a_ = x;
        der_ = derivative_type();
        return *this;
    }

    template <class Expr>
    XAD_INLINE FReal(const Expression<Scalar, Expr, derivative_type>& expr);
    template <class Expr>
    XAD_INLINE FReal& operator=(const Expression<Scalar, Expr, derivative_type>& expr);

    XAD_INLINE ~FReal() = default;

    XAD_INLINE void setDerivative(derivative_type a) { derivative() = a; }
    XAD_INLINE derivative_type getDerivative() const { return derivative(); }
    XAD_INLINE derivative_type& derivative() { return der_; }
    XAD_INLINE const derivative_type& derivative() const { return der_; }

    // functions in base class that are meant only for reverse mode are
    // implemented here as stubs. They are never called, but this avoids
    // warnings.
    XAD_INLINE bool shouldRecord() const { return false; }
    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape&, const Scalar&) const
    {
    }

  private:
    template <class Expr>
    XAD_INLINE void assign(const Expression<Scalar, Expr, derivative_type>& expr, std::true_type);
    template <class Expr>
    XAD_INLINE void assign(const Expression<Scalar, Expr, derivative_type>& expr, std::false_type);

    derivative_type der_;
};

template <class Scalar, std::size_t N>
template <class Expr>
XAD_INLINE FReal<Scalar, N>::FReal(
    const Expression<Scalar, Expr, typename FReal<Scalar, N>::derivative_type>& expr)
    : base_type(xad::value(expr)), der_(xad::derivative(expr))
{
}

template <class Scalar, std::size_t N>
template <class Expr>
XAD_INLINE FReal<Scalar, N>& FReal<Scalar, N>::operator=(
    const Expression<Scalar, Expr, typename FReal<Scalar, N>::derivative_type>& expr)
{
    assign(expr, std::is_reference<typename ExprStorage<FReal>::type>());
    return *this;
}

// held by reference: the tangent first, as expr may read this variable's value
template <class Scalar, std::size_t N>
template <class Expr>
XAD_INLINE void FReal<Scalar, N>::assign(
    const Expression<Scalar, Expr, typename FReal<Scalar, N>::derivative_type>& expr,
    std::true_type)
{
    using xad::derivative;
    using xad::value;
    der_ = derivative(expr);
    this->a_ = value(expr);
}

// held by value: expr has its own copy, and this order is the cheaper one
template <class Scalar, std::size_t N>
template <class Expr>
XAD_INLINE void FReal<Scalar, N>::assign(
    const Expression<Scalar, Expr, typename FReal<Scalar, N>::derivative_type>& expr,
    std::false_type)
{
    using xad::derivative;
    using xad::value;
    this->a_ = value(expr);
    der_ = derivative(expr);
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE const Scalar& value(const AReal<Scalar, M>& x)
{
    return x.value();
}

template <class Scalar, std::size_t M>
XAD_INLINE Scalar& value(AReal<Scalar, M>& x)
{
    return x.value();
}

template <class Scalar, std::size_t N>
XAD_INLINE const Scalar& value(const FReal<Scalar, N>& x)
{
    return x.value();
}

template <class Scalar, std::size_t N>
XAD_INLINE Scalar& value(FReal<Scalar, N>& x)
{
    return x.value();
}

template <class T>
XAD_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type&
value(T& x)
{
    return x;
}

template <class T>
XAD_INLINE const typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value,
                                         T>::type&
value(const T& x)
{
    return x;
}

template <class Scalar, std::size_t N>
XAD_INLINE const typename FReal<Scalar, N>::derivative_type& derivative(const FReal<Scalar, N>& fr)
{
    return fr.derivative();
}

template <class Scalar, std::size_t N>
XAD_INLINE typename FReal<Scalar, N>::derivative_type& derivative(FReal<Scalar, N>& fr)
{
    return fr.derivative();
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE const typename AReal<Scalar, M>::derivative_type& derivative(const AReal<Scalar, M>& fr)
{
    return fr.derivative();
}

template <class Scalar, std::size_t M = 1>
XAD_INLINE typename AReal<Scalar, M>::derivative_type& derivative(AReal<Scalar, M>& fr)
{
    return fr.derivative();
}

template <class T>
XAD_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type
derivative(T&)
{
    return T();
}

template <class T>
XAD_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type
derivative(const T&)
{
    return T();
}

template <class C, class T, class Scalar, class Derived, class Deriv>
XAD_INLINE std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                                const Expression<Scalar, Derived, Deriv>& x)
{
    return os << value(x);
}

// as x = v: drops the dependency or tangent
template <class C, class T, class Scalar, std::size_t N>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, AReal<Scalar, N>& x)
{
    typename AReal<Scalar, N>::nested_type v;
    if (is >> v)
        x = v;
    return is;
}

template <class C, class T, class Scalar, std::size_t N>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, FReal<Scalar, N>& x)
{
    typename FReal<Scalar, N>::nested_type v;
    if (is >> v)
        x = v;
    return is;
}

typedef AReal<double> AD;
typedef AReal<float> AF;

typedef FReal<double> FAD;
typedef FReal<float> FAF;

typedef ARealDirect<double, 1> ADD;
typedef ARealDirect<float, 1> AFD;
}  // namespace xad


// libc++ 21 replaced the __promote class template with the SFINAE-friendly
// alias __promote_t, which no longer needs (or allows) these specializations
#if __clang_major__ > 16 && defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 210000

namespace std {

// to make libc++ happy when calling pow(AReal, int) and similar functions
template<class T, class T1, std::size_t N>
class __promote<xad::AReal<T, N>, T1> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T1, std::size_t N>
class __promote<T1, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, std::size_t N>
class __promote<xad::AReal<T, N>, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<xad::AReal<T, N>, T1, T2> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, xad::AReal<T, N>, T2> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, T2, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<xad::AReal<T, N>, xad::AReal<T, N>, T2> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<xad::AReal<T, N>, T2, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<T2, xad::AReal<T, N>, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

template<class T, std::size_t N>
class __promote<xad::AReal<T, N>, xad::AReal<T, N>, xad::AReal<T, N>> {
public:
   using type = xad::AReal<T, N>;
};

// for for FReal
template<class T, class T1, std::size_t N>
class __promote<xad::FReal<T, N>, T1> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T1, std::size_t N>
class __promote<T1, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, std::size_t N>
class __promote<xad::FReal<T, N>, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<xad::FReal<T, N>, T1, T2> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, xad::FReal<T, N>, T2> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, T2, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<xad::FReal<T, N>, xad::FReal<T, N>, T2> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<xad::FReal<T, N>, T2, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<T2, xad::FReal<T, N>, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

template<class T, std::size_t N>
class __promote<xad::FReal<T, N>, xad::FReal<T, N>, xad::FReal<T, N>> {
public:
   using type = xad::FReal<T, N>;
};

// direct mode types
#define XAD_DIRECT_PROMOTE(Direct)                                                                 \
    template <class T, class T1, std::size_t N>                                                    \
    class __promote<xad::Direct<T, N>, T1>                                                         \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T1, std::size_t N>                                                    \
    class __promote<T1, xad::Direct<T, N>>                                                         \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, std::size_t N>                                                              \
    class __promote<xad::Direct<T, N>, xad::Direct<T, N>>                                          \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T1, class T2, std::size_t N>                                          \
    class __promote<xad::Direct<T, N>, T1, T2>                                                     \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T1, class T2, std::size_t N>                                          \
    class __promote<T1, xad::Direct<T, N>, T2>                                                     \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T1, class T2, std::size_t N>                                          \
    class __promote<T1, T2, xad::Direct<T, N>>                                                     \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T2, std::size_t N>                                                    \
    class __promote<xad::Direct<T, N>, xad::Direct<T, N>, T2>                                      \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T2, std::size_t N>                                                    \
    class __promote<xad::Direct<T, N>, T2, xad::Direct<T, N>>                                      \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, class T2, std::size_t N>                                                    \
    class __promote<T2, xad::Direct<T, N>, xad::Direct<T, N>>                                      \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };                                                                                             \
    template <class T, std::size_t N>                                                              \
    class __promote<xad::Direct<T, N>, xad::Direct<T, N>, xad::Direct<T, N>>                       \
    {                                                                                              \
      public:                                                                                      \
        using type = xad::Direct<T, N>;                                                            \
    };

XAD_DIRECT_PROMOTE(ARealDirect)
XAD_DIRECT_PROMOTE(FRealDirect)

#undef XAD_DIRECT_PROMOTE
}

#endif
/*******************************************************************************

   Literal AD types for all modes.

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

    XAD_INLINE Derived& operator+=(Scalar x)
    {
        a_ += x;
        return derived();
    }
    template <class I>
    XAD_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator+=(I x)
    {
        return *this += Scalar(x);
    }
    XAD_INLINE Derived& operator-=(Scalar rhs)
    {
        a_ -= rhs;
        return derived();
    }
    template <class I>
    XAD_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator-=(I x)
    {
        return *this -= Scalar(x);
    }
    XAD_INLINE Derived& operator*=(Scalar x) { return derived() = (derived() * x); }
    template <class I>
    XAD_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator*=(I x)
    {
        return *this *= Scalar(x);
    }
    XAD_INLINE Derived& operator/=(Scalar x) { return derived() = (derived() / x); }
    template <class I>
    XAD_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator/=(I x)
    {
        return *this /= Scalar(x);
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

    XAD_INLINE AReal(const AReal& o) : base_type(), slot_(INVALID_SLOT)
    {
        auto s = tape_type::getActive();
        if (s && o.shouldRecord())
        {
            slot_ = s->registerVariable();
            pushAll<1>(s, o);
            s->pushLhs(slot_);
        }
#ifdef XAD_ENABLE_JIT
        else if (!s)
        {
            // Only check JIT if tape is not active
            jit_type* j = jit_type::getActive();
            if (j && o.shouldRecord())
            {
                // Copy the slot directly - preserves JIT dependency chain
                slot_ = o.slot_;
            }
        }
#endif
        this->a_ = o.getValue();
    }

    static constexpr slot_type INVALID_SLOT = tape_type::INVALID_SLOT;
    XAD_INLINE slot_type getSlot() const { return slot_; }
    XAD_INLINE tape_type* getTape() const { return tape_type::getActive(); }

    XAD_INLINE AReal(AReal&& o) noexcept : base_type(static_cast<base_type&&>(o)), slot_(o.slot_)
    {
        o.slot_ = INVALID_SLOT;
    }

    XAD_INLINE AReal& operator=(AReal&& o) noexcept
    {
        static_cast<base_type&>(*this) = static_cast<base_type&&>(o);
        // object moved from still gets destructor called, so this makes sure that
        // the old slot we had gets destructed
        std::swap(slot_, o.slot_);
        return *this;
    }

    XAD_INLINE ~AReal()
    {
        if (auto tape = tape_type::getActive())
            if (slot_ != INVALID_SLOT)
                tape->unregisterVariable(slot_);
    }

    XAD_INLINE AReal& operator=(const AReal& o);

    XAD_INLINE AReal& operator=(nested_type x)
    {
        this->a_ = x;
        auto tape = tape_type::getActive();
        if (tape && slot_ != INVALID_SLOT)
            tape->pushLhs(slot_);
        return *this;
    }

    template <class Expr>
    XAD_INLINE AReal(const Expression<Scalar, Expr, derivative_type>&
                         expr);  // cppcheck-suppress noExplicitConstructor

    template <class Expr>
    XAD_INLINE AReal& operator=(const Expression<Scalar, Expr, derivative_type>& expr);

    XAD_INLINE void setDerivative(derivative_type a) { derivative() = a; }
    XAD_INLINE void setAdjoint(derivative_type a) { setDerivative(a); }
    XAD_INLINE derivative_type getAdjoint() const { return getDerivative(); }

    template <int Size>
    XAD_FORCE_INLINE void pushRhs(DerivInfo<tape_type, Size>& info, const Scalar& mul,
                                  slot_type slot) const
    {
        info.multipliers[info.index] = mul;
        info.slots[info.index++] = slot;
    }

    template <int Size>
    XAD_FORCE_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type&,
                                           const Scalar& mul) const
    {
        if (slot_ != INVALID_SLOT)
            pushRhs(info, mul, slot_);
    }

    template <int Size>
    XAD_FORCE_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type&) const
    {
        if (slot_ != INVALID_SLOT)
            pushRhs(info, Scalar(1), slot_);
    }

    XAD_INLINE derivative_type getDerivative() const { return derivative(); }

    XAD_INLINE const derivative_type& derivative() const
    {
        auto t = tape_type::getActive();
        if (!t)
        {
#ifdef XAD_ENABLE_JIT
            // JIT only works when Scalar is the same as nested_type (no higher-order AD)
            if (std::is_same<Scalar, nested_type>::value)
            {
                auto j = jit_type::getActive();
                if (j)
                {
                    if (slot_ == INVALID_SLOT)
                    {
                        static const derivative_type zero = derivative_type();
                        return zero;
                    }
                    // JITCompiler::derivative(slot) returns derivative_type& for the configured Scalar/N
                    return j->derivative(slot_);
                }
            }
#endif
            throw NoTapeException();
        }
        if (slot_ == INVALID_SLOT)
        {
            // we return a dummy const ref if not registered on tape - always zero
            static const derivative_type zero = derivative_type();
            return zero;
        }
        return t->derivative(slot_);
    }

    XAD_INLINE derivative_type& derivative()
    {
        auto t = tape_type::getActive();
        if (!t)
        {
#ifdef XAD_ENABLE_JIT
            // JIT only works when Scalar is the same as nested_type (no higher-order AD)
            if (std::is_same<Scalar, nested_type>::value)
            {
                auto j = jit_type::getActive();
                if (j)
                {
                    if (slot_ == INVALID_SLOT)
                    {
                        slot_ = j->registerVariable();
                    }
                    // JITCompiler::derivative(slot) returns derivative_type& for the configured Scalar/N
                    return j->derivative(slot_);
                }
            }
#endif
            throw NoTapeException();
        }
        // register ourselves if not already done
        if (slot_ == INVALID_SLOT)
        {
            slot_ = t->registerVariable();
            t->pushLhs(slot_);
        }
        return t->derivative(slot_);
    }
    XAD_INLINE bool shouldRecord() const { return slot_ != INVALID_SLOT; }

#ifdef XAD_ENABLE_JIT
    uint32_t recordJIT(JITGraph& graph) const
    {
        if (slot_ != INVALID_SLOT)
            return slot_;
        // Not registered - treat as constant (handles nested AD types)
        return recordJITConstant(graph, getNestedDoubleValue(this->a_));
    }
#endif

  private:
    template <int Size, typename Expr>
    XAD_FORCE_INLINE void pushAll(tape_type* t, const Expr& expr) const
    {
        DerivInfo<tape_type, Size> info;

        expr.calc_derivatives(info, *t);

        t->pushAll(info.multipliers, info.slots, info.index);
    }

    template <class T, std::size_t d__cnt>
    friend class Tape;
#ifdef XAD_ENABLE_JIT
    template <class T, std::size_t d__cnt>
    friend class JITCompiler;
    template <class T, std::size_t d__cnt>
    friend class ABool;
#endif
    typename tape_type::slot_type slot_;
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

    XAD_INLINE Scalar getValue() const { return ar_.getValue(); }

    XAD_INLINE const Scalar& value() const { return ar_.value(); }

    template <int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type& s,
                                     const Scalar& mul) const
    {
        ar_.calc_derivatives(info, s, mul);
    }

    template <int Size>
    XAD_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type& s) const
    {
        ar_.calc_derivative(info, s);
    }

    XAD_INLINE const typename areal_type::derivative_type& derivative() const
    {
        return ar_.derivative();
    }

    XAD_INLINE bool shouldRecord() const { return shouldRecord_; }

#ifdef XAD_ENABLE_JIT
    uint32_t recordJIT(JITGraph& graph) const { return ar_.recordJIT(graph); }
#endif

  private:
    areal_type const& ar_;
    bool shouldRecord_;
};

template <class Scalar, std::size_t M>
XAD_INLINE AReal<Scalar, M>& AReal<Scalar, M>::operator=(const AReal& o)
{
    tape_type* s = tape_type::getActive();
    if (s && (o.shouldRecord() || this->shouldRecord()))
    {
        if (slot_ == INVALID_SLOT)
            slot_ = s->registerVariable();
        pushAll<1>(s, o);
        s->pushLhs(slot_);
    }
#ifdef XAD_ENABLE_JIT
    else if (!s)
    {
        // Only check JIT if tape is not active
        auto* j = jit_type::getActive();
        if (j && (o.shouldRecord() || this->shouldRecord()))
        {
            // Copy the slot directly - preserves JIT dependency chain
            slot_ = o.slot_;
        }
    }
#endif
    this->a_ = o.getValue();
    return *this;
}

template <class Scalar, std::size_t M>
template <class Expr>
XAD_INLINE AReal<Scalar, M>::AReal(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
    : base_type(expr.getValue()), slot_(INVALID_SLOT)
{
    if (expr.shouldRecord())
    {
        tape_type* s = tape_type::getActive();
        if (s)
        {
            slot_ = s->registerVariable();
            pushAll<ExprTraits<Expr>::numVariables>(s, expr);
            s->pushLhs(slot_);
        }
#ifdef XAD_ENABLE_JIT
        else
        {
            auto* j = JITCompiler<Scalar, M>::getActive();
            if (j)
                slot_ = static_cast<const Expr&>(expr).recordJIT(j->getGraph());
        }
#endif
    }
}

template <class Scalar, std::size_t M>
template <class Expr>
XAD_INLINE AReal<Scalar, M>& AReal<Scalar, M>::operator=(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
{
    if (expr.shouldRecord() || this->shouldRecord())
    {
        tape_type* s = tape_type::getActive();
        if (s)
        {
            pushAll<ExprTraits<Expr>::numVariables>(s, expr);
            // only register this variable after evaluating the expression, as this
            // variable might appear on the rhs of the equation too and if not yet
            // registered, it doesn't need recording of derivatives
            if (slot_ == INVALID_SLOT)
                slot_ = s->registerVariable();
            s->pushLhs(slot_);
        }
#ifdef XAD_ENABLE_JIT
        else
        {
            auto* j = JITCompiler<Scalar, M>::getActive();
            if (j)
                slot_ = static_cast<const Expr&>(expr).recordJIT(j->getGraph());
        }
#endif
    }
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
    using xad::derivative;
    using xad::value;
    this->a_ = value(expr);
    der_ = derivative(expr);
    return *this;
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

template <class C, class T, class Scalar, std::size_t N>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, AReal<Scalar, N>& x)
{
    return is >> value(x);
}

template <class C, class T, class Scalar, std::size_t N>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, FReal<Scalar, N>& x)
{
    return is >> value(x);
}

typedef AReal<double> AD;
typedef AReal<float> AF;

typedef FReal<double> FAD;
typedef FReal<float> FAF;

typedef ARealDirect<double, 1> ADD;
typedef ARealDirect<float, 1> AFD;
}  // namespace xad


#if __clang_major__ > 16 && defined(_LIBCPP_VERSION)

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

}

#endif
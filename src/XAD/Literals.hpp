/*******************************************************************************

   Literal AD types for all modes.

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
#include <XAD/Tape.hpp>
#include <XAD/Traits.hpp>

#include <algorithm>
#include <iosfwd>
#include <utility>

namespace xad
{
template <class>
class Tape;

template <class Scalar, class Derived>
struct ADTypeBase : public Expression<Scalar, Derived>
{
    typedef typename ExprTraits<Derived>::value_type value_type;
    typedef typename ExprTraits<Derived>::nested_type nested_type;

    static_assert(std::is_floating_point<nested_type>::value, "Active AD types only work with floating point");

    constexpr XAD_INLINE ADTypeBase(Scalar val = Scalar()) : a_(val) {}
    constexpr XAD_INLINE ADTypeBase(ADTypeBase&& o) noexcept = default;
    constexpr XAD_INLINE ADTypeBase(const ADTypeBase& o) = default;
    XAD_INLINE ADTypeBase& operator=(ADTypeBase&& o) noexcept = default;
    XAD_INLINE ADTypeBase& operator=(const ADTypeBase& o) = default;
    XAD_INLINE ~ADTypeBase() = default;

    constexpr XAD_INLINE Scalar getValue() const { return value(); }
    XAD_INLINE const Scalar& value() const { return a_; }
    XAD_INLINE Scalar& value() { return a_; }

    template <class E>
    XAD_INLINE Derived& operator+=(const Expression<Scalar, E>& x)
    {
        return derived() = (derived() + x);
    }
    template <class E>
    XAD_INLINE Derived& operator-=(const Expression<Scalar, E>& x)
    {
        return derived() = (derived() - x);
    }
    template <class E>
    XAD_INLINE Derived& operator*=(const Expression<Scalar, E>& x)
    {
        return derived() = (derived() * x);
    }
    template <class E>
    XAD_INLINE Derived& operator/=(const Expression<Scalar, E>& x)
    {
        return derived() = (derived() / x);
    }

    XAD_INLINE Derived& operator+=(Scalar x)
    {
        a_ += x;
        return derived();
    }
    XAD_INLINE Derived& operator-=(Scalar rhs)
    {
        a_ -= rhs;
        return derived();
    }
    XAD_INLINE Derived& operator*=(Scalar x) { return derived() = (derived() * x); }
    XAD_INLINE Derived& operator/=(Scalar x) { return derived() = (derived() / x); }
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

template <class>
struct AReal;
template <class>
struct ADVar;

template <class Scalar>
struct ExprTraits<AReal<Scalar>>
{
    static const bool isExpr = true;
    static const int numVariables = 1;
    static const bool isForward = false;
    static const bool isReverse = true;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_REVERSE;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef AReal<Scalar> value_type;
    typedef Scalar scalar_type;
};

template <class Scalar>
struct ExprTraits<ADVar<Scalar>> : public ExprTraits<AReal<Scalar>>
{
};

template <class Scalar>
struct AReal : public ADTypeBase<Scalar, AReal<Scalar>>
{
    typedef Tape<Scalar> tape_type;
    typedef ADTypeBase<Scalar, AReal<Scalar>> base_type;
    typedef typename tape_type::slot_type slot_type;
    typedef Scalar value_type;
    typedef typename ExprTraits<Scalar>::nested_type nested_type;

    XAD_INLINE AReal(nested_type val = nested_type()) : base_type(val), slot_(INVALID_SLOT)
    {
    }

    // explicit conversion from int (also used by static_cast) to avoid warnings
    template <class U>
    explicit AReal(U val, typename std::enable_if<std::is_integral<U>::value>::type* = 0)
        : base_type(static_cast<nested_type>(val)), slot_(INVALID_SLOT)
    {
    }

    XAD_INLINE AReal(const AReal& o) : base_type(), slot_(INVALID_SLOT)
    {
        if (o.shouldRecord())
        {
            auto s = tape_type::getActive();
            slot_ = s->registerVariable();
            o.calc_derivatives(*s);
            s->pushLhs(slot_);
        }
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
        if (slot_ != INVALID_SLOT)
            if (auto tape = tape_type::getActive())
                tape->unregisterVariable(slot_);
    }

    XAD_INLINE AReal& operator=(const AReal& o);

    XAD_INLINE AReal& operator=(nested_type x)
    {
        this->a_ = x;
        if (slot_ != INVALID_SLOT)
            tape_type::getActive()->pushLhs(slot_);
        return *this;
    }

    template <class Expr>
    XAD_INLINE AReal(const Expression<Scalar, Expr>& expr);

    template <class Expr>
    XAD_INLINE AReal& operator=(const Expression<Scalar, Expr>& expr);

    XAD_INLINE void setDerivative(Scalar a) { derivative() = a; }
    XAD_INLINE void setAdjoint(Scalar a) { setDerivative(a); }
    XAD_INLINE Scalar getAdjoint() const { return getDerivative(); }

    XAD_INLINE void calc_derivatives(tape_type& s, const Scalar& mul) const
    {
        if (slot_ != INVALID_SLOT)
            s.pushRhs(mul, slot_);
    }

    XAD_INLINE void calc_derivatives(tape_type& s) const
    {
        if (slot_ != INVALID_SLOT)
            s.pushRhs(Scalar(1), slot_);
    }

    template <typename Slot>
    XAD_INLINE void calc_derivatives(Slot* slot, Scalar* muls, int& n, const Scalar& mul) const
    {
        assert(false);
        slot[n] = slot_;
        muls[n] = mul;
        ++n;
    }

    template <typename It1, typename It2>
    XAD_INLINE void calc_derivatives(It1& sit, It2& mit, const Scalar& mul) const
    {
        assert(false);
        ::new (&*sit) slot_type(slot_);
        ::new (&*mit) Scalar(mul);
        ++sit;
        ++mit;
    }

    XAD_INLINE Scalar getDerivative() const { return derivative(); }

    XAD_INLINE const Scalar& derivative() const
    {
        if (slot_ == INVALID_SLOT)
        {
            // we return a dummy const ref if not registered on tape - always zero
            static const Scalar zero = Scalar();
            return zero;
        }
        return tape_type::getActive()->derivative(slot_);
    }

    XAD_INLINE Scalar& derivative()
    {
        auto t = tape_type::getActive();
        // register ourselves if not already done
        if (slot_ == INVALID_SLOT)
        {
            slot_ = t->registerVariable();
            t->pushLhs(slot_);
        }
        return t->derivative(slot_);
    }
    XAD_INLINE bool shouldRecord() const { return slot_ != INVALID_SLOT; }

  private:
    template <class T>
    friend class Tape;
    typename tape_type::slot_type slot_;
};

// this class wraps AReal<T> and makes sure that no new copies are created on
// the Tape
// when this guy is copied (unlike the AReal<T> copy)
// therefore we can use auto = ... in expressions
template <class Scalar>
struct ADVar : public Expression<Scalar, ADVar<Scalar>>
{
    typedef AReal<Scalar> areal_type;
    typedef typename areal_type::tape_type tape_type;

    XAD_INLINE explicit ADVar(const AReal<Scalar>& a) : ar_(a), shouldRecord_(a.shouldRecord()) {}

    XAD_INLINE Scalar getValue() const { return ar_.getValue(); }

    XAD_INLINE const Scalar& value() const { return ar_.value(); }

    XAD_INLINE void calc_derivatives(tape_type& s, const Scalar& mul) const
    {
        ar_.calc_derivatives(s, mul);
    }
    XAD_INLINE void calc_derivatives(tape_type& s) const { ar_.calc_derivative(s); }
    template <typename Slot>
    XAD_INLINE void calc_derivatives(Slot* slot, Scalar* muls, int& n, const Scalar& mul) const
    {
        ar_.calc_derivatives(slot, muls, n, mul);
    }

    template <typename It1, typename It2>
    XAD_INLINE void calc_derivatives(It1& sit, It2& mit, const Scalar& mul) const
    {
        ar_.calc_derivatives(sit, mit, mul);
    }

    XAD_INLINE const Scalar& derivative() const { return ar_.derivative(); }

    XAD_INLINE bool shouldRecord() const { return shouldRecord_; }

  private:
    areal_type const& ar_;
    bool shouldRecord_;
};

template <class Scalar>
XAD_INLINE AReal<Scalar>& AReal<Scalar>::operator=(const AReal& o)
{
    if (o.shouldRecord() || this->shouldRecord())
    {
        tape_type* s = tape_type::getActive();
        if (slot_ == INVALID_SLOT)
            slot_ = s->registerVariable();
        o.calc_derivatives(*s);
        s->pushLhs(slot_);
    }
    this->a_ = o.getValue();
    return *this;
}

template <class Scalar>
template <class Expr>
XAD_INLINE AReal<Scalar>::AReal(Expression<Scalar, Expr> const& expr)
    : base_type(expr.getValue()), slot_(INVALID_SLOT)
{
    if (expr.shouldRecord())
    {
        tape_type* s = tape_type::getActive();
        slot_ = s->registerVariable();
        expr.calc_derivatives(*s);
        s->pushLhs(slot_);
    }
}

template <class Scalar>
template <class Expr>
XAD_INLINE AReal<Scalar>& AReal<Scalar>::operator=(const Expression<Scalar, Expr>& expr)
{
    if (expr.shouldRecord() || this->shouldRecord())
    {
        tape_type* s = tape_type::getActive();
        expr.calc_derivatives(*s);
        // only register this variable after evaluating the expression, as this
        // variable might appear on the rhs of the equation too and if not yet
        // registered, it doesn't need recording of derivatives
        if (slot_ == INVALID_SLOT)
            slot_ = s->registerVariable();
        s->pushLhs(slot_);
    }
    this->a_ = expr.getValue();
    return *this;
}

template <class>
struct FReal;

template <class Scalar>
struct ExprTraits<FReal<Scalar>>
{
    static const bool isExpr = true;
    static const int numVariables = 1;
    static const bool isForward = true;
    static const bool isReverse = false;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_FORWARD;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef FReal<Scalar> value_type;
    typedef Scalar scalar_type;
};

template <class Scalar>
struct FReal : public ADTypeBase<Scalar, FReal<Scalar>>
{
    typedef ADTypeBase<Scalar, FReal<Scalar>> base_type;
    typedef Scalar value_type;
    typedef typename ExprTraits<Scalar>::nested_type nested_type;

    constexpr XAD_INLINE FReal(nested_type val = nested_type(), nested_type der = nested_type())
        : base_type(val), der_(der)
    {
    }

    // explicit conversion from int (also used by static_cast) to avoid warnings
    template <class U>
    constexpr explicit FReal(U val, typename std::enable_if<std::is_integral<U>::value>::type* = 0)
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
        der_ = nested_type();
        return *this;
    }

    template <class Expr>
    XAD_INLINE FReal(const Expression<Scalar, Expr>& expr);
    template <class Expr>
    XAD_INLINE FReal& operator=(const Expression<Scalar, Expr>& expr);

    XAD_INLINE ~FReal() = default;

    XAD_INLINE void setDerivative(Scalar a) { derivative() = a; }
    XAD_INLINE Scalar getDerivative() const { return derivative(); }
    XAD_INLINE Scalar& derivative() { return der_; }
    XAD_INLINE const Scalar& derivative() const { return der_; }

    // functions in base class that are meant only for reverse mode are
    // implemented here as stubs. They are never called, but this avoids
    // warnings.
    XAD_INLINE bool shouldRecord() const { return false; }
    template <class Tape>
    XAD_INLINE void calc_derivatives(Tape&, const Scalar&) const
    {
    }

  private:
    Scalar der_;
};

template <class Scalar>
template <class Expr>
XAD_INLINE FReal<Scalar>::FReal(const Expression<Scalar, Expr>& expr)
    : base_type(xad::value(expr)), der_(xad::derivative(expr))
{
}

template <class Scalar>
template <class Expr>
XAD_INLINE FReal<Scalar>& FReal<Scalar>::operator=(const Expression<Scalar, Expr>& expr)
{
    using xad::derivative;
    using xad::value;
    this->a_ = value(expr);
    der_ = derivative(expr);
    return *this;
}

template <class Scalar>
XAD_INLINE const Scalar& value(const AReal<Scalar>& x)
{
    return x.value();
}

template <class Scalar>
XAD_INLINE Scalar& value(AReal<Scalar>& x)
{
    return x.value();
}

template <class Scalar>
XAD_INLINE const Scalar& value(const FReal<Scalar>& x)
{
    return x.value();
}

template <class Scalar>
XAD_INLINE Scalar& value(FReal<Scalar>& x)
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

template <class Scalar>
XAD_INLINE const Scalar& derivative(const FReal<Scalar>& fr)
{
    return fr.derivative();
}

template <class Scalar>
XAD_INLINE Scalar& derivative(FReal<Scalar>& fr)
{
    return fr.derivative();
}

template <class Scalar>
XAD_INLINE const Scalar& derivative(const AReal<Scalar>& fr)
{
    return fr.derivative();
}

template <class Scalar>
XAD_INLINE Scalar& derivative(AReal<Scalar>& fr)
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

template <class C, class T, class Scalar, class Derived>
XAD_INLINE std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                                const Expression<Scalar, Derived>& x)
{
    return os << value(x);
}

template <class C, class T, class Scalar>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, AReal<Scalar>& x)
{
    return is >> value(x);
}

template <class C, class T, class Scalar>
XAD_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, FReal<Scalar>& x)
{
    return is >> value(x);
}

typedef AReal<double> AD;
typedef AReal<float> AF;

typedef FReal<double> FAD;
typedef FReal<float> FAF;
}  // namespace xad

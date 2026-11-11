/*******************************************************************************

   Declare expression types.

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
#include <XAD/Traits.hpp>

#include <new>
#include <type_traits>
#include <utility>
#ifdef XAD_SKIP_ZERO_PARTIALS
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#endif

namespace xad
{

#ifdef XAD_SKIP_ZERO_PARTIALS
// whether a partial can be left off the tape (floating-point values only): +0 or -0, not NaN
template <typename T>
XAD_INLINE bool isSkippedPartial(const T& m, std::integral_constant<int, 8>)
{
    std::uint64_t b;
    std::memcpy(&b, &m, sizeof(b));
    return (b << 1) == 0;
}

template <typename T>
XAD_INLINE bool isSkippedPartial(const T& m, std::integral_constant<int, 4>)
{
    std::uint32_t b;
    std::memcpy(&b, &m, sizeof(b));
    return std::uint32_t(b << 1) == 0;
}

template <typename T>
XAD_INLINE bool isSkippedPartial(const T& m, std::integral_constant<int, 1>)
{
    return m == T(0);
}

template <typename T>
XAD_INLINE bool isSkippedPartial(const T&, std::integral_constant<int, 0>)
{
    return false;
}

// built-in floating-point types only: a nested partial (e.g. FReal) equals zero by value even with
// a non-zero tangent, and StdCompatibility.hpp makes std::is_floating_point true for it
template <typename T>
using SkippedPartialKind =
    std::integral_constant<int, !(std::is_same<T, float>::value || std::is_same<T, double>::value ||
                                  std::is_same<T, long double>::value)
                                    ? 0
                                : sizeof(T) == 8 && std::numeric_limits<T>::is_iec559 ? 8
                                : sizeof(T) == 4 && std::numeric_limits<T>::is_iec559 ? 4
                                                                                      : 1>;

template <typename T>
XAD_INLINE bool isSkippedPartial(const T& m)
{
    return isSkippedPartial(m, SkippedPartialKind<T>());
}

// drops the zero partials, keeping the order of the others; returns how many are left
template <typename T, typename S>
XAD_FORCE_INLINE unsigned dropZeroPartials(T* multipliers, S* slots, unsigned n)
{
    unsigned k = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (isSkippedPartial(multipliers[i]))
            continue;
        multipliers[k] = multipliers[i];
        slots[k++] = slots[i];
    }
    return k;
}

#ifndef XAD_REDUCED_MEMORY
template <typename T, typename S>
XAD_FORCE_INLINE unsigned dropZeroPartials(std::pair<T, S>* ops, unsigned n)
{
    unsigned k = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        if (!isSkippedPartial(ops[i].first))
            ops[k++] = ops[i];
    }
    return k;
}
#endif
#endif

namespace detail
{
// T(args...) into tape memory at p, assigned or constructed in place (see is_assigned_partial);
// directly recorded types are trivially destructible, so constructing over an old value is fine
template <class T, class... Args>
XAD_FORCE_INLINE void storeAt(T* p, std::true_type, const Args&... args)
{
    *p = T(args...);
}
template <class T, class... Args>
XAD_FORCE_INLINE void storeAt(T* p, std::false_type, const Args&... args)
{
    ::new (static_cast<void*>(p)) T(args...);
}
}  // namespace detail

// keeps information about multipliers and slots for an expression
// locally on stack
template <typename TapeType, int N>
struct DerivInfo
{
    unsigned index = 0;
    typename TapeType::value_type multipliers[N];
    typename TapeType::slot_type slots[N];
#ifdef XAD_SKIP_ZERO_PARTIALS
    unsigned zeroCount = 0;
#endif

    XAD_INLINE void add(const typename TapeType::value_type& m, typename TapeType::slot_type slot)
    {
#ifdef XAD_SKIP_ZERO_PARTIALS
        // stored anyway, dropped later if zero: no branch per partial
        zeroCount += isSkippedPartial(m);
#endif
        multipliers[index] = m;
        slots[index++] = slot;
    }

#ifdef XAD_SKIP_ZERO_PARTIALS
    // drops the zero partials; false if there were partials and all were zero
    XAD_FORCE_INLINE bool dropZeros()
    {
        if (zeroCount != 0)
            return (index = dropZeroPartials(multipliers, slots, index)) != 0;
#ifdef XAD_TAPE_CHECKS
        return index != 0;  // another tape's operands add no partials
#else
        return true;
#endif
    }
#endif
};

#ifdef XAD_REDUCED_MEMORY
// same, but writing straight into the tape's separate multiplier and slot arrays
template <typename TapeType>
struct DerivInfoDirect
{
    // how partials are written: assigned, or constructed in place (see detail::storeAt)
    typedef detail::is_assigned_partial<typename TapeType::value_type> IsAssigned;

    typename TapeType::value_type* multipliers = nullptr;
    typename TapeType::slot_type* slots = nullptr;
    std::size_t pos = 0;  // where the next one goes
#ifdef XAD_SKIP_ZERO_PARTIALS
    std::size_t start = 0;  // where the first one went
    unsigned zeroCount = 0;
#endif

    XAD_INLINE void add(const typename TapeType::value_type& m, typename TapeType::slot_type slot)
    {
#ifdef XAD_SKIP_ZERO_PARTIALS
        zeroCount += isSkippedPartial(m);
#endif
        detail::storeAt(multipliers + pos, IsAssigned(), m);
        slots[pos++] = slot;
    }

#ifdef XAD_SKIP_ZERO_PARTIALS
    // as in DerivInfo, for Size variables; with one variable the drop itself is the all-zero test
    template <int Size>
    XAD_FORCE_INLINE bool dropZeros()
    {
        if (zeroCount != 0)
        {
            if (Size > 1 && zeroCount == pos - start)
            {
                pos = start;
                return false;
            }
            pos =
                start + dropZeroPartials(multipliers + start, slots + start, unsigned(pos - start));
            return Size > 1 || pos != start;  // with more, all zero was handled above
        }
#ifdef XAD_TAPE_CHECKS
        return pos != start;  // another tape's operands add no partials
#else
        return true;
#endif
    }
#endif
};
#else
// same, but writing straight into the tape instead of a local buffer
template <typename TapeType>
struct DerivInfoDirect
{
    // how partials are written: assigned, or constructed in place (see detail::storeAt)
    typedef detail::is_assigned_partial<typename TapeType::value_type> IsAssigned;

    using pair_type = std::pair<typename TapeType::value_type, typename TapeType::slot_type>;

    unsigned index = 0;
    pair_type* dst = nullptr;
#ifdef XAD_SKIP_ZERO_PARTIALS
    unsigned zeroCount = 0;
#endif

    XAD_INLINE void add(const typename TapeType::value_type& m, typename TapeType::slot_type slot)
    {
#ifdef XAD_SKIP_ZERO_PARTIALS
        zeroCount += isSkippedPartial(m);
#endif
        detail::storeAt(dst + index++, IsAssigned(), m, slot);
    }

#ifdef XAD_SKIP_ZERO_PARTIALS
    // as in DerivInfo
    XAD_FORCE_INLINE bool dropZeros()
    {
        if (zeroCount != 0)
            return (index = dropZeroPartials(dst, index)) != 0;
#ifdef XAD_TAPE_CHECKS
        return index != 0;  // another tape's operands add no partials
#else
        return true;
#endif
    }
#endif
};
#endif

namespace detail
{
// incoming multipliers of calc_derivatives besides a value: none (+1), and -1
struct unit_multiplier
{
};
struct minus_unit
{
};

// -x; FReal and Vec component-wise (Literals.hpp), as FReal's unary minus multiplies its tangent
template <class T>
struct Negate
{
    static XAD_FORCE_INLINE T apply(const T& x) { return -x; }
};

template <class T>
XAD_FORCE_INLINE T negated(const T& x)
{
    return Negate<T>::apply(x);
}

// e's partials with the multiplier m: none, minus_unit or a value
template <class E, class Info, class Tape>
XAD_FORCE_INLINE void calcWith(const E& e, Info& info, Tape& s, unit_multiplier)
{
    e.calc_derivatives(info, s);
}
template <class E, class Info, class Tape, class M>
XAD_FORCE_INLINE void calcWith(const E& e, Info& info, Tape& s, const M& m)
{
    e.calc_derivatives(info, s, m);
}

// the multiplier m with its sign flipped
XAD_FORCE_INLINE minus_unit flipSign(unit_multiplier) { return minus_unit(); }
XAD_FORCE_INLINE unit_multiplier flipSign(minus_unit) { return unit_multiplier(); }
template <class S>
XAD_FORCE_INLINE S flipSign(const S& m)
{
    return negated(m);
}

// the multiplier m times the partial d
template <class S>
XAD_FORCE_INLINE const S& times(unit_multiplier, const S& d)
{
    return d;
}
template <class S>
XAD_FORCE_INLINE S times(minus_unit, const S& d)
{
    return negated(d);
}
template <class S>
XAD_FORCE_INLINE S times(const S& m, const S& d)
{
    return m * d;
}
}  // namespace detail

/// Represents a generic expression, for the Scalar base type.
///
/// It uses the CTRP pattern, where derived classes register themselves with
/// the base class in the second template parameter
template <class Scalar, class Derived, class DerivativeType = Scalar>
struct Expression
{
    /// get a reference to the derived object
    XAD_INLINE const Derived& derived() const { return static_cast<const Derived&>(*this); }

    /// get the value
    XAD_INLINE Scalar value() const { return derived().value(); }

    XAD_INLINE Scalar getValue() const { return value(); }

#ifdef XAD_ALLOW_INT_CONVERSION
    XAD_INLINE explicit operator char() const { return static_cast<char>(getValue()); }
    XAD_INLINE explicit operator unsigned char() const
    {
        return static_cast<unsigned char>(getValue());
    }
    XAD_INLINE explicit operator signed char() const
    {
        return static_cast<signed char>(getValue());
    }
    XAD_INLINE explicit operator short() const { return static_cast<short>(getValue()); }
    XAD_INLINE explicit operator unsigned short() const
    {
        return static_cast<unsigned short>(getValue());
    }
    XAD_INLINE explicit operator int() const { return static_cast<int>(getValue()); }
    XAD_INLINE explicit operator unsigned int() const
    {
        return static_cast<unsigned int>(getValue());
    }
    XAD_INLINE explicit operator long() const { return static_cast<long>(getValue()); }
    XAD_INLINE explicit operator unsigned long() const
    {
        return static_cast<unsigned long>(getValue());
    }
    XAD_INLINE explicit operator long long() const { return static_cast<long long>(getValue()); }
    XAD_INLINE explicit operator unsigned long long() const
    {
        return static_cast<unsigned long long>(getValue());
    }
#endif

    // convert to boolean
    XAD_INLINE explicit operator bool() const { return value() != Scalar(0); }

    /// calculate the derivatives, given a tape object
    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s) const
    {
        derived().calc_derivatives(info, s, Scalar(1));
    }

    /// calculate the derivatives, given tape and multiplier
    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s, const Scalar& multiplier) const
    {
        derived().calc_derivatives(info, s, multiplier);
    }

    /// calculate the derivatives with the multiplier -1
    template <class Info, class Tape>
    XAD_INLINE void calc_derivatives(Info& info, Tape& s, detail::minus_unit m) const
    {
        derived().calc_derivatives(info, s, m);
    }

    XAD_INLINE bool shouldRecord() const { return derived().shouldRecord(); }

    XAD_INLINE DerivativeType derivative() const { return derived().derivative(); }
};

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE Scalar value(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.value();
}

template <class Scalar, class Expr, class DerivativeType>
XAD_INLINE DerivativeType derivative(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.derivative();
}
}  // namespace xad

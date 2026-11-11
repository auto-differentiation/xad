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

template <typename T>
using SkippedPartialKind =
    std::integral_constant<int, !std::is_floating_point<T>::value ? 0
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
        multipliers[pos] = m;
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
        dst[index++] = pair_type(m, slot);
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

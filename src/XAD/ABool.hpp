/**
 *
 *   ABool: Trackable boolean for JIT-compiled conditional expressions.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2025 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Affero General Public License for more details.
 *
 *   You should have received a copy of the GNU Affero General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_JIT

#include <XAD/JITCompiler.hpp>
#include <XAD/JITExprTraits.hpp>
#include <XAD/JITGraph.hpp>
#include <XAD/Literals.hpp>

namespace xad
{

/**
 * ABool: Trackable boolean for conditional graph recording.
 *
 * Stores both:
 *   - passive_ : normal C++ boolean (used when not recording / for tape)
 *   - slot_    : JIT graph node ID for the comparison result (used during JIT)
 *
 * This enables JIT to record both branches of a conditional and select
 * at runtime, allowing graph reuse with different inputs that may take
 * different branches.
 */
template <class Scalar, std::size_t N = 1>
class ABool
{
  public:
    using jit_type = JITCompiler<Scalar, N>;
    using areal_type = AReal<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    static constexpr slot_type INVALID_SLOT = static_cast<slot_type>(-1);

    // Constructor: from plain bool (no graph tracking)
    explicit ABool(bool b = false) : passive_(b), slot_(INVALID_SLOT) {}

    // Constructor: from JIT slot + passive value (graph tracking enabled)
    ABool(slot_type slot, bool passive) : passive_(passive), slot_(slot) {}

    // Accessors
    bool passive() const { return passive_; }
    slot_type slot() const { return slot_; }
    bool hasSlot() const { return slot_ != INVALID_SLOT; }

    // Allow seamless use in existing bool contexts (for tape mode)
    operator bool() const { return passive_; }

    // Core API: Conditional selection
    // Returns: trueVal if condition is true, falseVal otherwise.
    // During JIT recording: creates OpCode::If node in the graph.
    // During tape recording: returns passive result (tape re-records anyway)
    areal_type If(const areal_type& trueVal, const areal_type& falseVal) const
    {
        // Check if JIT is active and we have a valid condition slot
        auto* jit = jit_type::getActive();
        if (jit && hasSlot())
        {
            // Get slots for both branches (they must be recorded in the graph)
            uint32_t trueSlot = trueVal.getSlot();
            uint32_t falseSlot = falseVal.getSlot();

            // If either operand doesn't have a slot, record it as a constant
            if (trueSlot == INVALID_SLOT)
            {
                trueSlot = jit->recordConstant(getNestedDoubleValue(value(trueVal)));
            }
            if (falseSlot == INVALID_SLOT)
            {
                falseSlot = jit->recordConstant(getNestedDoubleValue(value(falseVal)));
            }

            // Record the If node: If(condition, trueVal, falseVal)
            uint32_t resultSlot = jit->recordNode(JITOpCode::If, slot_, trueSlot, falseSlot);

            // Create result with the passive value and set the JIT slot
            return createWithSlot(passive_ ? value(trueVal) : value(falseVal), resultSlot);
        }

        // Fallback: no JIT or no active condition - just return passive result
        return passive_ ? trueVal : falseVal;
    }

    // Static helper: alternative call style
    static areal_type If(const ABool& cond, const areal_type& trueVal, const areal_type& falseVal)
    {
        return cond.If(trueVal, falseVal);
    }

  private:
    // Helper to create AReal with a specific slot (using friend access)
    static areal_type createWithSlot(Scalar val, slot_type slot)
    {
        areal_type result(val);
        result.slot_ = slot;
        return result;
    }

    bool passive_;      // C++ truth value
    slot_type slot_;    // JIT graph node ID for condition
};

// ============================================================================
// Comparison helper functions
// ============================================================================

namespace detail
{

// Helper: Create ABool from comparing two AReal values
template <class Scalar, std::size_t N, class Compare>
ABool<Scalar, N> compareAReal(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b,
                               Compare cmp, JITOpCode opcode)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = cmp(value(a), value(b));

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        uint32_t slotB = b.getSlot();

        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
        {
            slotA = jit->recordConstant(getNestedDoubleValue(value(a)));
        }
        if (slotB == ABool<Scalar, N>::INVALID_SLOT)
        {
            slotB = jit->recordConstant(getNestedDoubleValue(value(b)));
        }

        slot_type cmpSlot = jit->recordNode(opcode, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

// Helper: Create ABool from comparing AReal with scalar
template <class Scalar, std::size_t N, class Compare>
ABool<Scalar, N> compareAReal(const AReal<Scalar, N>& a, Scalar b,
                               Compare cmp, JITOpCode opcode)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = cmp(value(a), b);

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
        {
            slotA = jit->recordConstant(getNestedDoubleValue(value(a)));
        }

        uint32_t slotB = jit->recordConstant(getNestedDoubleValue(b));
        slot_type cmpSlot = jit->recordNode(opcode, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

// Comparison functors
struct LessThan { template<class T> bool operator()(T a, T b) const { return a < b; } };
struct GreaterThan { template<class T> bool operator()(T a, T b) const { return a > b; } };
struct LessEqual { template<class T> bool operator()(T a, T b) const { return a <= b; } };
struct GreaterEqual { template<class T> bool operator()(T a, T b) const { return a >= b; } };

}  // namespace detail

// ============================================================================
// Public comparison functions
// ============================================================================

template <class Scalar, std::size_t N>
ABool<Scalar, N> less(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::LessThan{}, JITOpCode::CmpLT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> less(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::LessThan{}, JITOpCode::CmpLT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::GreaterThan{}, JITOpCode::CmpGT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::GreaterThan{}, JITOpCode::CmpGT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::LessEqual{}, JITOpCode::CmpLE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::LessEqual{}, JITOpCode::CmpLE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::GreaterEqual{}, JITOpCode::CmpGE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::GreaterEqual{}, JITOpCode::CmpGE);
}

// Convenience typedef
using ADBool = ABool<double, 1>;

}  // namespace xad

#endif  // XAD_ENABLE_JIT

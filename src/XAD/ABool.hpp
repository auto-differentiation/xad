/**
 *
 *   ABool: Trackable boolean for conditional expressions in recorded graphs.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
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

#ifdef XAD_ENABLE_GRAPH

#include <XAD/GraphRecorder.hpp>
#include <XAD/GraphExprTraits.hpp>
#include <XAD/Graph.hpp>
#include <XAD/Literals.hpp>

namespace xad
{

/**
 * ABool: Trackable boolean for conditional graph recording.
 *
 * Stores both:
 *   - passive_ : normal C++ boolean (used when not recording / for tape)
 *   - slot_    : graph node ID for the comparison result (used during graph recording)
 *
 * This lets the graph recorder record both branches of a conditional and select
 * at runtime, allowing graph reuse with different inputs that may take
 * different branches.
 */
template <class Scalar, std::size_t N = 1>
class ABool
{
  public:
    using graph_recorder_type = GraphRecorder<Scalar, N>;
    using areal_type = AReal<Scalar, N>;
    using slot_type = typename graph_recorder_type::slot_type;

    static constexpr slot_type INVALID_SLOT = static_cast<slot_type>(-1);

    // Constructor: from plain bool (no graph tracking)
    explicit ABool(bool b = false) : passive_(b), slot_(INVALID_SLOT) {}

    // Constructor: from graph slot + passive value (graph tracking enabled)
    ABool(slot_type slot, bool passive) : passive_(passive), slot_(slot) {}

    // Accessors
    bool passive() const { return passive_; }
    slot_type slot() const { return slot_; }
    bool hasSlot() const { return slot_ != INVALID_SLOT; }

    // Allow seamless use in existing bool contexts (for tape mode)
    operator bool() const { return passive_; }

    // Core API: Conditional selection
    // Returns: trueVal if condition is true, falseVal otherwise.
    // During graph recording: creates OpCode::If node in the graph.
    // During tape recording: returns passive result (tape re-records anyway)
    areal_type If(const areal_type& trueVal, const areal_type& falseVal) const
    {
        // Check if a graph recorder is active and we have a valid condition slot
        auto* recorder = graph_recorder_type::getActive();
        if (recorder && hasSlot())
        {
            // Get nodes for both branches, constants for those not recorded in the graph
            uint32_t trueSlot = trueVal.recordGraph(recorder->getGraph());
            uint32_t falseSlot = falseVal.recordGraph(recorder->getGraph());

            // Record the If node: If(condition, trueVal, falseVal)
            uint32_t resultSlot = recorder->recordNode(GraphOpCode::If, slot_, trueSlot, falseSlot);

            // Create result with the passive value and set the graph slot
            return createWithSlot(passive_ ? value(trueVal) : value(falseVal), resultSlot);
        }

        // Fallback: no graph recorder or no active condition - just return passive result
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
        result.slot_ = graphSlot(slot);
        return result;
    }

    bool passive_;      // C++ truth value
    slot_type slot_;    // graph node ID for condition
};

// ============================================================================
// Comparison helper functions
// ============================================================================

namespace detail
{

// Helper: Create ABool from comparing two AReal values
template <class Scalar, std::size_t N, class Compare>
ABool<Scalar, N> compareAReal(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b,
                               Compare cmp, GraphOpCode opcode)
{
    using graph_recorder_type = GraphRecorder<Scalar, N>;
    using slot_type = typename graph_recorder_type::slot_type;

    bool passive = cmp(value(a), value(b));

    auto* recorder = graph_recorder_type::getActive();
    if (recorder)
    {
        uint32_t slotA = a.recordGraph(recorder->getGraph());
        uint32_t slotB = b.recordGraph(recorder->getGraph());

        slot_type cmpSlot = recorder->recordNode(opcode, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

// Helper: Create ABool from comparing AReal with scalar
template <class Scalar, std::size_t N, class Compare>
ABool<Scalar, N> compareAReal(const AReal<Scalar, N>& a, Scalar b,
                               Compare cmp, GraphOpCode opcode)
{
    using graph_recorder_type = GraphRecorder<Scalar, N>;
    using slot_type = typename graph_recorder_type::slot_type;

    bool passive = cmp(value(a), b);

    auto* recorder = graph_recorder_type::getActive();
    if (recorder)
    {
        uint32_t slotA = a.recordGraph(recorder->getGraph());
        uint32_t slotB = recorder->recordConstant(getNestedDoubleValue(b));
        slot_type cmpSlot = recorder->recordNode(opcode, slotA, slotB);
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
    return detail::compareAReal(a, b, detail::LessThan{}, GraphOpCode::CmpLT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> less(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::LessThan{}, GraphOpCode::CmpLT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::GreaterThan{}, GraphOpCode::CmpGT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::GreaterThan{}, GraphOpCode::CmpGT);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::LessEqual{}, GraphOpCode::CmpLE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::LessEqual{}, GraphOpCode::CmpLE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    return detail::compareAReal(a, b, detail::GreaterEqual{}, GraphOpCode::CmpGE);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, Scalar b)
{
    return detail::compareAReal(a, b, detail::GreaterEqual{}, GraphOpCode::CmpGE);
}

// Convenience typedef
using ADBool = ABool<double, 1>;

}  // namespace xad

#endif  // XAD_ENABLE_GRAPH

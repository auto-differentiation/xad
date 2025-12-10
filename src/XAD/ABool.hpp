#pragma once

#include <XAD/Literals.hpp>
#include <XAD/JITCompiler.hpp>
#include <XAD/JITGraph.hpp>

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

    static constexpr slot_type INVALID_SLOT = jit_type::INVALID_SLOT;

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
                trueSlot = jit->recordConstant(static_cast<double>(value(trueVal)));
            if (falseSlot == INVALID_SLOT)
                falseSlot = jit->recordConstant(static_cast<double>(value(falseVal)));

            // Record the If node: If(condition, trueVal, falseVal)
            uint32_t resultSlot = jit->recordNode(JITOpCode::If, slot_, trueSlot, falseSlot);

            // Create result with the passive value and set the JIT slot
            areal_type result(passive_ ? value(trueVal) : value(falseVal));
            // Use JIT to set the slot via derivative access (which resizes derivatives_)
            jit->derivative(resultSlot);
            // We need to return a value with the correct slot - construct directly
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

// Helper functions to create ABool from AReal comparisons

template <class Scalar, std::size_t N>
ABool<Scalar, N> less(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) < value(b));

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        uint32_t slotB = b.getSlot();

        // Record constants if needed
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));
        if (slotB == ABool<Scalar, N>::INVALID_SLOT)
            slotB = jit->recordConstant(static_cast<double>(value(b)));

        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpLT, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> less(const AReal<Scalar, N>& a, Scalar b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) < b);

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));

        uint32_t slotB = jit->recordConstant(static_cast<double>(b));
        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpLT, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) > value(b));

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        uint32_t slotB = b.getSlot();

        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));
        if (slotB == ABool<Scalar, N>::INVALID_SLOT)
            slotB = jit->recordConstant(static_cast<double>(value(b)));

        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpGT, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greater(const AReal<Scalar, N>& a, Scalar b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) > b);

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));

        uint32_t slotB = jit->recordConstant(static_cast<double>(b));
        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpGT, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) <= value(b));

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        uint32_t slotB = b.getSlot();

        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));
        if (slotB == ABool<Scalar, N>::INVALID_SLOT)
            slotB = jit->recordConstant(static_cast<double>(value(b)));

        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpLE, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> lessEqual(const AReal<Scalar, N>& a, Scalar b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) <= b);

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));

        uint32_t slotB = jit->recordConstant(static_cast<double>(b));
        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpLE, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, const AReal<Scalar, N>& b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) >= value(b));

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        uint32_t slotB = b.getSlot();

        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));
        if (slotB == ABool<Scalar, N>::INVALID_SLOT)
            slotB = jit->recordConstant(static_cast<double>(value(b)));

        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpGE, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

template <class Scalar, std::size_t N>
ABool<Scalar, N> greaterEqual(const AReal<Scalar, N>& a, Scalar b)
{
    using jit_type = JITCompiler<Scalar, N>;
    using slot_type = typename jit_type::slot_type;

    bool passive = (value(a) >= b);

    auto* jit = jit_type::getActive();
    if (jit)
    {
        uint32_t slotA = a.getSlot();
        if (slotA == ABool<Scalar, N>::INVALID_SLOT)
            slotA = jit->recordConstant(static_cast<double>(value(a)));

        uint32_t slotB = jit->recordConstant(static_cast<double>(b));
        slot_type cmpSlot = jit->recordNode(JITOpCode::CmpGE, slotA, slotB);
        return ABool<Scalar, N>(cmpSlot, passive);
    }

    return ABool<Scalar, N>(passive);
}

// Convenience typedef
using ADBool = ABool<double, 1>;

}  // namespace xad

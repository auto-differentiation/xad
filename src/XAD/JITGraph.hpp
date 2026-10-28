/**
 *
 *   JITGraph: a compact opcode graph representation for JIT backends.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, version 3 of the License only.
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

#include <XAD/ChunkContainer.hpp>
#include <XAD/Exceptions.hpp>
#include <XAD/Macros.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace xad
{

enum class JITOpCode : uint16_t
{
    Input = 0,
    Constant = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    Neg = 6,
    Abs = 7,
    Square = 8,
    Recip = 9,
    Mod = 10,
    Exp = 11,
    Log = 12,
    Sqrt = 13,
    Pow = 14,
    Sin = 15,
    Cos = 16,
    Tan = 17,
    Min = 18,
    Max = 19,
    If = 20,
    CmpLT = 21,
    CmpLE = 22,
    CmpGT = 23,
    CmpGE = 24,
    CmpEQ = 25,
    CmpNE = 26,
    Asin = 27,
    Acos = 28,
    Atan = 29,
    Sinh = 30,
    Cosh = 31,
    Tanh = 32,
    Atan2 = 33,
    Floor = 34,
    Ceil = 35,
    Cbrt = 36,
    Erf = 37,
    Erfc = 38,
    Expm1 = 39,
    Log1p = 40,
    Log10 = 41,
    Log2 = 42,
    Asinh = 43,
    Acosh = 44,
    Atanh = 45,
    Exp2 = 46,
    Trunc = 47,
    Round = 48,
    Fmod = 49,
    Remainder = 50,
    Remquo = 51,
    Hypot = 52,
    Nextafter = 53,
    Ldexp = 54,
    Frexp = 55,
    Modf = 56,
    Copysign = 57,
    SmoothAbs = 58
};

// AReal slots of graph nodes take the top quarter, below INVALID_SLOT, so tapes and JITs each
// take the other's variables as passive
constexpr unsigned FIRST_JIT_SLOT = 0xC0000000u;
constexpr unsigned jitSlot(uint32_t node) { return node | FIRST_JIT_SLOT; }
constexpr uint32_t jitNode(unsigned slot) { return slot & ~FIRST_JIT_SLOT; }
constexpr bool isJitSlot(unsigned slot) { return slot >= FIRST_JIT_SLOT && slot != ~0u; }
// nodes, and nodeCount() as the next one, need distinct slots below INVALID_SLOT
constexpr uint32_t MAX_JIT_NODES = jitNode(~0u) - 1;

struct JITNodeFlags
{
    static constexpr uint8_t IsActive = 0x01;
};

struct JITNode
{
    uint16_t op = 0;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    double imm = 0.0;
    uint8_t flags = 0;
};

struct JITGraph
{
    ChunkContainer<JITNode> nodes;
    std::vector<double> const_pool;
    std::vector<uint32_t> input_ids;
    std::vector<uint32_t> output_ids;

    std::size_t nodeCount() const { return nodes.size(); }
    bool empty() const { return nodes.empty(); }

    void clear()
    {
        nodes.clear();
        const_pool.clear();
        input_ids.clear();
        output_ids.clear();
    }

    void reserve(std::size_t n)
    {
        nodes.reserve(n);
    }

    uint32_t addNode(JITOpCode op, uint32_t a = 0, uint32_t b = 0,
                     uint32_t c = 0, double imm = 0.0,
                     uint8_t fl = JITNodeFlags::IsActive)
    {
        uint32_t id = static_cast<uint32_t>(nodes.size());
        if (XAD_VERY_UNLIKELY(id >= MAX_JIT_NODES))
            throwTooManyNodes();
        JITNode n;
        n.op = static_cast<uint16_t>(op);
        n.a = a;
        n.b = b;
        n.c = c;
        n.imm = imm;
        n.flags = fl;
        nodes.push_back(n);
        return id;
    }

    uint32_t addUnary(JITOpCode op, uint32_t operand) { return addNode(op, operand); }
    uint32_t addBinary(JITOpCode op, uint32_t left, uint32_t right) { return addNode(op, left, right); }
    uint32_t addTernary(JITOpCode op, uint32_t a, uint32_t b, uint32_t c) { return addNode(op, a, b, c); }

    uint32_t addConstant(double value)
    {
        for (std::size_t i = 0; i < const_pool.size(); ++i)
        {
            // bitwise, so -0.0 and 0.0 stay apart
            if (std::memcmp(&const_pool[i], &value, sizeof(double)) == 0)
                return addNode(JITOpCode::Constant, 0, 0, 0, static_cast<double>(i));
        }
        uint32_t poolIndex = static_cast<uint32_t>(const_pool.size());
        const_pool.push_back(value);
        return addNode(JITOpCode::Constant, 0, 0, 0, static_cast<double>(poolIndex));
    }

    uint32_t addInput()
    {
        uint32_t id = addNode(JITOpCode::Input);
        input_ids.push_back(id);
        return id;
    }

    void markOutput(uint32_t nodeId) { output_ids.push_back(nodeId); }

    JITOpCode getOpCode(uint32_t nodeId) const { return static_cast<JITOpCode>(nodes[nodeId].op); }
    bool isInput(uint32_t nodeId) const { return getOpCode(nodeId) == JITOpCode::Input; }
    bool isConstant(uint32_t nodeId) const { return getOpCode(nodeId) == JITOpCode::Constant; }

    double getConstantValue(uint32_t nodeId) const
    {
        return const_pool[static_cast<std::size_t>(nodes[nodeId].imm)];
    }

  private:
    // out of line, so the check costs addNode only a compare
    [[noreturn]] static XAD_NEVER_INLINE void throwTooManyNodes()
    {
        throw OutOfRange("too many JIT graph nodes");
    }
};

}  // namespace xad

#endif  // XAD_ENABLE_JIT

#pragma once

#include <cstdint>
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
    Log2 = 42
};

struct JITNodeFlags
{
    static constexpr uint8_t IsActive = 0x01;
    static constexpr uint8_t IsDead = 0x02;
    static constexpr uint8_t NeedsGradient = 0x04;
};

struct JITGraph
{
    std::vector<uint16_t> opcodes;
    std::vector<uint32_t> operand_a;
    std::vector<uint32_t> operand_b;
    std::vector<uint32_t> operand_c;
    std::vector<double> immediates;
    std::vector<uint8_t> flags;
    std::vector<double> const_pool;
    std::vector<uint32_t> input_ids;
    std::vector<uint32_t> output_ids;

    std::size_t nodeCount() const { return opcodes.size(); }
    bool empty() const { return opcodes.empty(); }

    void clear()
    {
        opcodes.clear();
        operand_a.clear();
        operand_b.clear();
        operand_c.clear();
        immediates.clear();
        flags.clear();
        const_pool.clear();
        input_ids.clear();
        output_ids.clear();
    }

    void reserve(std::size_t n)
    {
        opcodes.reserve(n);
        operand_a.reserve(n);
        operand_b.reserve(n);
        operand_c.reserve(n);
        immediates.reserve(n);
        flags.reserve(n);
    }

    uint32_t addNode(JITOpCode op, uint32_t a = 0, uint32_t b = 0,
                     uint32_t c = 0, double imm = 0.0,
                     uint8_t fl = JITNodeFlags::IsActive)
    {
        uint32_t id = static_cast<uint32_t>(opcodes.size());
        opcodes.push_back(static_cast<uint16_t>(op));
        operand_a.push_back(a);
        operand_b.push_back(b);
        operand_c.push_back(c);
        immediates.push_back(imm);
        flags.push_back(fl);
        return id;
    }

    uint32_t addUnary(JITOpCode op, uint32_t operand) { return addNode(op, operand); }
    uint32_t addBinary(JITOpCode op, uint32_t left, uint32_t right) { return addNode(op, left, right); }
    uint32_t addTernary(JITOpCode op, uint32_t a, uint32_t b, uint32_t c) { return addNode(op, a, b, c); }

    uint32_t addConstant(double value)
    {
        for (std::size_t i = 0; i < const_pool.size(); ++i)
        {
            if (const_pool[i] == value)
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

    JITOpCode getOpCode(uint32_t nodeId) const { return static_cast<JITOpCode>(opcodes[nodeId]); }
    bool isInput(uint32_t nodeId) const { return getOpCode(nodeId) == JITOpCode::Input; }
    bool isConstant(uint32_t nodeId) const { return getOpCode(nodeId) == JITOpCode::Constant; }

    double getConstantValue(uint32_t nodeId) const
    {
        return const_pool[static_cast<std::size_t>(immediates[nodeId])];
    }
};

}  // namespace xad

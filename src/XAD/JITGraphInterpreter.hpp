#pragma once

#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace xad
{

class JITGraphInterpreter : public IJITBackend
{
  public:
    JITGraphInterpreter() = default;

    void compile(const JITGraph& graph) override
    {
        nodeValues_.resize(graph.nodeCount());
        nodeAdjoints_.resize(graph.nodeCount());
    }

    void forward(const JITGraph& graph,
                 const double* inputs, std::size_t numInputs,
                 double* outputs, std::size_t numOutputs) override
    {
        if (numInputs != graph.input_ids.size())
            throw std::runtime_error("Input count mismatch");
        if (numOutputs != graph.output_ids.size())
            throw std::runtime_error("Output count mismatch");

        nodeValues_.resize(graph.nodeCount());

        for (std::size_t i = 0; i < numInputs; ++i)
            nodeValues_[graph.input_ids[i]] = inputs[i];

        for (std::size_t i = 0; i < graph.nodeCount(); ++i)
            evaluateNode(graph, static_cast<uint32_t>(i));

        for (std::size_t i = 0; i < numOutputs; ++i)
            outputs[i] = nodeValues_[graph.output_ids[i]];
    }

    void forwardAndBackward(const JITGraph& graph,
                            const double* inputs, std::size_t numInputs,
                            const double* outputAdjoints, std::size_t numOutputs,
                            double* outputs,
                            double* inputAdjoints) override
    {
        // Run forward pass
        forward(graph, inputs, numInputs, outputs, numOutputs);

        // Run backward pass
        nodeAdjoints_.assign(graph.nodeCount(), 0.0);

        for (std::size_t i = 0; i < numOutputs; ++i)
            nodeAdjoints_[graph.output_ids[i]] = outputAdjoints[i];

        for (std::size_t i = graph.nodeCount(); i > 0; --i)
            propagateAdjoint(graph, static_cast<uint32_t>(i - 1));

        for (std::size_t i = 0; i < numInputs; ++i)
            inputAdjoints[i] = nodeAdjoints_[graph.input_ids[i]];
    }

    void reset() override
    {
        nodeValues_.clear();
        nodeAdjoints_.clear();
    }

  private:
    std::vector<double> nodeValues_;
    std::vector<double> nodeAdjoints_;

    void evaluateNode(const JITGraph& graph, uint32_t nodeId)
    {
        JITOpCode op = static_cast<JITOpCode>(graph.opcodes[nodeId]);
        uint32_t a = graph.operand_a[nodeId];
        uint32_t b = graph.operand_b[nodeId];
        uint32_t c = graph.operand_c[nodeId];
        double imm = graph.immediates[nodeId];

        double va = (a < nodeValues_.size()) ? nodeValues_[a] : 0.0;
        double vb = (b < nodeValues_.size()) ? nodeValues_[b] : 0.0;
        double vc = (c < nodeValues_.size()) ? nodeValues_[c] : 0.0;

        double result = 0.0;

        switch (op)
        {
            case JITOpCode::Input: return;
            case JITOpCode::Constant:
            {
                std::size_t idx = static_cast<std::size_t>(imm);
                if (idx >= graph.const_pool.size())
                    throw std::runtime_error("const_pool index out of bounds");
                result = graph.const_pool[idx];
                break;
            }
            case JITOpCode::Add: result = va + vb; break;
            case JITOpCode::Sub: result = va - vb; break;
            case JITOpCode::Mul: result = va * vb; break;
            case JITOpCode::Div: result = va / vb; break;
            case JITOpCode::Neg: result = -va; break;
            case JITOpCode::Abs: result = std::abs(va); break;
            case JITOpCode::Square: result = va * va; break;
            case JITOpCode::Recip: result = 1.0 / va; break;
            case JITOpCode::Sqrt: result = std::sqrt(va); break;
            case JITOpCode::Exp: result = std::exp(va); break;
            case JITOpCode::Log: result = std::log(va); break;
            case JITOpCode::Sin: result = std::sin(va); break;
            case JITOpCode::Cos: result = std::cos(va); break;
            case JITOpCode::Tan: result = std::tan(va); break;
            case JITOpCode::Asin: result = std::asin(va); break;
            case JITOpCode::Acos: result = std::acos(va); break;
            case JITOpCode::Atan: result = std::atan(va); break;
            case JITOpCode::Sinh: result = std::sinh(va); break;
            case JITOpCode::Cosh: result = std::cosh(va); break;
            case JITOpCode::Tanh: result = std::tanh(va); break;
            case JITOpCode::Pow: result = std::pow(va, vb); break;
            case JITOpCode::Min: result = std::min(va, vb); break;
            case JITOpCode::Max: result = std::max(va, vb); break;
            case JITOpCode::Mod: result = std::fmod(va, vb); break;
            case JITOpCode::Atan2: result = std::atan2(va, vb); break;
            case JITOpCode::Floor: result = std::floor(va); break;
            case JITOpCode::Ceil: result = std::ceil(va); break;
            case JITOpCode::Cbrt: result = std::cbrt(va); break;
            case JITOpCode::Erf: result = std::erf(va); break;
            case JITOpCode::Erfc: result = std::erfc(va); break;
            case JITOpCode::Expm1: result = std::expm1(va); break;
            case JITOpCode::Log1p: result = std::log1p(va); break;
            case JITOpCode::Log10: result = std::log10(va); break;
            case JITOpCode::Log2: result = std::log2(va); break;
            case JITOpCode::CmpLT: result = (va < vb) ? 1.0 : 0.0; break;
            case JITOpCode::CmpLE: result = (va <= vb) ? 1.0 : 0.0; break;
            case JITOpCode::CmpGT: result = (va > vb) ? 1.0 : 0.0; break;
            case JITOpCode::CmpGE: result = (va >= vb) ? 1.0 : 0.0; break;
            case JITOpCode::CmpEQ: result = (va == vb) ? 1.0 : 0.0; break;
            case JITOpCode::CmpNE: result = (va != vb) ? 1.0 : 0.0; break;
            case JITOpCode::If: result = (va != 0.0) ? vb : vc; break;
            default: throw std::runtime_error("Unknown opcode");
        }
        nodeValues_[nodeId] = result;
    }

    void propagateAdjoint(const JITGraph& graph, uint32_t nodeId)
    {
        double adj = nodeAdjoints_[nodeId];
        if (adj == 0.0) return;

        JITOpCode op = static_cast<JITOpCode>(graph.opcodes[nodeId]);
        uint32_t a = graph.operand_a[nodeId];
        uint32_t b = graph.operand_b[nodeId];
        uint32_t c = graph.operand_c[nodeId];

        double va = (a < nodeValues_.size()) ? nodeValues_[a] : 0.0;
        double vb = (b < nodeValues_.size()) ? nodeValues_[b] : 0.0;
        double vResult = nodeValues_[nodeId];

        switch (op)
        {
            case JITOpCode::Input:
            case JITOpCode::Constant:
                break;
            case JITOpCode::Add:
                nodeAdjoints_[a] += adj;
                nodeAdjoints_[b] += adj;
                break;
            case JITOpCode::Sub:
                nodeAdjoints_[a] += adj;
                nodeAdjoints_[b] -= adj;
                break;
            case JITOpCode::Mul:
                nodeAdjoints_[a] += adj * vb;
                nodeAdjoints_[b] += adj * va;
                break;
            case JITOpCode::Div:
                nodeAdjoints_[a] += adj / vb;
                nodeAdjoints_[b] -= adj * va / (vb * vb);
                break;
            case JITOpCode::Neg:
                nodeAdjoints_[a] -= adj;
                break;
            case JITOpCode::Abs:
                nodeAdjoints_[a] += adj * ((va >= 0.0) ? 1.0 : -1.0);
                break;
            case JITOpCode::Square:
                nodeAdjoints_[a] += adj * 2.0 * va;
                break;
            case JITOpCode::Recip:
                nodeAdjoints_[a] -= adj / (va * va);
                break;
            case JITOpCode::Sqrt:
                nodeAdjoints_[a] += adj / (2.0 * vResult);
                break;
            case JITOpCode::Exp:
                nodeAdjoints_[a] += adj * vResult;
                break;
            case JITOpCode::Log:
                nodeAdjoints_[a] += adj / va;
                break;
            case JITOpCode::Sin:
                nodeAdjoints_[a] += adj * std::cos(va);
                break;
            case JITOpCode::Cos:
                nodeAdjoints_[a] -= adj * std::sin(va);
                break;
            case JITOpCode::Tan:
                {
                    double cosv = std::cos(va);
                    nodeAdjoints_[a] += adj / (cosv * cosv);
                }
                break;
            case JITOpCode::Asin:
                nodeAdjoints_[a] += adj / std::sqrt(1.0 - va * va);
                break;
            case JITOpCode::Acos:
                nodeAdjoints_[a] -= adj / std::sqrt(1.0 - va * va);
                break;
            case JITOpCode::Atan:
                nodeAdjoints_[a] += adj / (1.0 + va * va);
                break;
            case JITOpCode::Sinh:
                nodeAdjoints_[a] += adj * std::cosh(va);
                break;
            case JITOpCode::Cosh:
                nodeAdjoints_[a] += adj * std::sinh(va);
                break;
            case JITOpCode::Tanh:
                {
                    double t = std::tanh(va);
                    nodeAdjoints_[a] += adj * (1.0 - t * t);
                }
                break;
            case JITOpCode::Pow:
                nodeAdjoints_[a] += adj * vb * std::pow(va, vb - 1.0);
                if (va > 0.0)
                    nodeAdjoints_[b] += adj * vResult * std::log(va);
                break;
            case JITOpCode::Min:
                if (va <= vb)
                    nodeAdjoints_[a] += adj;
                else
                    nodeAdjoints_[b] += adj;
                break;
            case JITOpCode::Max:
                if (va >= vb)
                    nodeAdjoints_[a] += adj;
                else
                    nodeAdjoints_[b] += adj;
                break;
            case JITOpCode::Mod:
                nodeAdjoints_[a] += adj;
                break;
            case JITOpCode::Atan2:
                {
                    double denom = va * va + vb * vb;
                    nodeAdjoints_[a] += adj * vb / denom;
                    nodeAdjoints_[b] -= adj * va / denom;
                }
                break;
            case JITOpCode::Floor:
            case JITOpCode::Ceil:
                break;
            case JITOpCode::Cbrt:
                nodeAdjoints_[a] += adj / (3.0 * vResult * vResult);
                break;
            case JITOpCode::Erf:
                nodeAdjoints_[a] += adj * (2.0 / std::sqrt(M_PI)) * std::exp(-va * va);
                break;
            case JITOpCode::Erfc:
                nodeAdjoints_[a] -= adj * (2.0 / std::sqrt(M_PI)) * std::exp(-va * va);
                break;
            case JITOpCode::Expm1:
                nodeAdjoints_[a] += adj * std::exp(va);
                break;
            case JITOpCode::Log1p:
                nodeAdjoints_[a] += adj / (1.0 + va);
                break;
            case JITOpCode::Log10:
                nodeAdjoints_[a] += adj / (va * std::log(10.0));
                break;
            case JITOpCode::Log2:
                nodeAdjoints_[a] += adj / (va * std::log(2.0));
                break;
            case JITOpCode::CmpLT:
            case JITOpCode::CmpLE:
            case JITOpCode::CmpGT:
            case JITOpCode::CmpGE:
            case JITOpCode::CmpEQ:
            case JITOpCode::CmpNE:
                break;
            case JITOpCode::If:
                if (va != 0.0)
                    nodeAdjoints_[b] += adj;
                else
                    nodeAdjoints_[c] += adj;
                break;
            default:
                break;
        }
    }
};

}  // namespace xad

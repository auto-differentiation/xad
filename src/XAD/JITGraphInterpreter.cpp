/*******************************************************************************
 *
 *   Reference JITBackend implementation that interprets a JITGraph.
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

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_JIT

#include <XAD/JITGraphInterpreter.hpp>
#include <XAD/Macros.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Provide a fallback for environments where M_PI is not defined.
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace xad
{

void JITGraphInterpreter::compile(const JITGraph& graph)
{
    nodeValues_.resize(graph.nodeCount());
    nodeAdjoints_.resize(graph.nodeCount());
}

void JITGraphInterpreter::forward(const JITGraph& graph,
                                  const double* inputs, std::size_t numInputs,
                                  double* outputs, std::size_t numOutputs)
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

void JITGraphInterpreter::forwardAndBackward(const JITGraph& graph,
                                             const double* inputs, std::size_t numInputs,
                                             const double* outputAdjoints, std::size_t numOutputs,
                                             double* outputs,
                                             double* inputAdjoints)
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

void JITGraphInterpreter::reset()
{
    nodeValues_.clear();
    nodeAdjoints_.clear();
}

double JITGraphInterpreter::invSqrtPi()
{
    return 2.0 / std::sqrt(M_PI);
}

void JITGraphInterpreter::evaluateNode(const JITGraph& graph, uint32_t nodeId)
{
    const auto& node = graph.nodes[nodeId];
    JITOpCode op = static_cast<JITOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;
    double imm = node.imm;

    double va = (a < nodeValues_.size()) ? nodeValues_[a] : 0.0;
    double vb = (b < nodeValues_.size()) ? nodeValues_[b] : 0.0;

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
        case JITOpCode::Min: result = (std::min)(va, vb); break;
        case JITOpCode::Max: result = (std::max)(va, vb); break;
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
        case JITOpCode::Asinh: result = std::asinh(va); break;
        case JITOpCode::Acosh: result = std::acosh(va); break;
        case JITOpCode::Atanh: result = std::atanh(va); break;
        case JITOpCode::Exp2: result = std::exp2(va); break;
        case JITOpCode::Trunc: result = std::trunc(va); break;
        case JITOpCode::Round: result = std::round(va); break;
        case JITOpCode::Remainder: result = std::remainder(va, vb); break;
        case JITOpCode::Remquo:
        {
            int quo;
            result = std::remquo(va, vb, &quo);
            // Store quotient in operand_c if needed
            break;
        }
        case JITOpCode::Hypot: result = std::hypot(va, vb); break;
        case JITOpCode::Nextafter: result = std::nextafter(va, vb); break;
        case JITOpCode::Ldexp: result = std::ldexp(va, static_cast<int>(imm)); break;
        case JITOpCode::Frexp:
        {
            int exp;
            result = std::frexp(va, &exp);
            // Store exponent somewhere if needed
            break;
        }
        case JITOpCode::Modf:
        {
            double intpart;
            result = std::modf(va, &intpart);
            // Store integer part somewhere if needed
            break;
        }
        case JITOpCode::Copysign: result = std::copysign(va, vb); break;
        case JITOpCode::SmoothAbs:
        {
            // Smooth abs: if |x| > c return |x|, else smooth function
            if (std::abs(va) > vb)
                result = std::abs(va);
            else if (va < 0.0)
                result = va * va * (2.0 / vb + va / (vb * vb));
            else
                result = va * va * (2.0 / vb - va / (vb * vb));
            break;
        }
        case JITOpCode::CmpLT: result = (va < vb) ? 1.0 : 0.0; break;
        case JITOpCode::CmpLE: result = (va <= vb) ? 1.0 : 0.0; break;
        case JITOpCode::CmpGT: result = (va > vb) ? 1.0 : 0.0; break;
        case JITOpCode::CmpGE: result = (va >= vb) ? 1.0 : 0.0; break;
        case JITOpCode::CmpEQ: result = (va == vb) ? 1.0 : 0.0; break;
        case JITOpCode::CmpNE: result = (va != vb) ? 1.0 : 0.0; break;
        case JITOpCode::If:
        {
            const uint32_t c = node.c;
            const double vc = (c < nodeValues_.size()) ? nodeValues_[c] : 0.0;
            result = (va != 0.0) ? vb : vc;
            break;
        }
        default: throw std::runtime_error("Unknown opcode");
    }
    nodeValues_[nodeId] = result;
}

void JITGraphInterpreter::propagateAdjoint(const JITGraph& graph, uint32_t nodeId)
{
    double adj = nodeAdjoints_[nodeId];
    if (adj == 0.0) return;

    const auto& node = graph.nodes[nodeId];
    JITOpCode op = static_cast<JITOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;

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
            // Match XAD's derivative: (a > 0) - (a < 0), which is 0 at a=0
            nodeAdjoints_[a] += adj * ((va > 0.0) ? 1.0 : ((va < 0.0) ? -1.0 : 0.0));
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
            if (va < vb)
                nodeAdjoints_[a] += adj;
            else if (vb < va)
                nodeAdjoints_[b] += adj;
            else  // va == vb
            {
                nodeAdjoints_[a] += adj * 0.5;
                nodeAdjoints_[b] += adj * 0.5;
            }
            break;
        case JITOpCode::Max:
            if (vb < va)
                nodeAdjoints_[a] += adj;
            else if (va < vb)
                nodeAdjoints_[b] += adj;
            else  // va == vb
            {
                nodeAdjoints_[a] += adj * 0.5;
                nodeAdjoints_[b] += adj * 0.5;
            }
            break;
        case JITOpCode::Mod:
            nodeAdjoints_[a] += adj;
            nodeAdjoints_[b] -= adj * std::floor(va / vb);
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
            nodeAdjoints_[a] += adj * invSqrtPi() * std::exp(-va * va);
            break;
        case JITOpCode::Erfc:
            nodeAdjoints_[a] -= adj * invSqrtPi() * std::exp(-va * va);
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
        case JITOpCode::Asinh:
            nodeAdjoints_[a] += adj / std::sqrt(va * va + 1.0);
            break;
        case JITOpCode::Acosh:
            nodeAdjoints_[a] += adj / std::sqrt(va * va - 1.0);
            break;
        case JITOpCode::Atanh:
            nodeAdjoints_[a] += adj / (1.0 - va * va);
            break;
        case JITOpCode::Exp2:
            nodeAdjoints_[a] += adj * std::log(2.0) * vResult;
            break;
        case JITOpCode::Trunc:
        case JITOpCode::Round:
            // Zero derivative
            break;
        case JITOpCode::Remainder:
        {
            int quo;
            XAD_UNUSED_VARIABLE(std::remquo(va, vb, &quo));
            nodeAdjoints_[a] += adj;
            nodeAdjoints_[b] -= adj * quo;
        }
        break;
        case JITOpCode::Remquo:
        {
            int quo;
            XAD_UNUSED_VARIABLE(std::remquo(va, vb, &quo));
            nodeAdjoints_[a] += adj;
            nodeAdjoints_[b] -= adj * quo;
        }
        break;
        case JITOpCode::Hypot:
            nodeAdjoints_[a] += adj * va / vResult;
            nodeAdjoints_[b] += adj * vb / vResult;
            break;
        case JITOpCode::Nextafter:
            nodeAdjoints_[a] += adj;
            // Second operand has zero derivative
            break;
        case JITOpCode::Ldexp:
        {
            double imm = node.imm;
            int exp = static_cast<int>(imm);
            nodeAdjoints_[a] += adj * (1 << exp);
        }
        break;
        case JITOpCode::Frexp:
        {
            // Derivative is 1 / 2^exp, but we need to recompute frexp
            int exp;
            std::frexp(va, &exp);
            nodeAdjoints_[a] += adj / (1 << exp);
        }
        break;
        case JITOpCode::Modf:
            // Derivative of fractional part is 1
            nodeAdjoints_[a] += adj;
            break;
        case JITOpCode::Copysign:
            // d/da copysign(a, b) = sign(b)
            nodeAdjoints_[a] += adj * ((vb >= 0.0) ? 1.0 : -1.0);
            // d/db copysign(a, b) = 0
            break;
        case JITOpCode::SmoothAbs:
        {
            double dval;
            if (va > vb)
                dval = 1.0;
            else if (va < -vb)
                dval = -1.0;
            else if (va < 0.0)
                dval = va / (vb * vb) * (3.0 * va + 4.0 * vb);
            else
                dval = -va / (vb * vb) * (3.0 * va - 4.0 * vb);
            nodeAdjoints_[a] += adj * dval;

            // Derivative w.r.t. c (second parameter)
            double dcval;
            if (va > vb || va < -vb)
                dcval = 0.0;
            else if (va < 0.0)
                dcval = -2.0 * va * va * (vb + va) / (vb * vb * vb);
            else
                dcval = -2.0 * va * va * (vb - va) / (vb * vb * vb);
            nodeAdjoints_[b] += adj * dcval;
        }
        break;
        case JITOpCode::CmpLT:
        case JITOpCode::CmpLE:
        case JITOpCode::CmpGT:
        case JITOpCode::CmpGE:
        case JITOpCode::CmpEQ:
        case JITOpCode::CmpNE:
            break;
        case JITOpCode::If:
        {
            if (va != 0.0)
                nodeAdjoints_[b] += adj;
            else
                nodeAdjoints_[node.c] += adj;
            break;
        }
        default:
            break;
    }
}

}  // namespace xad

#endif  // XAD_ENABLE_JIT



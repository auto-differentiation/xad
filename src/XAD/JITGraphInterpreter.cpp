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
#include <vector>

// Provide a fallback for environments where M_PI is not defined.
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace xad
{

template <class Scalar>
struct JITGraphInterpreter<Scalar>::Impl
{
    const JITGraph* graph = nullptr;  // Stored from compile()
    std::vector<Scalar> inputValues;  // Current input values (set via setInput)
    std::vector<Scalar> nodeValues;   // Forward pass intermediate values
    std::vector<Scalar> nodeAdjoints; // Backward pass adjoints
};

template <class Scalar>
JITGraphInterpreter<Scalar>::JITGraphInterpreter()
    : impl_(new Impl())
{
}

template <class Scalar>
JITGraphInterpreter<Scalar>::~JITGraphInterpreter() = default;

template <class Scalar>
void JITGraphInterpreter<Scalar>::compile(const JITGraph& graph)
{
    impl_->graph = &graph;
    impl_->inputValues.resize(graph.input_ids.size());
    impl_->nodeValues.resize(graph.nodeCount());
    impl_->nodeAdjoints.resize(graph.nodeCount());
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::reset()
{
    impl_->graph = nullptr;
    impl_->inputValues.clear();
    impl_->nodeValues.clear();
    impl_->nodeAdjoints.clear();
}

template <class Scalar>
std::size_t JITGraphInterpreter<Scalar>::numInputs() const
{
    return impl_->graph ? impl_->graph->input_ids.size() : 0;
}

template <class Scalar>
std::size_t JITGraphInterpreter<Scalar>::numOutputs() const
{
    return impl_->graph ? impl_->graph->output_ids.size() : 0;
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::setInput(std::size_t inputIndex, const Scalar* values)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");
    if (inputIndex >= impl_->graph->input_ids.size())
        throw std::runtime_error("Input index out of range");

    impl_->inputValues[inputIndex] = values[0];
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::forward(Scalar* outputs)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");

    const JITGraph& graph = *impl_->graph;

    // Load input values into node values
    for (std::size_t i = 0; i < graph.input_ids.size(); ++i)
        impl_->nodeValues[graph.input_ids[i]] = impl_->inputValues[i];

    // Evaluate all nodes
    for (std::size_t i = 0; i < graph.nodeCount(); ++i)
        evaluateNode(static_cast<uint32_t>(i));

    // Collect outputs (scalar: 1 value per output)
    for (std::size_t i = 0; i < graph.output_ids.size(); ++i)
        outputs[i] = impl_->nodeValues[graph.output_ids[i]];
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::forwardAndBackward(Scalar* outputs, Scalar* inputGradients)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");

    const JITGraph& graph = *impl_->graph;

    // Run forward pass
    forward(outputs);

    // Run backward pass - seed output adjoints to 1.0
    impl_->nodeAdjoints.assign(graph.nodeCount(), Scalar(0));
    for (std::size_t i = 0; i < graph.output_ids.size(); ++i)
        impl_->nodeAdjoints[graph.output_ids[i]] = Scalar(1);

    // Propagate adjoints backward
    for (std::size_t i = graph.nodeCount(); i > 0; --i)
        propagateAdjoint(static_cast<uint32_t>(i - 1));

    // Collect input gradients (scalar: 1 value per input)
    for (std::size_t i = 0; i < graph.input_ids.size(); ++i)
        inputGradients[i] = impl_->nodeAdjoints[graph.input_ids[i]];
}

template <class Scalar>
Scalar JITGraphInterpreter<Scalar>::invSqrtPi()
{
    return Scalar(2) / std::sqrt(Scalar(M_PI));
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::evaluateNode(uint32_t nodeId)
{
    const JITGraph& graph = *impl_->graph;
    std::vector<Scalar>& nodeValues = impl_->nodeValues;
    const auto& node = graph.nodes[nodeId];
    JITOpCode op = static_cast<JITOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;
    double imm = node.imm;

    Scalar va = (a < nodeValues.size()) ? nodeValues[a] : Scalar(0);
    Scalar vb = (b < nodeValues.size()) ? nodeValues[b] : Scalar(0);

    Scalar result = Scalar(0);

    switch (op)
    {
        case JITOpCode::Input: return;
        case JITOpCode::Constant:
        {
            std::size_t idx = static_cast<std::size_t>(imm);
            if (idx >= graph.const_pool.size())
                throw std::runtime_error("const_pool index out of bounds");
            result = static_cast<Scalar>(graph.const_pool[idx]);
            break;
        }
        case JITOpCode::Add: result = va + vb; break;
        case JITOpCode::Sub: result = va - vb; break;
        case JITOpCode::Mul: result = va * vb; break;
        case JITOpCode::Div: result = va / vb; break;
        case JITOpCode::Neg: result = -va; break;
        case JITOpCode::Abs: result = std::abs(va); break;
        case JITOpCode::Square: result = va * va; break;
        case JITOpCode::Recip: result = Scalar(1) / va; break;
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
            Scalar intpart;
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
            else if (va < Scalar(0))
                result = va * va * (Scalar(2) / vb + va / (vb * vb));
            else
                result = va * va * (Scalar(2) / vb - va / (vb * vb));
            break;
        }
        case JITOpCode::CmpLT: result = (va < vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::CmpLE: result = (va <= vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::CmpGT: result = (va > vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::CmpGE: result = (va >= vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::CmpEQ: result = (va == vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::CmpNE: result = (va != vb) ? Scalar(1) : Scalar(0); break;
        case JITOpCode::If:
        {
            const uint32_t c = node.c;
            const Scalar vc = (c < nodeValues.size()) ? nodeValues[c] : Scalar(0);
            result = (va != Scalar(0)) ? vb : vc;
            break;
        }
        default: throw std::runtime_error("Unknown opcode");
    }
    nodeValues[nodeId] = result;
}

template <class Scalar>
void JITGraphInterpreter<Scalar>::propagateAdjoint(uint32_t nodeId)
{
    const JITGraph& graph = *impl_->graph;
    std::vector<Scalar>& nodeValues = impl_->nodeValues;
    std::vector<Scalar>& nodeAdjoints = impl_->nodeAdjoints;
    Scalar adj = nodeAdjoints[nodeId];
    if (adj == Scalar(0)) return;

    const auto& node = graph.nodes[nodeId];
    JITOpCode op = static_cast<JITOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;

    Scalar va = (a < nodeValues.size()) ? nodeValues[a] : Scalar(0);
    Scalar vb = (b < nodeValues.size()) ? nodeValues[b] : Scalar(0);
    Scalar vResult = nodeValues[nodeId];

    switch (op)
    {
        case JITOpCode::Input:
        case JITOpCode::Constant:
            break;
        case JITOpCode::Add:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] += adj;
            break;
        case JITOpCode::Sub:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj;
            break;
        case JITOpCode::Mul:
            nodeAdjoints[a] += adj * vb;
            nodeAdjoints[b] += adj * va;
            break;
        case JITOpCode::Div:
            nodeAdjoints[a] += adj / vb;
            nodeAdjoints[b] -= adj * va / (vb * vb);
            break;
        case JITOpCode::Neg:
            nodeAdjoints[a] -= adj;
            break;
        case JITOpCode::Abs:
            // Match XAD's derivative: (a > 0) - (a < 0), which is 0 at a=0
            nodeAdjoints[a] += adj * ((va > Scalar(0)) ? Scalar(1) : ((va < Scalar(0)) ? Scalar(-1) : Scalar(0)));
            break;
        case JITOpCode::Square:
            nodeAdjoints[a] += adj * Scalar(2) * va;
            break;
        case JITOpCode::Recip:
            nodeAdjoints[a] -= adj / (va * va);
            break;
        case JITOpCode::Sqrt:
            nodeAdjoints[a] += adj / (Scalar(2) * vResult);
            break;
        case JITOpCode::Exp:
            nodeAdjoints[a] += adj * vResult;
            break;
        case JITOpCode::Log:
            nodeAdjoints[a] += adj / va;
            break;
        case JITOpCode::Sin:
            nodeAdjoints[a] += adj * std::cos(va);
            break;
        case JITOpCode::Cos:
            nodeAdjoints[a] -= adj * std::sin(va);
            break;
        case JITOpCode::Tan:
        {
            Scalar cosv = std::cos(va);
            nodeAdjoints[a] += adj / (cosv * cosv);
        }
        break;
        case JITOpCode::Asin:
            nodeAdjoints[a] += adj / std::sqrt(Scalar(1) - va * va);
            break;
        case JITOpCode::Acos:
            nodeAdjoints[a] -= adj / std::sqrt(Scalar(1) - va * va);
            break;
        case JITOpCode::Atan:
            nodeAdjoints[a] += adj / (Scalar(1) + va * va);
            break;
        case JITOpCode::Sinh:
            nodeAdjoints[a] += adj * std::cosh(va);
            break;
        case JITOpCode::Cosh:
            nodeAdjoints[a] += adj * std::sinh(va);
            break;
        case JITOpCode::Tanh:
        {
            Scalar t = std::tanh(va);
            nodeAdjoints[a] += adj * (Scalar(1) - t * t);
        }
        break;
        case JITOpCode::Pow:
            nodeAdjoints[a] += adj * vb * std::pow(va, vb - Scalar(1));
            if (va > Scalar(0))
                nodeAdjoints[b] += adj * vResult * std::log(va);
            break;
        case JITOpCode::Min:
            if (va < vb)
                nodeAdjoints[a] += adj;
            else if (vb < va)
                nodeAdjoints[b] += adj;
            else  // va == vb
            {
                nodeAdjoints[a] += adj * Scalar(0.5);
                nodeAdjoints[b] += adj * Scalar(0.5);
            }
            break;
        case JITOpCode::Max:
            if (vb < va)
                nodeAdjoints[a] += adj;
            else if (va < vb)
                nodeAdjoints[b] += adj;
            else  // va == vb
            {
                nodeAdjoints[a] += adj * Scalar(0.5);
                nodeAdjoints[b] += adj * Scalar(0.5);
            }
            break;
        case JITOpCode::Mod:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj * std::floor(va / vb);
            break;
        case JITOpCode::Atan2:
        {
            Scalar denom = va * va + vb * vb;
            nodeAdjoints[a] += adj * vb / denom;
            nodeAdjoints[b] -= adj * va / denom;
        }
        break;
        case JITOpCode::Floor:
        case JITOpCode::Ceil:
            break;
        case JITOpCode::Cbrt:
            nodeAdjoints[a] += adj / (Scalar(3) * vResult * vResult);
            break;
        case JITOpCode::Erf:
            nodeAdjoints[a] += adj * invSqrtPi() * std::exp(-va * va);
            break;
        case JITOpCode::Erfc:
            nodeAdjoints[a] -= adj * invSqrtPi() * std::exp(-va * va);
            break;
        case JITOpCode::Expm1:
            nodeAdjoints[a] += adj * std::exp(va);
            break;
        case JITOpCode::Log1p:
            nodeAdjoints[a] += adj / (Scalar(1) + va);
            break;
        case JITOpCode::Log10:
            nodeAdjoints[a] += adj / (va * std::log(Scalar(10)));
            break;
        case JITOpCode::Log2:
            nodeAdjoints[a] += adj / (va * std::log(Scalar(2)));
            break;
        case JITOpCode::Asinh:
            nodeAdjoints[a] += adj / std::sqrt(va * va + Scalar(1));
            break;
        case JITOpCode::Acosh:
            nodeAdjoints[a] += adj / std::sqrt(va * va - Scalar(1));
            break;
        case JITOpCode::Atanh:
            nodeAdjoints[a] += adj / (Scalar(1) - va * va);
            break;
        case JITOpCode::Exp2:
            nodeAdjoints[a] += adj * std::log(Scalar(2)) * vResult;
            break;
        case JITOpCode::Trunc:
        case JITOpCode::Round:
            // Zero derivative
            break;
        case JITOpCode::Remainder:
        {
            int quo;
            XAD_UNUSED_VARIABLE(std::remquo(va, vb, &quo));
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj * Scalar(quo);
        }
        break;
        case JITOpCode::Remquo:
        {
            int quo;
            XAD_UNUSED_VARIABLE(std::remquo(va, vb, &quo));
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj * Scalar(quo);
        }
        break;
        case JITOpCode::Hypot:
            nodeAdjoints[a] += adj * va / vResult;
            nodeAdjoints[b] += adj * vb / vResult;
            break;
        case JITOpCode::Nextafter:
            nodeAdjoints[a] += adj;
            // Second operand has zero derivative
            break;
        case JITOpCode::Ldexp:
        {
            double imm = node.imm;
            int exp = static_cast<int>(imm);
            nodeAdjoints[a] += adj * Scalar(1 << exp);
        }
        break;
        case JITOpCode::Frexp:
        {
            // Derivative is 1 / 2^exp, but we need to recompute frexp
            int exp;
            std::frexp(va, &exp);
            nodeAdjoints[a] += adj / Scalar(1 << exp);
        }
        break;
        case JITOpCode::Modf:
            // Derivative of fractional part is 1
            nodeAdjoints[a] += adj;
            break;
        case JITOpCode::Copysign:
            // d/da copysign(a, b) = sign(b)
            nodeAdjoints[a] += adj * ((vb >= Scalar(0)) ? Scalar(1) : Scalar(-1));
            // d/db copysign(a, b) = 0
            break;
        case JITOpCode::SmoothAbs:
        {
            Scalar dval;
            if (va > vb)
                dval = Scalar(1);
            else if (va < -vb)
                dval = Scalar(-1);
            else if (va < Scalar(0))
                dval = va / (vb * vb) * (Scalar(3) * va + Scalar(4) * vb);
            else
                dval = -va / (vb * vb) * (Scalar(3) * va - Scalar(4) * vb);
            nodeAdjoints[a] += adj * dval;

            // Derivative w.r.t. c (second parameter)
            Scalar dcval;
            if (va > vb || va < -vb)
                dcval = Scalar(0);
            else if (va < Scalar(0))
                dcval = Scalar(-2) * va * va * (vb + va) / (vb * vb * vb);
            else
                dcval = Scalar(-2) * va * va * (vb - va) / (vb * vb * vb);
            nodeAdjoints[b] += adj * dcval;
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
            if (va != Scalar(0))
                nodeAdjoints[b] += adj;
            else
                nodeAdjoints[node.c] += adj;
            break;
        }
        default:
            break;
    }
}

// Explicit instantiations
template class JITGraphInterpreter<float>;
template class JITGraphInterpreter<double>;

}  // namespace xad

#endif  // XAD_ENABLE_JIT



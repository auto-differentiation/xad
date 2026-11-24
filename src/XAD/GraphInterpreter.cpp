/*******************************************************************************
 *
 *   Reference GraphBackend implementation that interprets a Graph.
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

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_GRAPH

#include <XAD/GraphInterpreter.hpp>
#include <XAD/Macros.hpp>
#include <XAD/MathFunctions.hpp>

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
struct GraphInterpreter<Scalar>::Impl
{
    const Graph* graph = nullptr;  // Stored from compile()
    std::vector<Scalar> inputValues;  // Current input values (set via setInput)
    std::vector<Scalar> nodeValues;   // Forward pass intermediate values
    std::vector<Scalar> nodeAdjoints; // Backward pass adjoints
    // sizes at compile(): nodes recorded later are not evaluated
    std::size_t numNodes = 0;
    std::size_t numInputs = 0;
    std::size_t numOutputs = 0;
};

template <class Scalar>
GraphInterpreter<Scalar>::GraphInterpreter()
    : impl_(new Impl())
{
}

template <class Scalar>
GraphInterpreter<Scalar>::~GraphInterpreter() = default;

template <class Scalar>
void GraphInterpreter<Scalar>::compile(const Graph& graph)
{
    impl_->graph = &graph;
    impl_->numNodes = graph.nodeCount();
    impl_->numInputs = graph.input_ids.size();
    impl_->numOutputs = graph.output_ids.size();
    impl_->inputValues.resize(impl_->numInputs);
    impl_->nodeValues.resize(impl_->numNodes);
    impl_->nodeAdjoints.resize(impl_->numNodes);
}

template <class Scalar>
void GraphInterpreter<Scalar>::reset()
{
    impl_->graph = nullptr;
    impl_->inputValues.clear();
    impl_->nodeValues.clear();
    impl_->nodeAdjoints.clear();
    impl_->numNodes = impl_->numInputs = impl_->numOutputs = 0;
}

template <class Scalar>
std::size_t GraphInterpreter<Scalar>::numInputs() const
{
    return impl_->numInputs;
}

template <class Scalar>
std::size_t GraphInterpreter<Scalar>::numOutputs() const
{
    return impl_->numOutputs;
}

template <class Scalar>
void GraphInterpreter<Scalar>::setInput(std::size_t inputIndex, const Scalar* values)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");
    if (inputIndex >= impl_->numInputs)
        throw std::runtime_error("Input index out of range");

    impl_->inputValues[inputIndex] = values[0];
}

template <class Scalar>
void GraphInterpreter<Scalar>::forward(Scalar* outputs)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");

    const Graph& graph = *impl_->graph;

    // Load input values into node values
    for (std::size_t i = 0; i < impl_->numInputs; ++i)
        impl_->nodeValues[graph.input_ids[i]] = impl_->inputValues[i];

    // Evaluate all nodes
    for (std::size_t i = 0; i < impl_->numNodes; ++i)
        evaluateNode(static_cast<uint32_t>(i));

    // Collect outputs (scalar: 1 value per output)
    for (std::size_t i = 0; i < impl_->numOutputs; ++i)
        outputs[i] = impl_->nodeValues[graph.output_ids[i]];
}

template <class Scalar>
void GraphInterpreter<Scalar>::forwardAndBackward(Scalar* outputs, Scalar* inputGradients)
{
    forwardAndBackwardSeeded(outputs, nullptr, inputGradients);
}

template <class Scalar>
void GraphInterpreter<Scalar>::forwardAndBackwardSeeded(Scalar* outputs,
                                                           const Scalar* outputAdjoints,
                                                           Scalar* inputGradients)
{
    if (!impl_->graph)
        throw std::runtime_error("Backend not compiled");

    const Graph& graph = *impl_->graph;

    // Run forward pass
    forward(outputs);

    // Run backward pass - seed output adjoints (1.0 without outputAdjoints)
    impl_->nodeAdjoints.assign(impl_->numNodes, Scalar(0));
    for (std::size_t i = 0; i < impl_->numOutputs; ++i)
        impl_->nodeAdjoints[graph.output_ids[i]] = outputAdjoints ? outputAdjoints[i] : Scalar(1);

    // Propagate adjoints backward
    for (std::size_t i = impl_->numNodes; i > 0; --i)
        propagateAdjoint(static_cast<uint32_t>(i - 1));

    // Collect input gradients (scalar: 1 value per input)
    for (std::size_t i = 0; i < impl_->numInputs; ++i)
        inputGradients[i] = impl_->nodeAdjoints[graph.input_ids[i]];
}

template <class Scalar>
Scalar GraphInterpreter<Scalar>::invSqrtPi()
{
    return Scalar(2) / std::sqrt(Scalar(M_PI));
}

template <class Scalar>
void GraphInterpreter<Scalar>::evaluateNode(uint32_t nodeId)
{
    const Graph& graph = *impl_->graph;
    std::vector<Scalar>& nodeValues = impl_->nodeValues;
    const auto& node = graph.nodes[nodeId];
    GraphOpCode op = static_cast<GraphOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;
    double imm = node.imm;

    Scalar va = (a < nodeValues.size()) ? nodeValues[a] : Scalar(0);
    Scalar vb = (b < nodeValues.size()) ? nodeValues[b] : Scalar(0);

    Scalar result = Scalar(0);

    switch (op)
    {
        case GraphOpCode::Input: return;
        case GraphOpCode::Constant:
        {
            std::size_t idx = static_cast<std::size_t>(imm);
            if (idx >= graph.const_pool.size())
                throw std::runtime_error("const_pool index out of bounds");
            result = static_cast<Scalar>(graph.const_pool[idx]);
            break;
        }
        case GraphOpCode::Add: result = va + vb; break;
        case GraphOpCode::Sub: result = va - vb; break;
        case GraphOpCode::Mul: result = va * vb; break;
        case GraphOpCode::Div: result = va / vb; break;
        case GraphOpCode::Neg: result = -va; break;
        case GraphOpCode::Abs: result = std::abs(va); break;
        case GraphOpCode::Square: result = va * va; break;
        case GraphOpCode::Recip: result = Scalar(1) / va; break;
        case GraphOpCode::Sqrt: result = std::sqrt(va); break;
        case GraphOpCode::Exp: result = std::exp(va); break;
        case GraphOpCode::Log: result = std::log(va); break;
        case GraphOpCode::Sin: result = std::sin(va); break;
        case GraphOpCode::Cos: result = std::cos(va); break;
        case GraphOpCode::Tan: result = std::tan(va); break;
        case GraphOpCode::Asin: result = std::asin(va); break;
        case GraphOpCode::Acos: result = std::acos(va); break;
        case GraphOpCode::Atan: result = std::atan(va); break;
        case GraphOpCode::Sinh: result = std::sinh(va); break;
        case GraphOpCode::Cosh: result = std::cosh(va); break;
        case GraphOpCode::Tanh: result = std::tanh(va); break;
        case GraphOpCode::Pow: result = std::pow(va, vb); break;
        case GraphOpCode::Min: result = (std::min)(va, vb); break;
        case GraphOpCode::Max: result = (std::max)(va, vb); break;
        case GraphOpCode::Mod:
        case GraphOpCode::Fmod: result = std::fmod(va, vb); break;
        case GraphOpCode::Atan2: result = std::atan2(va, vb); break;
        case GraphOpCode::Fdim: result = std::fdim(va, vb); break;
        case GraphOpCode::Floor: result = std::floor(va); break;
        case GraphOpCode::Ceil: result = std::ceil(va); break;
        case GraphOpCode::Cbrt: result = std::cbrt(va); break;
        case GraphOpCode::Erf: result = std::erf(va); break;
        case GraphOpCode::Erfc: result = std::erfc(va); break;
        case GraphOpCode::Expm1: result = std::expm1(va); break;
        case GraphOpCode::Log1p: result = std::log1p(va); break;
        case GraphOpCode::Log10: result = std::log10(va); break;
        case GraphOpCode::Log2: result = std::log2(va); break;
        case GraphOpCode::Asinh: result = std::asinh(va); break;
        case GraphOpCode::Acosh: result = std::acosh(va); break;
        case GraphOpCode::Atanh: result = std::atanh(va); break;
        case GraphOpCode::Exp2: result = std::exp2(va); break;
        case GraphOpCode::Trunc: result = std::trunc(va); break;
        case GraphOpCode::Round: result = std::round(va); break;
        case GraphOpCode::Tgamma: result = std::tgamma(va); break;
        case GraphOpCode::Lgamma: result = std::lgamma(va); break;
        case GraphOpCode::Digamma: result = xad::digamma(va); break;
        case GraphOpCode::Rint: result = std::rint(va); break;
        case GraphOpCode::Nearbyint: result = std::nearbyint(va); break;
        case GraphOpCode::Logb: result = std::logb(va); break;
        case GraphOpCode::Remainder: result = std::remainder(va, vb); break;
        case GraphOpCode::Remquo:
        {
            int quo;
            result = std::remquo(va, vb, &quo);
            // Store quotient in operand_c if needed
            break;
        }
        case GraphOpCode::Hypot: result = std::hypot(va, vb); break;
        case GraphOpCode::Nextafter: result = std::nextafter(va, vb); break;
        case GraphOpCode::Ldexp: result = std::ldexp(va, static_cast<int>(imm)); break;
        case GraphOpCode::Frexp:
        {
            int exp;
            result = std::frexp(va, &exp);
            // Store exponent somewhere if needed
            break;
        }
        case GraphOpCode::Modf:
        {
            Scalar intpart;
            result = std::modf(va, &intpart);
            // Store integer part somewhere if needed
            break;
        }
        case GraphOpCode::Copysign: result = std::copysign(va, vb); break;
        case GraphOpCode::SmoothAbs:
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
        case GraphOpCode::CmpLT: result = (va < vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::CmpLE: result = (va <= vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::CmpGT: result = (va > vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::CmpGE: result = (va >= vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::CmpEQ: result = (va == vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::CmpNE: result = (va != vb) ? Scalar(1) : Scalar(0); break;
        case GraphOpCode::If:
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
void GraphInterpreter<Scalar>::propagateAdjoint(uint32_t nodeId)
{
    const Graph& graph = *impl_->graph;
    std::vector<Scalar>& nodeValues = impl_->nodeValues;
    std::vector<Scalar>& nodeAdjoints = impl_->nodeAdjoints;
    Scalar adj = nodeAdjoints[nodeId];
    if (adj == Scalar(0)) return;

    const auto& node = graph.nodes[nodeId];
    GraphOpCode op = static_cast<GraphOpCode>(node.op);
    uint32_t a = node.a;
    uint32_t b = node.b;

    Scalar va = (a < nodeValues.size()) ? nodeValues[a] : Scalar(0);
    Scalar vb = (b < nodeValues.size()) ? nodeValues[b] : Scalar(0);
    Scalar vResult = nodeValues[nodeId];

    switch (op)
    {
        case GraphOpCode::Input:
        case GraphOpCode::Constant:
            break;
        case GraphOpCode::Add:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] += adj;
            break;
        case GraphOpCode::Sub:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj;
            break;
        case GraphOpCode::Mul:
            nodeAdjoints[a] += adj * vb;
            nodeAdjoints[b] += adj * va;
            break;
        case GraphOpCode::Div:
            nodeAdjoints[a] += adj / vb;
            nodeAdjoints[b] -= adj * va / (vb * vb);
            break;
        case GraphOpCode::Neg:
            nodeAdjoints[a] -= adj;
            break;
        case GraphOpCode::Abs:
            // Match XAD's derivative: (a > 0) - (a < 0), which is 0 at a=0
            nodeAdjoints[a] += adj * ((va > Scalar(0)) ? Scalar(1) : ((va < Scalar(0)) ? Scalar(-1) : Scalar(0)));
            break;
        case GraphOpCode::Square:
            nodeAdjoints[a] += adj * Scalar(2) * va;
            break;
        case GraphOpCode::Recip:
            nodeAdjoints[a] -= adj / (va * va);
            break;
        case GraphOpCode::Sqrt:
            nodeAdjoints[a] += adj / (Scalar(2) * vResult);
            break;
        case GraphOpCode::Exp:
            nodeAdjoints[a] += adj * vResult;
            break;
        case GraphOpCode::Log:
            nodeAdjoints[a] += adj / va;
            break;
        case GraphOpCode::Sin:
            nodeAdjoints[a] += adj * std::cos(va);
            break;
        case GraphOpCode::Cos:
            nodeAdjoints[a] -= adj * std::sin(va);
            break;
        case GraphOpCode::Tan:
        {
            Scalar cosv = std::cos(va);
            nodeAdjoints[a] += adj / (cosv * cosv);
        }
        break;
        case GraphOpCode::Asin:
            nodeAdjoints[a] += adj / std::sqrt(Scalar(1) - va * va);
            break;
        case GraphOpCode::Acos:
            nodeAdjoints[a] -= adj / std::sqrt(Scalar(1) - va * va);
            break;
        case GraphOpCode::Atan:
            nodeAdjoints[a] += adj / (Scalar(1) + va * va);
            break;
        case GraphOpCode::Sinh:
            nodeAdjoints[a] += adj * std::cosh(va);
            break;
        case GraphOpCode::Cosh:
            nodeAdjoints[a] += adj * std::sinh(va);
            break;
        case GraphOpCode::Tanh:
        {
            Scalar t = std::tanh(va);
            nodeAdjoints[a] += adj * (Scalar(1) - t * t);
        }
        break;
        case GraphOpCode::Pow:
            if (vb != Scalar(0))  // x^0 is 1 everywhere, but 0 * 0^-1 is NaN
                nodeAdjoints[a] += adj * vb * std::pow(va, vb - Scalar(1));
            if (va > Scalar(0))
                nodeAdjoints[b] += adj * vResult * std::log(va);
            break;
        case GraphOpCode::Min:
        case GraphOpCode::Max:
            if (va == vb)
            {
                nodeAdjoints[a] += adj * Scalar(0.5);
                nodeAdjoints[b] += adj * Scalar(0.5);
            }
            else  // to the result: va unless vb is beyond it, also with a NaN operand
                nodeAdjoints[(op == GraphOpCode::Min ? vb < va : va < vb) ? b : a] += adj;
            break;
        case GraphOpCode::Mod:
        case GraphOpCode::Fmod:
        case GraphOpCode::Remainder:
        case GraphOpCode::Remquo:
            nodeAdjoints[a] += adj;
            nodeAdjoints[b] -= adj * detail::remainder_quotient(va, vb, vResult);
            break;
        case GraphOpCode::Fdim:
            // 0 at a tie; !(va <= vb) also takes a NaN operand, as the tape does
            if (!(va <= vb))
            {
                nodeAdjoints[a] += adj;
                nodeAdjoints[b] -= adj;
            }
            break;
        case GraphOpCode::Atan2:
        {
            Scalar denom = va * va + vb * vb;
            nodeAdjoints[a] += adj * vb / denom;
            nodeAdjoints[b] -= adj * va / denom;
        }
        break;
        case GraphOpCode::Floor:
        case GraphOpCode::Ceil:
            break;
        case GraphOpCode::Cbrt:
            nodeAdjoints[a] += adj / (Scalar(3) * vResult * vResult);
            break;
        case GraphOpCode::Erf:
            nodeAdjoints[a] += adj * invSqrtPi() * std::exp(-va * va);
            break;
        case GraphOpCode::Erfc:
            nodeAdjoints[a] -= adj * invSqrtPi() * std::exp(-va * va);
            break;
        case GraphOpCode::Expm1:
            nodeAdjoints[a] += adj * std::exp(va);
            break;
        case GraphOpCode::Log1p:
            nodeAdjoints[a] += adj / (Scalar(1) + va);
            break;
        case GraphOpCode::Log10:
            nodeAdjoints[a] += adj / (va * std::log(Scalar(10)));
            break;
        case GraphOpCode::Log2:
            nodeAdjoints[a] += adj / (va * std::log(Scalar(2)));
            break;
        case GraphOpCode::Asinh:
            nodeAdjoints[a] += adj / std::sqrt(va * va + Scalar(1));
            break;
        case GraphOpCode::Acosh:
            nodeAdjoints[a] += adj / std::sqrt(va * va - Scalar(1));
            break;
        case GraphOpCode::Atanh:
            nodeAdjoints[a] += adj / (Scalar(1) - va * va);
            break;
        case GraphOpCode::Exp2:
            nodeAdjoints[a] += adj * std::log(Scalar(2)) * vResult;
            break;
        case GraphOpCode::Trunc:
        case GraphOpCode::Round:
        case GraphOpCode::Rint:
        case GraphOpCode::Nearbyint:
        case GraphOpCode::Logb:
            // Zero derivative
            break;
        case GraphOpCode::Tgamma:
            nodeAdjoints[a] += adj * vResult * xad::digamma(va);
            break;
        case GraphOpCode::Lgamma:
            nodeAdjoints[a] += adj * xad::digamma(va);
            break;
        case GraphOpCode::Digamma:
            nodeAdjoints[a] += adj * detail::trigamma(va);
            break;
        case GraphOpCode::Hypot:
            nodeAdjoints[a] += adj * va / vResult;
            nodeAdjoints[b] += adj * vb / vResult;
            break;
        case GraphOpCode::Nextafter:
            nodeAdjoints[a] += adj;
            // Second operand has zero derivative
            break;
        case GraphOpCode::Ldexp:
        {
            double imm = node.imm;
            int exp = static_cast<int>(imm);
            nodeAdjoints[a] += adj * std::ldexp(Scalar(1), exp);
        }
        break;
        case GraphOpCode::Frexp:
        {
            // Derivative is 2^-exp, but we need to recompute frexp
            int exp = 0;
            std::frexp(va, &exp);
            nodeAdjoints[a] += adj * std::ldexp(Scalar(1), -exp);
        }
        break;
        case GraphOpCode::Modf:
            // Derivative of fractional part is 1
            nodeAdjoints[a] += adj;
            break;
        case GraphOpCode::Copysign:
            // a or -a by sign bits; d/db = 0
            nodeAdjoints[a] += (std::signbit(va) == std::signbit(vb)) ? adj : -adj;
            break;
        case GraphOpCode::SmoothAbs:
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
        case GraphOpCode::CmpLT:
        case GraphOpCode::CmpLE:
        case GraphOpCode::CmpGT:
        case GraphOpCode::CmpGE:
        case GraphOpCode::CmpEQ:
        case GraphOpCode::CmpNE:
            break;
        case GraphOpCode::If:
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
#ifdef _MSC_VER
#pragma warning(disable : 4251 4275)
#endif
template class XAD_TEMPLATE_EXPORT GraphInterpreter<float>;
template class XAD_TEMPLATE_EXPORT GraphInterpreter<double>;

}  // namespace xad

#endif  // XAD_ENABLE_GRAPH



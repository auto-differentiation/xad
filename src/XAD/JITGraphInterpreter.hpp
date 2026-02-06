/**
 *
 *   Reference JITBackend implementation that interprets a JITGraph.
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

#ifdef XAD_ENABLE_JIT

#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>
#include <cstddef>
#include <memory>

namespace xad
{

/**
 * @brief Reference JITBackend implementation that interprets a JITGraph.
 *
 * This is a simple interpreter-based backend that evaluates the computation
 * graph node by node. It serves as a reference implementation and fallback
 * when no native code generation backend is available.
 *
 * The template parameter Scalar specifies the floating-point type used for
 * computation (typically float or double).
 */
template <class Scalar>
class JITGraphInterpreter : public JITBackend<Scalar>
{
  public:
    JITGraphInterpreter();
    ~JITGraphInterpreter() override;

    void compile(const JITGraph& graph) override;
    void reset() override;

    std::size_t vectorWidth() const override { return 1; }
    std::size_t numInputs() const override;
    std::size_t numOutputs() const override;

    void setInput(std::size_t inputIndex, const Scalar* values) override;
    void forward(Scalar* outputs) override;
    void forwardAndBackward(Scalar* outputs, Scalar* inputGradients) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static Scalar invSqrtPi();
    void evaluateNode(uint32_t nodeId);
    void propagateAdjoint(uint32_t nodeId);
};

// Declare external explicit instantiations
extern template class JITGraphInterpreter<float>;
extern template class JITGraphInterpreter<double>;

}  // namespace xad

#endif  // XAD_ENABLE_JIT

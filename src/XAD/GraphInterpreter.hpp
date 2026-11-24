/**
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

#pragma once

#include <XAD/Config.hpp>
#include <XAD/Export.hpp>

#ifdef XAD_ENABLE_GRAPH

#include <XAD/GraphBackendInterface.hpp>
#include <XAD/Graph.hpp>
#include <cstddef>
#include <memory>

namespace xad
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251 4275)  // members and base of the exported class need no dll interface
#endif

/**
 * @brief Reference GraphBackend implementation that interprets a Graph.
 *
 * This is a simple interpreter-based backend that evaluates the computation
 * graph node by node. It serves as a reference implementation and fallback
 * when no native code generation backend is available.
 *
 * The template parameter Scalar specifies the floating-point type used for
 * computation (typically float or double).
 */
template <class Scalar>
class GraphInterpreter : public GraphBackend<Scalar>
{
  public:
    GraphInterpreter();
    ~GraphInterpreter() override;

    void compile(const Graph& graph) override;
    void reset() override;

    std::size_t vectorWidth() const override { return 1; }
    std::size_t numInputs() const override;
    std::size_t numOutputs() const override;

    void setInput(std::size_t inputIndex, const Scalar* values) override;
    void forward(Scalar* outputs) override;
    void forwardAndBackward(Scalar* outputs, Scalar* inputGradients) override;
    void forwardAndBackwardSeeded(Scalar* outputs, const Scalar* outputAdjoints,
                                  Scalar* inputGradients) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static Scalar invSqrtPi();
    // the interpreter's hot functions start on a cache line, so that their speed does not depend
    // on the code linked before them
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(64)))
#endif
    void evaluateNode(uint32_t nodeId);
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(64)))
#endif
    void propagateAdjoint(uint32_t nodeId);
};

// Declare external explicit instantiations
extern template class XAD_TEMPLATE_API GraphInterpreter<float>;
extern template class XAD_TEMPLATE_API GraphInterpreter<double>;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace xad

#endif  // XAD_ENABLE_GRAPH

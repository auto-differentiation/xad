/**
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

#pragma once

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_JIT

#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>
#include <cstddef>
#include <vector>

namespace xad
{

class JITGraphInterpreter : public JITBackend
{
  public:

    void compile(const JITGraph& graph) override;

    void forward(const JITGraph& graph,
                 const double* inputs, std::size_t numInputs,
                 double* outputs, std::size_t numOutputs) override;

    void forwardAndBackward(const JITGraph& graph,
                            const double* inputs, std::size_t numInputs,
                            const double* outputAdjoints, std::size_t numOutputs,
                            double* outputs,
                            double* inputAdjoints) override;

    void reset() override;

  private:
    std::vector<double> nodeValues_;
    std::vector<double> nodeAdjoints_;

    static double invSqrtPi();
    void evaluateNode(const JITGraph& graph, uint32_t nodeId);
    void propagateAdjoint(const JITGraph& graph, uint32_t nodeId);
};

}  // namespace xad

#endif  // XAD_ENABLE_JIT

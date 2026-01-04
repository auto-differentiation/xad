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
#include <memory>

namespace xad
{

class JITGraphInterpreter : public JITBackend
{
  public:
    JITGraphInterpreter();
    ~JITGraphInterpreter() override;

    void compile(const JITGraph& graph) override;
    void reset() override;

    std::size_t vectorWidth() const override { return 1; }
    std::size_t numInputs() const override;
    std::size_t numOutputs() const override;

    void setInput(std::size_t inputIndex, const double* values) override;
    void forward(double* outputs) override;
    void forwardAndBackward(double* outputs, double* inputGradients) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static double invSqrtPi();
    void evaluateNode(uint32_t nodeId);
    void propagateAdjoint(uint32_t nodeId);
};

}  // namespace xad

#endif  // XAD_ENABLE_JIT

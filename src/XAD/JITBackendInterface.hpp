/**
 *
 *   Abstract interface for JIT compilation backends.
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

#include <XAD/JITGraph.hpp>
#include <cstddef>

namespace xad
{

/**
 * @brief Abstract interface for JIT compilation backends.
 *
 * This interface allows plugging in different execution backends for the
 * JIT-compiled computation graphs. Backends can range from simple interpreters
 * to native code generators or GPU executors.
 *
 * Some backends support evaluating multiple input sets in parallel (SIMD).
 * The vectorWidth() method returns how many parallel evaluations are performed:
 * 1 for scalar backends, or more for SIMD backends (e.g., 4 for AVX2).
 *
 * See JITGraphInterpreter for a reference implementation.
 */
class JITBackend
{
  public:
    virtual ~JITBackend() = default;

    /// Compile the computation graph for execution.
    virtual void compile(const JITGraph& graph) = 0;

    /// Reset/clear any compiled state.
    virtual void reset() = 0;

    /// Get the number of parallel evaluations per execution (1 for scalar).
    virtual std::size_t vectorWidth() const = 0;

    /// Get the number of inputs in the compiled graph.
    virtual std::size_t numInputs() const = 0;

    /// Get the number of outputs in the compiled graph.
    virtual std::size_t numOutputs() const = 0;

    /// Set input values for an input variable (vectorWidth() values).
    virtual void setInput(std::size_t inputIndex, const double* values) = 0;

    /// Execute forward pass only. Output array must have numOutputs() * vectorWidth() elements.
    virtual void forward(double* outputs) = 0;

    /// Execute forward and backward passes. Output adjoints are seeded to 1.0.
    virtual void forwardAndBackward(double* outputs, double* inputGradients) = 0;
};

}  // namespace xad

#endif  // XAD_ENABLE_JIT

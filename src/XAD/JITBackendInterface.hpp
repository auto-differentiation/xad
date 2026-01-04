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
 * to native code generators (e.g., Forge) or GPU executors.
 *
 * ## Batch Evaluation Support
 *
 * Some backends support evaluating multiple input sets in a single call for
 * better performance (e.g., via SIMD vectorization). The vectorWidth() method
 * returns how many evaluations the backend performs per execution:
 * - Scalar backends return 1
 * - SIMD backends return the number of parallel evaluations (e.g., 4 for AVX2)
 *
 * All input/output arrays are sized as count * vectorWidth() elements.
 *
 * ## Implementing a Custom Backend
 *
 * To create a custom backend, inherit from JITBackend and implement all
 * pure virtual methods:
 *
 * 1. **compile()**: Called once after graph recording is complete. Use this to:
 *    - Translate JITGraph opcodes to your target representation
 *    - Perform optimizations (CSE, constant folding, etc.)
 *    - Generate native code or prepare execution structures
 *    - Store the graph reference for subsequent execution
 *
 * 2. **vectorWidth()**: Return the number of parallel evaluations (1 for scalar)
 *
 * 3. **numInputs()** / **numOutputs()**: Return counts from compiled graph
 *
 * 4. **setInput()**: Set input values before execution. Called once per
 *    input, with vectorWidth() values per call.
 *
 * 5. **forward()**: Execute forward pass only. Writes vectorWidth() values
 *    per output.
 *
 * 6. **forwardAndBackward()**: Execute both forward and backward passes in
 *    one call. Returns both outputs and input gradients.
 *
 * 7. **reset()**: Clear any compiled state. Called when the graph is cleared
 *    or recompiled.
 *
 * ## Example Usage
 *
 * @code
 * class MyBackend : public xad::JITBackend {
 * public:
 *     void compile(const JITGraph& graph) override { ... }
 *     std::size_t vectorWidth() const override { return 1; }
 *     std::size_t numInputs() const override { return graph_->input_ids.size(); }
 *     std::size_t numOutputs() const override { return graph_->output_ids.size(); }
 *     void setInput(std::size_t idx, const double* v) override { ... }
 *     void forward(double* outputs) override { ... }
 *     void forwardAndBackward(double* outputs, double* grads) override { ... }
 *     void reset() override { ... }
 * };
 *
 * // Use with JITCompiler:
 * xad::JITCompiler<double> jit;
 * // ... record graph ...
 * jit.setBackend(std::make_unique<MyBackend>());
 * jit.compile();
 * jit.setInput(0, inputValues);
 * jit.forwardAndBackward(outputs, gradients);
 * @endcode
 *
 * ## Reference Implementation
 *
 * See JITGraphInterpreter for a reference implementation that interprets
 * the graph directly in C++ without code generation.
 */
class JITBackend
{
  public:
    virtual ~JITBackend() = default;

    //=========================================================================
    // Compilation
    //=========================================================================

    /**
     * @brief Compile the computation graph for execution.
     * @param graph The recorded computation graph to compile.
     *
     * Called once after graph recording is complete. Implementations should
     * store the graph reference and prepare any necessary data structures
     * or generate code for subsequent execution calls.
     */
    virtual void compile(const JITGraph& graph) = 0;

    /**
     * @brief Reset/clear any compiled state.
     *
     * Called when the graph is cleared or needs recompilation. Implementations
     * should release any resources allocated during compile().
     */
    virtual void reset() = 0;

    //=========================================================================
    // Query
    //=========================================================================

    /**
     * @brief Get the number of parallel evaluations per execution.
     * @return 1 for scalar backends, or higher for SIMD backends (e.g., 4 for AVX2).
     */
    virtual std::size_t vectorWidth() const = 0;

    /**
     * @brief Get the number of inputs in the compiled graph.
     * @return Number of inputs.
     */
    virtual std::size_t numInputs() const = 0;

    /**
     * @brief Get the number of outputs in the compiled graph.
     * @return Number of outputs.
     */
    virtual std::size_t numOutputs() const = 0;

    //=========================================================================
    // Execution
    //=========================================================================

    /**
     * @brief Set input values for an input variable.
     * @param inputIndex Index of the input (0 to numInputs()-1).
     * @param values Array of vectorWidth() doubles containing the input values.
     *
     * Must be called for each input before forward() or forwardAndBackward().
     * For scalar backends (vectorWidth() == 1), pass a pointer to a single double.
     * For SIMD backends, pass an array with vectorWidth() values for parallel evaluation.
     */
    virtual void setInput(std::size_t inputIndex, const double* values) = 0;

    /**
     * @brief Execute forward pass only.
     * @param outputs Array of numOutputs() * vectorWidth() doubles.
     *        Layout: [out0_v0, out0_v1, ..., out1_v0, out1_v1, ...]
     *        where v0, v1, ... are the parallel evaluations.
     *
     * Computes output values from the previously set input values.
     */
    virtual void forward(double* outputs) = 0;

    /**
     * @brief Execute combined forward and backward passes.
     * @param outputs Array of numOutputs() * vectorWidth() doubles.
     *        Layout: [out0_v0, out0_v1, ..., out1_v0, out1_v1, ...]
     * @param inputGradients Array of numInputs() * vectorWidth() doubles.
     *        Layout: [in0_v0, in0_v1, ..., in1_v0, in1_v1, ...]
     *        where v0, v1, ... are the parallel evaluations.
     *
     * Computes both output values and input gradients in a single pass.
     * This combined method enables backends to optimize by reusing
     * intermediate values from the forward pass in the backward pass.
     *
     * The backward pass computes gradients of outputs with respect to inputs
     * using reverse-mode automatic differentiation. Output adjoints are
     * implicitly seeded to 1.0.
     */
    virtual void forwardAndBackward(double* outputs, double* inputGradients) = 0;
};

}  // namespace xad

#endif  // XAD_ENABLE_JIT

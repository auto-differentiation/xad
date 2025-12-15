#pragma once

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
 * ## Implementing a Custom Backend
 *
 * To create a custom backend, inherit from IJITBackend and implement all
 * pure virtual methods:
 *
 * 1. **compile()**: Called once after graph recording is complete. Use this to:
 *    - Translate JITGraph opcodes to your target representation
 *    - Perform optimizations (CSE, constant folding, etc.)
 *    - Generate native code or prepare execution structures
 *
 * 2. **forward()**: Execute the forward pass only. Read input values, evaluate
 *    the graph, and write output values. Called for value-only evaluation.
 *
 * 3. **forwardAndBackward()**: Execute both forward and backward (adjoint) passes.
 *    This combined method allows backends to optimize by:
 *    - Reusing intermediate values from forward pass in backward pass
 *    - Fusing operations across both passes
 *    - Avoiding redundant memory allocations
 *
 * 4. **reset()**: Clear any compiled state. Called when the graph is cleared
 *    or recompiled.
 *
 * ## Example Usage
 *
 * @code
 * class MyBackend : public xad::IJITBackend {
 * public:
 *     void compile(const JITGraph& graph) override { ... }
 *     void forward(...) override { ... }
 *     void forwardAndBackward(...) override { ... }
 *     void reset() override { ... }
 * };
 *
 * // Use with JITCompiler:
 * auto backend = std::unique_ptr<IJITBackend>(new MyBackend());
 * xad::JITCompiler<double> jit(std::move(backend));
 * @endcode
 *
 * ## Reference Implementation
 *
 * See JITGraphInterpreter for a reference implementation that interprets
 * the graph directly in C++ without code generation.
 */
class IJITBackend
{
  public:
    virtual ~IJITBackend() = default;

    /**
     * @brief Compile the computation graph for execution.
     * @param graph The recorded computation graph to compile.
     *
     * Called once after graph recording is complete. Implementations should
     * prepare any necessary data structures or generate code for subsequent
     * forward() and forwardAndBackward() calls.
     */
    virtual void compile(const JITGraph& graph) = 0;

    /**
     * @brief Execute forward pass only (compute outputs from inputs).
     * @param graph The computation graph.
     * @param inputs Pointer to input values array.
     * @param numInputs Number of input values (must match graph.input_ids.size()).
     * @param outputs Pointer to output values array (caller-allocated).
     * @param numOutputs Number of output values (must match graph.output_ids.size()).
     */
    virtual void forward(const JITGraph& graph,
                         const double* inputs, std::size_t numInputs,
                         double* outputs, std::size_t numOutputs) = 0;

    /**
     * @brief Execute combined forward and backward (adjoint) passes.
     * @param graph The computation graph.
     * @param inputs Pointer to input values array.
     * @param numInputs Number of input values.
     * @param outputAdjoints Pointer to output adjoint (gradient) seeds.
     * @param numOutputs Number of outputs.
     * @param outputs Pointer to output values array (caller-allocated).
     * @param inputAdjoints Pointer to input adjoints array (caller-allocated).
     *
     * This combined method enables backends to optimize the forward+backward
     * computation, for example by reusing intermediate values or fusing
     * operations. The backward pass computes gradients of outputs with respect
     * to inputs using reverse-mode automatic differentiation.
     */
    virtual void forwardAndBackward(const JITGraph& graph,
                                    const double* inputs, std::size_t numInputs,
                                    const double* outputAdjoints, std::size_t numOutputs,
                                    double* outputs,
                                    double* inputAdjoints) = 0;

    /**
     * @brief Reset/clear any compiled state.
     *
     * Called when the graph is cleared or needs recompilation. Implementations
     * should release any resources allocated during compile().
     */
    virtual void reset() = 0;
};

}  // namespace xad

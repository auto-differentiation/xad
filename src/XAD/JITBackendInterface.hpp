#pragma once

#include <XAD/JITGraph.hpp>
#include <cstddef>

namespace xad
{

// Backend interface for JIT compilation
// Implementations: JITGraphInterpreter, ForgeBackend
class IJITBackend
{
  public:
    virtual ~IJITBackend() = default;

    virtual void compile(const JITGraph& graph) = 0;

    // Forward pass only (outputs only, no gradients)
    virtual void forward(const JITGraph& graph,
                         const double* inputs, std::size_t numInputs,
                         double* outputs, std::size_t numOutputs) = 0;

    // Combined forward + backward pass (outputs + gradients in one execution)
    // This is the efficient path for backends like Forge that compute both together
    virtual void forwardAndBackward(const JITGraph& graph,
                                    const double* inputs, std::size_t numInputs,
                                    const double* outputAdjoints, std::size_t numOutputs,
                                    double* outputs,
                                    double* inputAdjoints) = 0;

    virtual void reset() = 0;
};

}  // namespace xad

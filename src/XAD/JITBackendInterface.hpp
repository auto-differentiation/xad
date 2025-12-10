#pragma once

#include <XAD/JITGraph.hpp>
#include <cstddef>

namespace xad
{

// Backend interface for JIT compilation
// Implementations: JITGraphInterpreter, JITForgeBackend
class IJITBackend
{
  public:
    virtual ~IJITBackend() = default;

    virtual void compile(const JITGraph& graph) = 0;

    virtual void forward(const JITGraph& graph,
                         const double* inputs, std::size_t numInputs,
                         double* outputs, std::size_t numOutputs) = 0;

    virtual void computeAdjoints(const JITGraph& graph,
                                 const double* inputValues, std::size_t numInputs,
                                 const double* outputAdjoints, std::size_t numOutputs,
                                 double* inputAdjoints) = 0;

    virtual void reset() = 0;
};

}  // namespace xad

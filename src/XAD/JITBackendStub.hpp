#pragma once

#include <XAD/JITGraph.hpp>
#include <cstddef>
#include <cstring>

namespace xad
{

class JITBackendStub
{
  public:
    JITBackendStub() = default;
    ~JITBackendStub() = default;
    JITBackendStub(JITBackendStub&&) noexcept = default;
    JITBackendStub& operator=(JITBackendStub&&) noexcept = default;
    JITBackendStub(const JITBackendStub&) = default;
    JITBackendStub& operator=(const JITBackendStub&) = default;

    void compile(const JITGraph& graph)
    {
        // Stub: nothing to compile
        (void)graph;
    }

    void forward(const JITGraph& graph,
                 const double* inputs, std::size_t numInputs,
                 double* outputs, std::size_t numOutputs)
    {
        // Stub: zero all outputs
        (void)graph;
        (void)inputs;
        (void)numInputs;
        if (outputs && numOutputs > 0)
            std::memset(outputs, 0, numOutputs * sizeof(double));
    }

    void forwardAndBackward(const JITGraph& graph,
                            const double* inputs, std::size_t numInputs,
                            const double* outputAdjoints, std::size_t numOutputs,
                            double* outputs,
                            double* inputAdjoints)
    {
        // Stub: zero all outputs and input adjoints
        (void)graph;
        (void)inputs;
        (void)outputAdjoints;
        if (outputs && numOutputs > 0)
            std::memset(outputs, 0, numOutputs * sizeof(double));
        if (inputAdjoints && numInputs > 0)
            std::memset(inputAdjoints, 0, numInputs * sizeof(double));
    }

    void computeAdjoints(const JITGraph& graph,
                         const double* inputValues, std::size_t numInputs,
                         const double* outputAdjoints, std::size_t numOutputs,
                         double* inputAdjoints)
    {
        // Stub: zero all input adjoints
        (void)graph;
        (void)inputValues;
        (void)outputAdjoints;
        (void)numOutputs;
        if (inputAdjoints && numInputs > 0)
            std::memset(inputAdjoints, 0, numInputs * sizeof(double));
    }

    void reset()
    {
        // Stub: nothing to reset
    }
};

}  // namespace xad

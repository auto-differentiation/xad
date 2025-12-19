/**
 *
 *   JIT compiler: record expression graphs and execute them via a backend.
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
#include <XAD/Exceptions.hpp>
#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>
#include <XAD/JITGraphInterpreter.hpp>
#include <XAD/Macros.hpp>
#include <XAD/Tape.hpp>
#include <XAD/Traits.hpp>
#include <complex>
#include <memory>
#include <vector>

namespace xad
{

template <class Scalar, std::size_t M>
struct AReal;

template <class Real, std::size_t N = 1>
class JITCompiler
{
  public:
    typedef unsigned int size_type;
    typedef unsigned int slot_type;
    typedef slot_type position_type;
    typedef AReal<Real, N> active_type;
    typedef Real value_type;
    typedef JITCompiler<Real, N> jit_type;
    typedef typename DerivativesTraits<Real, N>::type derivative_type;
    typedef Tape<Real, N> tape_type;

    // Used across the JIT integration (e.g., ABool / AReal helpers). May not be referenced directly in this TU.
    static constexpr slot_type INVALID_SLOT = slot_type(-1);

    // Default constructor - uses interpreter backend
    explicit JITCompiler(bool activate = true)
        : backend_(std::unique_ptr<JITGraphInterpreter>(new JITGraphInterpreter()))
    {
        if (activate)
        {
            // Deactivate any active tape - JIT requires no tape to be active
            tape_type::deactivateAll();
            setActive(this);
        }
    }

    // Constructor with custom backend
    explicit JITCompiler(std::unique_ptr<IJITBackend> backend, bool activate = true)
        : backend_(std::move(backend))
    {
        if (activate)
        {
            // Deactivate any active tape - JIT requires no tape to be active
            tape_type::deactivateAll();
            setActive(this);
        }
    }

    ~JITCompiler() { deactivate(); }

    /// Set or replace the JIT backend.
    /// Resets any compiled state when the backend is changed.
    void setBackend(std::unique_ptr<IJITBackend> backend)
    {
        backend_ = std::move(backend);
        if (backend_)
            backend_->reset();
    }

    JITCompiler(JITCompiler&& other) noexcept
        : graph_(std::move(other.graph_)),
          backend_(std::move(other.backend_)),
          inputValues_(std::move(other.inputValues_)),
          derivatives_(std::move(other.derivatives_))
    {
        if (other.isActive())
        {
            other.deactivate();
            // `setActive()` can throw if another JIT is active. Here we know `other` was active and we
            // just deactivated it, so we can safely update the active pointer without risking a throw.
            active_jit_ = this;
        }
    }

    JITCompiler& operator=(JITCompiler&& other) noexcept
    {
        if (this != &other)
        {
            deactivate();
            graph_ = std::move(other.graph_);
            backend_ = std::move(other.backend_);
            inputValues_ = std::move(other.inputValues_);
            derivatives_ = std::move(other.derivatives_);
            if (other.isActive())
            {
                other.deactivate();
                // See move-ctor comment: avoid calling throwing `setActive()` inside a noexcept function.
                active_jit_ = this;
            }
        }
        return *this;
    }

    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;

    XAD_INLINE void activate() { setActive(this); }

    XAD_INLINE void deactivate()
    {
        if (active_jit_ == this)
            active_jit_ = nullptr;
    }

    XAD_INLINE bool isActive() const { return active_jit_ == this; }
    XAD_INLINE static JITCompiler* getActive() { return active_jit_; }

    XAD_INLINE static void setActive(JITCompiler* j)
    {
        if (active_jit_ != nullptr)
            throw OutOfRange("JIT Compiler already active");
        active_jit_ = j;
    }

    XAD_INLINE static void deactivateAll() { active_jit_ = nullptr; }

    const JITGraph& getGraph() const { return graph_; }
    JITGraph& getGraph() { return graph_; }

    void newRecording()
    {
        std::size_t numInputs = inputValues_.size();
        graph_.clear();
        derivatives_.clear();
        if (backend_)
            backend_->reset();
        for (std::size_t i = 0; i < numInputs; ++i)
            graph_.addInput();
    }

    XAD_INLINE void registerInput(active_type& inp)
    {
        if (!inp.shouldRecord())
        {
            inp.slot_ = graph_.addInput();
            inputValues_.push_back(&inp.value());
        }
    }

    XAD_INLINE void registerOutput(active_type& outp)
    {
        if (outp.shouldRecord())
            graph_.markOutput(outp.slot_);
    }

    template <class Inner>
    XAD_INLINE void registerInputs(std::vector<Inner>& v)
    {
        registerInputs(v.begin(), v.end());
    }

    template <class It>
    XAD_INLINE void registerInputs(It first, It last)
    {
        while (first != last)
            registerInput(*first++);
    }

    template <class Inner>
    XAD_INLINE void registerOutputs(std::vector<Inner>& v)
    {
        for (auto& x : v)
            registerOutput(x);
    }

    template <class It>
    XAD_INLINE void registerOutputs(It first, It last)
    {
        while (first != last)
            registerOutput(*first++);
    }

    slot_type registerVariable() { return static_cast<slot_type>(graph_.nodeCount()); }

    uint32_t recordNode(JITOpCode op, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0)
    {
        return graph_.addNode(op, a, b, c);
    }

    uint32_t recordConstant(double value) { return graph_.addConstant(value); }

    /**
     * Compile the recorded graph to native code.
     * Must be called after recording and before forward().
     */
    void compile() { backend_->compile(graph_); }

    /**
     * Execute the compiled kernel with current input values.
     * compile() must be called before the first forward() call.
     */
    void forward(double* outputs, std::size_t numOutputs)
    {
        if (numOutputs != graph_.output_ids.size())
            throw OutOfRange("Output count mismatch");

        std::size_t numInputs = graph_.input_ids.size();
        std::vector<double> inputs(numInputs);
        for (std::size_t i = 0; i < numInputs; ++i)
            inputs[i] = *inputValues_[i];

        backend_->forward(graph_, inputs.data(), numInputs, outputs, numOutputs);
    }

    /**
     * Compute adjoints (gradients) using reverse-mode AD.
     * compile() must be called before this.
     */
    void computeAdjoints()
    {

        std::size_t numInputs = graph_.input_ids.size();
        std::size_t numOutputs = graph_.output_ids.size();

        std::vector<double> inputs(numInputs);
        for (std::size_t i = 0; i < numInputs; ++i)
            inputs[i] = *inputValues_[i];

        std::vector<double> outputAdjoints(numOutputs, 0.0);
        for (std::size_t i = 0; i < numOutputs; ++i)
        {
            uint32_t outId = graph_.output_ids[i];
            if (outId < derivatives_.size())
                outputAdjoints[i] = derivatives_[outId];
        }

        std::vector<double> outputs(numOutputs);
        std::vector<double> inputAdjoints(numInputs);
        backend_->forwardAndBackward(graph_, inputs.data(), numInputs,
                                     outputAdjoints.data(), numOutputs,
                                     outputs.data(), inputAdjoints.data());

        derivatives_.resize(graph_.nodeCount(), derivative_type());
        for (std::size_t i = 0; i < numInputs; ++i)
            derivatives_[graph_.input_ids[i]] = inputAdjoints[i];
    }

    derivative_type& derivative(slot_type s)
    {
        if (s >= derivatives_.size())
            derivatives_.resize(s + 1, derivative_type());
        return derivatives_[s];
    }

    const derivative_type& derivative(slot_type s) const
    {
        if (s < derivatives_.size())
            return derivatives_[s];
        // Return reference to class member for out-of-range slots (thread-safe)
        return zero_;
    }

    derivative_type getDerivative(slot_type s) const
    {
        return (s < derivatives_.size()) ? derivatives_[s] : derivative_type();
    }

    void setDerivative(slot_type s, const derivative_type& d)
    {
        if (s >= derivatives_.size())
            derivatives_.resize(s + 1, derivative_type());
        derivatives_[s] = d;
    }

    void clearDerivatives()
    {
        std::fill(derivatives_.begin(), derivatives_.end(), derivative_type());
    }

    void clearAll()
    {
        graph_.clear();
        inputValues_.clear();
        derivatives_.clear();
        if (backend_)
            backend_->reset();
    }

    std::size_t getMemory() const { return graph_.nodeCount() * 32 + derivatives_.size() * sizeof(derivative_type); }
    position_type getPosition() const { return static_cast<position_type>(graph_.nodeCount()); }

  private:
    static XAD_THREAD_LOCAL JITCompiler* active_jit_;
    JITGraph graph_;
    std::unique_ptr<IJITBackend> backend_;
    std::vector<const Real*> inputValues_;
    std::vector<derivative_type> derivatives_;
    derivative_type zero_ = derivative_type();  // Thread-safe zero for out-of-range derivative access
};

template <class Real, std::size_t N>
XAD_THREAD_LOCAL JITCompiler<Real, N>* JITCompiler<Real, N>::active_jit_ = nullptr;

}  // namespace xad

#endif  // XAD_ENABLE_JIT

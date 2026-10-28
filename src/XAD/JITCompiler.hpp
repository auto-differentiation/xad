/**
 *
 *   JIT compiler: record expression graphs and execute them via a backend.
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

#ifdef XAD_ENABLE_JIT
#include <XAD/Exceptions.hpp>
#include <XAD/Export.hpp>
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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)  // members of the exported JITCompiler need no dll interface
#endif

template <class Real, std::size_t N = 1>
class JITCompiler
{
  public:
    static_assert(N == 1, "JITCompiler only supports scalar mode (N=1).");

    typedef unsigned int size_type;
    typedef unsigned int slot_type;
    typedef slot_type position_type;
    typedef AReal<Real, N> active_type;
    typedef Real value_type;
    typedef JITCompiler<Real, N> jit_type;
    typedef typename DerivativesTraits<Real, N>::type derivative_type;
    typedef Tape<Real, N> tape_type;

    // Default constructor - uses interpreter backend
    explicit JITCompiler(bool activate = true)
        : backend_(std::unique_ptr<JITGraphInterpreter<Real>>(new JITGraphInterpreter<Real>()))
    {
        if (activate)
            activateSuspendingTape();
    }

    // Constructor with custom backend
    explicit JITCompiler(std::unique_ptr<JITBackend<Real>> backend, bool activate = true)
        : backend_(std::move(backend))
    {
        if (activate)
            activateSuspendingTape();
    }

    ~JITCompiler()
    {
        deactivate();
        restoreTape();
    }

    /// Set or replace the JIT backend.
    /// Resets any compiled state when the backend is changed.
    void setBackend(std::unique_ptr<JITBackend<Real>> backend)
    {
        backend_ = std::move(backend);
        if (backend_)
            backend_->reset();
    }

    JITCompiler(JITCompiler&& other) noexcept
        : graph_(std::move(other.graph_)),
          backend_(std::move(other.backend_)),
          inputs_(std::move(other.inputs_)),
          derivatives_(std::move(other.derivatives_))
    {
        takeSuspendedTape(other);
        if (other.isActive())
        {
            other.deactivate();
            // `setActive()` can throw if another JIT is active. Here we know `other` was active and we
            // just deactivated it, so we can safely update the active pointer without risking a throw.
            setActiveUnchecked(this);
        }
    }

    JITCompiler& operator=(JITCompiler&& other) noexcept
    {
        if (this != &other)
        {
            deactivate();
            restoreTape();
            graph_ = std::move(other.graph_);
            backend_ = std::move(other.backend_);
            inputs_ = std::move(other.inputs_);
            derivatives_ = std::move(other.derivatives_);
            takeSuspendedTape(other);
            if (other.isActive())
            {
                other.deactivate();
                // See move-ctor comment: avoid calling throwing `setActive()` inside a noexcept function.
                setActiveUnchecked(this);
            }
        }
        return *this;
    }

    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;

    XAD_INLINE void activate() { setActive(this); }

    XAD_INLINE void deactivate()
    {
        if (isActive())
            deactivateAll();
    }

    XAD_INLINE bool isActive() const { return getActive() == this; }

#ifdef XAD_SHARED_LIBS
    // out of line, see Tape
    static JITCompiler* getActive();
    static void setActive(JITCompiler* j);
    static void deactivateAll();
#else
    XAD_INLINE static JITCompiler* getActive() { return active_jit_; }

    XAD_INLINE static void setActive(JITCompiler* j)
    {
        if (active_jit_ != nullptr)
            throw OutOfRange("JIT Compiler already active");
        active_jit_ = j;
    }

    XAD_INLINE static void deactivateAll() { active_jit_ = nullptr; }
#endif

    const JITGraph& getGraph() const { return graph_; }
    JITGraph& getGraph() { return graph_; }

    void newRecording()
    {
        graph_.clear();
        derivatives_.clear();
        if (backend_)
            backend_->reset();
        // the inputs lead the new graph, so their slots move
        for (std::size_t i = 0; i < inputs_.size(); ++i)
            inputs_[i]->slot_ = jitSlot(graph_.addInput());
    }

    XAD_INLINE void registerInput(active_type& inp)
    {
        if (!inp.shouldRecordJIT())
        {
            inp.slot_ = jitSlot(graph_.addInput());
            inputs_.push_back(&inp);
        }
    }

    XAD_INLINE void registerOutput(active_type& outp)
    {
        uint32_t node;
        if (outp.shouldRecordJIT())
            node = jitNode(outp.slot_);
        else  // independent of the inputs: a constant, leaving a tape variable its slot
        {
            node = graph_.addConstant(outp.value());
            if (!outp.shouldRecord())
                outp.slot_ = jitSlot(node);
        }
        graph_.markOutput(node);
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

    /// Compile the recorded graph. Must be called before execution methods.
    void compile() { backend_->compile(graph_); }

    std::size_t vectorWidth() const { return backend_->vectorWidth(); }
    std::size_t numInputs() const { return backend_->numInputs(); }
    std::size_t numOutputs() const { return backend_->numOutputs(); }

    void setInput(std::size_t inputIndex, const Real* values)
    {
        backend_->setInput(inputIndex, values);
    }

    /// Execute forward pass using registered input pointers.
    void forward(Real* outputs)
    {
        for (std::size_t i = 0; i < inputs_.size(); ++i)
        {
            Real value = inputs_[i]->value();
            backend_->setInput(i, &value);
        }
        backend_->forward(outputs);
    }

    void forwardAndBackward(Real* outputs, Real* inputGradients)
    {
        backend_->forwardAndBackward(outputs, inputGradients);
    }

    /// Compute adjoints using registered input pointers.
    void computeAdjoints()
    {
        std::size_t nInputs = graph_.input_ids.size();
        std::size_t nOutputs = graph_.output_ids.size();

        for (std::size_t i = 0; i < inputs_.size(); ++i)
        {
            Real value = inputs_[i]->value();
            backend_->setInput(i, &value);
        }

        std::vector<Real> outputs(nOutputs);
        std::vector<Real> outputAdjoints(nOutputs);
        for (std::size_t i = 0; i < nOutputs; ++i)
            outputAdjoints[i] = getDerivative(graph_.output_ids[i]);
        std::vector<Real> inputGradients(nInputs);
        backend_->forwardAndBackwardSeeded(outputs.data(), outputAdjoints.data(),
                                           inputGradients.data());

        derivatives_.resize(graph_.nodeCount(), derivative_type());
        for (std::size_t i = 0; i < nInputs; ++i)
            derivatives_[graph_.input_ids[i]] = static_cast<derivative_type>(inputGradients[i]);
    }

    // s: a variable's slot or a graph node
    derivative_type& derivative(slot_type s)
    {
        uint32_t n = jitNode(s);
        if (n >= derivatives_.size())
            derivatives_.resize(n + 1, derivative_type());
        return derivatives_[n];
    }

    const derivative_type& derivative(slot_type s) const
    {
        uint32_t n = jitNode(s);
        if (n < derivatives_.size())
            return derivatives_[n];
        // Return reference to class member for out-of-range slots (thread-safe)
        return zero_;
    }

    derivative_type getDerivative(slot_type s) const
    {
        uint32_t n = jitNode(s);
        return (n < derivatives_.size()) ? derivatives_[n] : derivative_type();
    }

    void setDerivative(slot_type s, const derivative_type& d)
    {
        uint32_t n = jitNode(s);
        if (n >= derivatives_.size())
            derivatives_.resize(n + 1, derivative_type());
        derivatives_[n] = d;
    }

    void clearDerivatives()
    {
        std::fill(derivatives_.begin(), derivatives_.end(), derivative_type());
    }

    void clearAll()
    {
        graph_.clear();
        inputs_.clear();
        derivatives_.clear();
        if (backend_)
            backend_->reset();
    }

    std::size_t getMemory() const { return graph_.nodeCount() * 32 + derivatives_.size() * sizeof(derivative_type); }
    position_type getPosition() const { return static_cast<position_type>(graph_.nodeCount()); }

  private:
    // JIT recording needs the tape off: this JIT reactivates it when it goes
    void activateSuspendingTape()
    {
        setActive(this);
        if (tape_type* t = tape_type::getActive())
        {
            if (t->jitSuspender_)  // deactivated by an earlier JIT and reactivated since
                *t->jitSuspender_ = nullptr;
            t->jitSuspender_ = &suspendedTape_;
            suspendedTape_ = t;
            t->deactivate();
        }
    }

    void takeSuspendedTape(JITCompiler& other) noexcept
    {
        suspendedTape_ = other.suspendedTape_;
        other.suspendedTape_ = nullptr;
        if (suspendedTape_)
            suspendedTape_->jitSuspender_ = &suspendedTape_;
    }

    // unless the tape is gone (it clears suspendedTape_) or something else is active now
    void restoreTape() noexcept
    {
        tape_type* t = suspendedTape_;
        if (!t)
            return;
        t->jitSuspender_ = nullptr;
        suspendedTape_ = nullptr;
        if (getActive() == nullptr && tape_type::getActive() == nullptr)
            t->activate();
    }

#ifdef XAD_SHARED_LIBS
    static void setActiveUnchecked(JITCompiler* j);
#else
    XAD_INLINE static void setActiveUnchecked(JITCompiler* j) { active_jit_ = j; }
    static XAD_THREAD_LOCAL JITCompiler* active_jit_;
#endif
    JITGraph graph_;
    std::unique_ptr<JITBackend<Real>> backend_;
    std::vector<active_type*> inputs_;
    std::vector<derivative_type> derivatives_;
    derivative_type zero_ = derivative_type();  // Thread-safe zero for out-of-range derivative access
    tape_type* suspendedTape_ = nullptr;
};

// Declare external explicit instantiations (mirrors Tape.hpp pattern).
// Definitions are provided in JITCompilerTLS.cpp.
extern template class XAD_TEMPLATE_API JITCompiler<float>;
extern template class XAD_TEMPLATE_API JITCompiler<double>;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace xad

#endif  // XAD_ENABLE_JIT

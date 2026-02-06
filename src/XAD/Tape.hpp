/*******************************************************************************

   Declaration of the tape.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <XAD/Config.hpp>
#include <XAD/Exceptions.hpp>
#include <XAD/Macros.hpp>
#include <XAD/ReusableRange.hpp>
#include <XAD/TapeContainer.hpp>
#include <XAD/Traits.hpp>
#include <XAD/TypeTraits.hpp>
#include <XAD/Vec.hpp>
#include <complex>
#include <list>
#include <stack>
#include <type_traits>
#include <vector>

namespace xad
{

template <class Scalar, std::size_t M>
struct AReal;
template <class Scalar, std::size_t N>
struct ARealDirect;
template <class Scalar, std::size_t N>
struct FReal;
template <class Scalar, std::size_t N>
struct FRealDirect;
template <class>
class CheckpointCallback;

template <class Tape>
class ScopedNestedRecording
{
  public:
    explicit ScopedNestedRecording(Tape* s) : s_(s) { s_->newNestedRecording(); }
    ~ScopedNestedRecording() { s_->endNestedRecording(); }
    void computeAdjoints() { s_->computeAdjoints(); }
    void incrementAdjoint(typename Tape::slot_type slot, const typename Tape::value_type& value)
    {
        s_->incrementAdjoint(slot, value);
    }

    Tape* getTape() { return s_; }

  private:
    ScopedNestedRecording(const ScopedNestedRecording&) = delete;
    ScopedNestedRecording& operator=(const ScopedNestedRecording&) = delete;
    Tape* s_;
};

template <class Real, std::size_t N = 1>
class Tape
{
  public:
    // types
    typedef unsigned int size_type;
    typedef unsigned int slot_type;
    typedef slot_type position_type;
    typedef AReal<Real, N> active_type;
    typedef Real value_type;
    typedef Tape<Real, N> tape_type;
    typedef CheckpointCallback<tape_type>* callback_type;
    typedef typename DerivativesTraits<Real, N>::type derivative_type;

    static constexpr slot_type INVALID_SLOT = slot_type(-1);

    // construct/destruct/assign
    explicit Tape(bool activate = true);
    ~Tape();
    Tape(Tape&&) noexcept;
    Tape& operator=(Tape&&) noexcept;

    // not copyable
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;

    // recording control
    XAD_INLINE void activate() { setActive(this); }

    XAD_INLINE void deactivate()
    {
        if (active_tape_ == this)
            active_tape_ = nullptr;
    }
    XAD_INLINE bool isActive() const { return active_tape_ == this; }
    XAD_INLINE static Tape* getActive() { return active_tape_; }

    XAD_INLINE static void setActive(Tape* t)
    {
        if (active_tape_ != nullptr)
            throw TapeAlreadyActive();
        else
            active_tape_ = t;
    }

    XAD_INLINE static void deactivateAll() { active_tape_ = nullptr; }

    XAD_INLINE void registerInput(active_type& inp)
    {
        if (!inp.shouldRecord())  // already registered
        {
            inp.slot_ = registerVariable();
            pushLhs(inp.slot_);
        }
    }
    XAD_INLINE void registerInput(std::complex<active_type>& inp)
    {
        // the standard explicitly states that such casting is safe
        auto reim_ptr = reinterpret_cast<active_type*>(&inp);
        registerInput(reim_ptr[0]);
        registerInput(reim_ptr[1]);
    }

    XAD_INLINE void registerOutput(active_type& outp)
    {
        if (!outp.shouldRecord())
        {
            outp.slot_ = registerVariable();
            pushLhs(outp.slot_);
        }
    }

    XAD_INLINE void registerOutput(std::complex<active_type>& outp)
    {
        // the standard explicitly states that such casting is safe
        auto reim_ptr = reinterpret_cast<active_type*>(&outp);
        registerOutput(reim_ptr[0]);
        registerOutput(reim_ptr[1]);
    }

    // convience registration API
    template <class Inner>
    XAD_INLINE void registerInputs(std::vector<Inner>& v)
    {
        registerInputs(v.begin(), v.end());
    }

    template <class It>
    XAD_INLINE void registerInputs(It first, It last)
    {
        while (first != last) registerInput(*first++);
    }

    XAD_INLINE void registerOutputs(std::vector<active_type>& v)
    {
        for (auto& x : v) registerOutput(x);
    }

    template <class It>
    XAD_INLINE void registerOutputs(It first, It last)
    {
        while (first != last) registerOutput(*first++);
    }

    // recording and adjoint control
    void newRecording();
    void computeAdjoints();
    void clearAll();

    // derivatives
    void clearDerivatives();
    derivative_type& derivative(slot_type s);
    const derivative_type& derivative(slot_type s) const;

    derivative_type getDerivative(slot_type s) const { return derivative(s); }
    void setDerivative(slot_type s, const derivative_type& d) { derivative(s) = d; }
    void setDerivative(slot_type s, derivative_type&& d) { derivative(s) = std::move(d); }

    // status
    void printStatus() const;
    std::size_t getMemory() const;

    // checkpointing API
    void insertCallback(callback_type cb);
    derivative_type getAndResetOutputAdjoint(slot_type slot);
    void incrementAdjoint(slot_type slot, const Real& x);
    void newNestedRecording();
    void endNestedRecording();

    // checkpoint callback memory management
    void pushCallback(callback_type cb);
    callback_type getLastCallback();
    size_type getNumCallbacks() const;
    bool haveCallbacks() const;
    void popCallback();

    // internal tape recording
    slot_type registerVariable()
    {
        ++currentRec_->numDerivatives_;
#ifdef XAD_TAPE_REUSE_SLOTS
        return registerVariableReuseSlots();
#else
        return registerVariableAtEnd();
#endif
    }

    XAD_INLINE void unregisterVariable(slot_type slot)
    {
#ifndef XAD_TAPE_REUSE_SLOTS
        --currentRec_->numDerivatives_;
        if (slot == currentRec_->iDerivative_ - 1)  // it's at the end of the tape
            --currentRec_->iDerivative_;
#else
        unregisterVariableReuseSlots(slot);
#endif
    }

    XAD_INLINE void pushLhs(slot_type slot)
    {
        assert(slot != INVALID_SLOT);
        statement_.emplace_back(size_type(operations_.size()), slot);
    }

    template <class MulIt, class SlotIt>
    XAD_FORCE_INLINE void pushAll(MulIt multipliers, SlotIt slots, unsigned n)
    {
        operations_.append_n(multipliers, slots, n);
    }

    // capacity
    size_type getNumVariables() const;
    size_type getNumOperations() const;
    size_type getNumStatements() const;

    std::string getReusableSlotsString() const;
    size_type getNumReusableSlotSections() const;
    size_type getNumReusableSlots() const;

    // mark current position in tape, for use with clearDerivativesAfter, resetTo, computeAdjointsTo
    position_type getPosition() const;
    // reset all adjoints after the given position in the tape, leaving the previous ones untouched
    void clearDerivativesAfter(position_type pos);
    // reset the tape to the given position, allowing to re-use the storage after
    void resetTo(position_type pos);
    // roll back the adjoints just to the given position (not to the start)
    void computeAdjointsTo(position_type pos);

  private:
    void computeAdjointsToImpl(position_type pos, position_type start);
    void initDerivatives();
    slot_type registerVariableAtEnd()
    {
        ++currentRec_->iDerivative_;
        currentRec_->maxDerivative_ =
            (std::max)(currentRec_->iDerivative_, currentRec_->maxDerivative_);
        return currentRec_->iDerivative_ - 1;
    }

    static XAD_THREAD_LOCAL Tape* active_tape_;
    typename TapeContainerTraits<Real, slot_type>::operations_type operations_;
    typename TapeContainerTraits<Real, slot_type>::statements_type statement_;
    std::vector<derivative_type> derivatives_;
    typedef std::pair<position_type, CheckpointCallback<Tape>*> chkpt_type;
    std::vector<chkpt_type> checkpoints_;
    std::vector<CheckpointCallback<Tape>*> callbacks_;
#ifdef XAD_TAPE_REUSE_SLOTS
    slot_type registerVariableReuseSlots();
    void unregisterVariableReuseSlots(slot_type slot);
    typedef ReusableRange<slot_type> slot_range_type;
    typedef std::list<slot_range_type> range_list;
    range_list reusable_ranges_;
#endif

    void foldSubrecordings();
    void foldSubrecording();

    struct SubRecording
    {
        explicit SubRecording(Tape<Real, N>* parent)
            : numDerivatives_(),
              iDerivative_(),
              maxDerivative_(),
              statementStartPos_(1),
              opStartPos_(),
              startDerivative_(),
              prevMax_(slot_type(-1)),
#ifdef XAD_TAPE_REUSE_SLOTS
              startRange_(parent->reusable_ranges_.end()),
              latestRange_(parent->reusable_ranges_.end()),
#endif
              derivativesInitialized_(false)
        {
            XAD_UNUSED_VARIABLE(parent);
        }
        slot_type numDerivatives_;
        slot_type iDerivative_;
        slot_type maxDerivative_;
        slot_type statementStartPos_;
        slot_type opStartPos_;
        slot_type startDerivative_;
        slot_type prevMax_;
#ifdef XAD_TAPE_REUSE_SLOTS
        range_list::iterator startRange_;
        range_list::iterator latestRange_;
#endif
        bool derivativesInitialized_;
    };
    std::stack<SubRecording> nestedRecordings_;
    SubRecording* currentRec_;
};

// declare external explicit instantiations
#define XAD_DECLARE_EXTERN_TAPE(type) extern template class Tape<type>;
#define XAD_SINGLE_ARG(...) __VA_ARGS__

// 1st order
XAD_DECLARE_EXTERN_TAPE(float)
XAD_DECLARE_EXTERN_TAPE(double)
// 2nd order
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(AReal<float, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(AReal<double, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(FReal<float, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(FReal<double, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(ARealDirect<float, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(ARealDirect<double, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(FRealDirect<double, 1>))
XAD_DECLARE_EXTERN_TAPE(XAD_SINGLE_ARG(FRealDirect<float, 1>))

#include <XAD/GenerateMode.hpp>

#undef XAD_DECLARE_EXTERN_TAPE
}  // namespace xad

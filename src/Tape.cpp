/*******************************************************************************

   Implementation of the Tape

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

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

#include <XAD/ARealDirect.hpp>
#include <XAD/BinaryOperators.hpp>
#include <XAD/CheckpointCallback.hpp>
#include <XAD/FRealDirect.hpp>
#include <XAD/Literals.hpp>
#include <XAD/Macros.hpp>
#include <XAD/Tape.hpp>
#include <XAD/Traits.hpp>
#include <XAD/UnaryOperators.hpp>

#include <iostream>
#include <numeric>
#include <sstream>

#if 0
#define LOG_DEBUG(msg) std::cout << msg << std::endl;
#else
#define LOG_DEBUG(msg)
#endif

namespace xad
{

template <class T, std::size_t N>
Tape<T, N>::Tape(bool activateNow)
{

    nestedRecordings_.push(SubRecording(this));
    currentRec_ = &nestedRecordings_.top();
    if (activateNow)
        activate();
    statement_.push_back(std::make_pair(size_type(operations_.size()), slot_type(INVALID_SLOT)));
}

template <class T, std::size_t N>
Tape<T, N>::Tape(Tape&& o) noexcept
    : operations_(std::move(o.operations_)),
      statement_(std::move(o.statement_)),
      derivatives_(std::move(o.derivatives_)),
      checkpoints_(std::move(o.checkpoints_)),
      callbacks_(std::move(o.callbacks_)),
#ifdef XAD_TAPE_REUSE_SLOTS
      reusable_ranges_(std::move(o.reusable_ranges_)),
#endif
      nestedRecordings_(std::move(o.nestedRecordings_)),
      currentRec_(o.currentRec_)
{
    if (o.isActive())
        activate();
    else
        deactivate();
}

template <class T, std::size_t N>
Tape<T, N>& Tape<T, N>::operator=(Tape&& o) noexcept
{
    operations_ = std::move(o.operations_);
    statement_ = std::move(o.statement_);
    derivatives_ = std::move(o.derivatives_);
    checkpoints_ = std::move(o.checkpoints_);
    callbacks_ = std::move(o.callbacks_);
#ifdef XAD_TAPE_REUSE_SLOTS
    reusable_ranges_ = std::move(o.reusable_ranges_);
#endif
    nestedRecordings_ = std::move(o.nestedRecordings_);
    currentRec_ = o.currentRec_;
    if (o.isActive())
        activate();
    else
        deactivate();
    return *this;
}

template <class T, std::size_t N>
Tape<T, N>::~Tape()
{
    deactivate();
    for (auto p : callbacks_) delete p;
}

template <class T, std::size_t N>
void Tape<T, N>::clearAll()
{
    operations_.clear();
    statement_.clear();
    derivatives_.clear();
    checkpoints_.clear();
#ifdef XAD_TAPE_REUSE_SLOTS
    reusable_ranges_.clear();
#endif
    while (!nestedRecordings_.empty()) nestedRecordings_.pop();
    statement_.push_back(std::make_pair(size_type(operations_.size()), slot_type(INVALID_SLOT)));
    nestedRecordings_.push(SubRecording(this));
    currentRec_ = &nestedRecordings_.top();
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumVariables() const
{
    return currentRec_->numDerivatives_;
}

#ifdef XAD_TAPE_REUSE_SLOTS
template <class T, std::size_t N>
typename Tape<T, N>::slot_type Tape<T, N>::registerVariableReuseSlots()
{
    // at the end
    if (currentRec_->startRange_ == reusable_ranges_.end())
        return registerVariableAtEnd();

    // insert in a reusable slot
    auto& first_range = *currentRec_->startRange_;
    auto ret = first_range.insert();
    if (first_range.isClosed())
    {
        if (currentRec_->latestRange_ == currentRec_->startRange_)
            currentRec_->latestRange_ = reusable_ranges_.end();
        auto nextRange = currentRec_->startRange_;
        ++nextRange;
        reusable_ranges_.erase(currentRec_->startRange_);
        currentRec_->startRange_ = nextRange;
        currentRec_->latestRange_ = nextRange;
    }
    return ret;
}

template <class T, std::size_t N>
void Tape<T, N>::unregisterVariableReuseSlots(slot_type slot)
{
    --currentRec_->numDerivatives_;
    if (slot == currentRec_->iDerivative_ - 1)  // it's at the end of the tape
    {
        --currentRec_->iDerivative_;
        // if this brings us into a re-usable range, delete the last range and update iDerivative_
        if (currentRec_->startRange_ != reusable_ranges_.end())
        {
            auto& last_range = reusable_ranges_.back();
            if (currentRec_->iDerivative_ == last_range.second())
            {
                currentRec_->iDerivative_ = last_range.first();
                auto it = reusable_ranges_.end();
                --it;
                if (currentRec_->latestRange_ == it)
                    currentRec_->latestRange_ = reusable_ranges_.end();
                if (it == currentRec_->startRange_)
                {
                    reusable_ranges_.pop_back();
                    currentRec_->startRange_ = reusable_ranges_.end();
                }
                else
                    reusable_ranges_.pop_back();
            }
        }
    }
    else  // not at the end of the tape
    {
        // unregistering in the middle of the tape requires inserting the slot into a reusable slot
        // list at the right point
        slot_range_type::ExpandResult status = slot_range_type::FAILED;
        // if element is at start or end of an existing range, expand
        if (currentRec_->startRange_ != reusable_ranges_.end() &&
            currentRec_->latestRange_ != reusable_ranges_.end())
        {
            auto& currentRange = *currentRec_->latestRange_;
            status = currentRange.expand(slot);
        }

        // if expanding failed, look over the other ranges to see if it fits
        if (status == slot_range_type::FAILED)
        {
#ifndef NDEBUG
            // make sure startRange is valid
            bool isValid = currentRec_->startRange_ == reusable_ranges_.end();
            for (auto it = reusable_ranges_.begin(), eit = reusable_ranges_.end(); it != eit; ++it)
                isValid = isValid || currentRec_->startRange_ == it;
            assert(isValid && "startRange is invalid");
#endif
            if (reusable_ranges_.empty() || slot > reusable_ranges_.back().second())
            {
                // insert a new range at the end
                reusable_ranges_.emplace_back(slot, slot + 1);
                currentRec_->latestRange_ = reusable_ranges_.end();
                --currentRec_->latestRange_;
                if (reusable_ranges_.size() == 1)
                    currentRec_->startRange_ = currentRec_->latestRange_;
            }
            else
            {
                // it must fit into another range now - using binary search
                auto it = std::lower_bound(currentRec_->startRange_, reusable_ranges_.end(), slot,
                                           [](const slot_range_type& range, slot_type s)
                                           { return range.second() < s; });
                status = it->expand(slot);
                if (status != slot_range_type::FAILED)
                {
                    currentRec_->latestRange_ = it;
                }
                else
                {
                    // insert a new range
                    currentRec_->latestRange_ = reusable_ranges_.emplace(it, slot, slot + 1);
                    if (it == currentRec_->startRange_)
                        currentRec_->startRange_ = currentRec_->latestRange_;
                }
            }
        }
        // check if ranges can now be merged
        if (status == slot_range_type::START &&
            currentRec_->latestRange_ != currentRec_->startRange_)
        {
            auto it = currentRec_->latestRange_;
            --it;
            if (currentRec_->latestRange_->isJoinableStart(*it))
            {
                // merge two ranges
                currentRec_->latestRange_->joinStart(*it);
                if (it == currentRec_->startRange_)
                    ++currentRec_->startRange_;
                reusable_ranges_.erase(it);
            }
        }
        else if (status == slot_range_type::END)
        {
            auto it = currentRec_->latestRange_;
            ++it;
            if (it != reusable_ranges_.end() && currentRec_->latestRange_->isJoinableEnd(*it))
            {
                currentRec_->latestRange_->joinEnd(*it);
                if (currentRec_->startRange_ == it)
                    ++currentRec_->startRange_;
                reusable_ranges_.erase(it);
            }
        }
    }
}
#endif

template <class T, std::size_t N>
std::string Tape<T, N>::getReusableSlotsString() const
{
#ifdef XAD_TAPE_REUSE_SLOTS
    std::stringstream sstr;
    for (auto& i : reusable_ranges_)
    {
        sstr << i << ", ";
    }
    return sstr.str();
#else
    return "";
#endif
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumReusableSlotSections() const
{
#ifdef XAD_TAPE_REUSE_SLOTS
    return size_type(reusable_ranges_.size());
#else
    return 1U;
#endif
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumReusableSlots() const
{
#ifdef XAD_TAPE_REUSE_SLOTS
    return std::accumulate(reusable_ranges_.begin(), reusable_ranges_.end(), size_type(),
                           [](size_type a, const slot_range_type& b) { return a + b.size(); });
#else
    return 0U;
#endif
}

template <class T, std::size_t N>
void Tape<T, N>::foldSubrecording()
{
    // std::cout << "folding down from " << nestedRecordings_.size() << "\n";
    auto prev = nestedRecordings_.top();
    nestedRecordings_.pop();
    auto cur = &nestedRecordings_.top();
    currentRec_ = cur;
    // std::cout << "prev.max: " << prev.maxDerivative_
    //     << ", cur max: " << cur->maxDerivative_ << "\n";
    if (derivatives_.size() > cur->maxDerivative_)
        derivatives_.resize(cur->maxDerivative_);
    if (operations_.size() > prev.opStartPos_)
    {
        operations_.resize(prev.opStartPos_);
    }
    if (statement_.size() > prev.statementStartPos_)
        statement_.resize(prev.statementStartPos_);
    // find first element in chkpt that is <= than startpos
    auto it =
        std::lower_bound(checkpoints_.begin(), checkpoints_.end(), prev.statementStartPos_,
                         [=](const chkpt_type& ckpt, slot_type pos) { return ckpt.first < pos; });
    checkpoints_.erase(it, checkpoints_.end());
#ifdef XAD_TAPE_REUSE_SLOTS
    reusable_ranges_.erase(prev.startRange_, reusable_ranges_.end());
#endif
}

template <class T, std::size_t N>
void Tape<T, N>::foldSubrecordings()
{
    while (nestedRecordings_.size() > 1)
    {
        foldSubrecording();
    }
}

template <class T, std::size_t N>
void Tape<T, N>::newNestedRecording()
{
    SubRecording newr(*currentRec_);
#ifdef XAD_TAPE_REUSE_SLOTS
    newr.startRange_ = reusable_ranges_.end();
    newr.latestRange_ = newr.startRange_;
#endif
    // clearDerivativesAfter(position_type(statement_.size())-1);
    derivatives_.resize(currentRec_->prevMax_);
    currentRec_->maxDerivative_ = currentRec_->prevMax_;

    newr.statementStartPos_ = slot_type(statement_.size());
    newr.opStartPos_ = slot_type(operations_.size());
    newr.derivativesInitialized_ = false;
    newr.startDerivative_ = currentRec_->maxDerivative_;
    nestedRecordings_.push(newr);

    currentRec_ = &nestedRecordings_.top();
    // std::cout << "new nested recording at stmt " << newr.statementStartPos_
    //    << ", startDer: " << currentRec_->maxDerivative_ << "\n";
}

template <class T, std::size_t N>
void Tape<T, N>::endNestedRecording()
{
    foldSubrecording();

    // std::cout << "finished nested recording, stmt pos " << statement_.size() <<
    // "\n";
}

template <class T, std::size_t N>
void Tape<T, N>::newRecording()
{
    operations_.clear();
    statement_.clear();
    checkpoints_.clear();
    foldSubrecordings();
    currentRec_->maxDerivative_ = currentRec_->iDerivative_ + 1;
    statement_.push_back(std::make_pair(size_type(operations_.size()), slot_type(INVALID_SLOT)));
    currentRec_->derivativesInitialized_ = false;
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumOperations() const
{
    return size_type(operations_.size());
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumStatements() const
{
    // return size_type(statement_endpoint_.size()) - 1;
    return size_type(statement_.size()) - 1;
}

template <class T, std::size_t N>
void Tape<T, N>::initDerivatives()
{
    if (!currentRec_->derivativesInitialized_ &&
        derivatives_.size() > currentRec_->startDerivative_)
    {
        std::fill(derivatives_.begin() + currentRec_->startDerivative_, derivatives_.end(), T());
    }

    derivatives_.resize(currentRec_->maxDerivative_, Tape<T, N>::derivative_type());
    currentRec_->derivativesInitialized_ = true;
}

template <class T, std::size_t N>
typename Tape<T, N>::derivative_type& Tape<T, N>::derivative(slot_type s)
{
    if (s >= currentRec_->maxDerivative_)
        throw OutOfRange("given derivative slot is out of range - did you register the outputs?");

    initDerivatives();
    return derivatives_[s];
}

template <class T, std::size_t N>
const typename Tape<T, N>::derivative_type& Tape<T, N>::derivative(slot_type s) const
{
#ifndef NDEBUG
    if (s >= currentRec_->maxDerivative_)
        throw OutOfRange("given derivative slot is out of range");
    if (!currentRec_->derivativesInitialized_)
        throw DerivativesNotInitialized(
            "attempt to get derivative value from const object without setting "
            "derivatives first");
#endif
    return derivatives_[s];
}

template <class T, std::size_t N>
void Tape<T, N>::printStatus() const
{
    /*
    std::cout << "**** Operations: ******\n"
              << "Slot\tMultiplier\n";
    for (unsigned i = 0, e = getNumOperations(); i < e; ++i)
    {
      std::cout << i << ":  " << slot_[i] << "\t" << multiplier_[i] << "\n";
    }
    std::cout << "\n***** Statements: *****\n"
              << "res_slot\tendpoint\n";
    for (unsigned i = 1, e = unsigned(statement_.size()); i < e; ++i)
    {
      std::cout << i << ":   " << statement_[i].second << "\t"
                << statement_[i].first << "\n";
    }

    std::cout << "\n***** Derivatives: ****\n"
              << "value\n";
    for (unsigned i = 0, e = unsigned(derivatives_.size()); i < e; ++i)
    {
      if (derivatives_[i] != 0.0)
        std::cout << i << ":   " << derivatives_[i] << "\n";
    }
    */
    /*
  #ifdef XAD_TAPE_REUSE_SLOTS
    if (!reusable_ranges_.empty()) {
      std::cout << "\n*** Gaps: ********\n";
      std::cout << getReusableSlotsString() << std::endl;
    }
    else
      std::cout << "\n no reusable slots" << std::endl;
  #endif
  */
    unsigned actmax = 0;
    for (unsigned i = 1; i < unsigned(statement_.size()); ++i)
    {
        if (statement_[i].second < INVALID_SLOT)
            actmax = std::max(actmax, statement_[i].second);
    }
    std::cout << "XAD Tape Info:\n"
              << "   Statements: " << statement_.size() - 1 << "\n"
              << "   Operations: " << operations_.size() << "\n"
              << "   Total der : " << currentRec_->maxDerivative_ << "\n"
              << "   Der alloc : " << derivatives_.size() << "\n"
              << "   curr der  : " << currentRec_->numDerivatives_ << "\n"
              << "   act. max  : " << actmax << "\n"
              << "   next idx  : " << currentRec_->iDerivative_ << "\n"
              << "   Gaps      : " << getReusableSlotsString() << std::endl;
}

template <class T, std::size_t N>
void Tape<T, N>::computeAdjoints()
{
    if (!currentRec_->derivativesInitialized_)
        throw DerivativesNotInitialized();

    LOG_DEBUG("stmts: " << statement_.size() - 1 << "\n"
                        << "endp:  " << currentRec_->statementStartPos_ - 1);
    computeAdjointsTo(currentRec_->statementStartPos_ - 1);
}

template <class T, std::size_t N>
void Tape<T, N>::computeAdjointsTo(position_type pos)
{
    position_type start = position_type(statement_.size() - 1);
    LOG_DEBUG("number checkpoints: " << checkpoints_.size());
    for (unsigned i = unsigned(checkpoints_.size()); i > 0; --i)
    {
        auto& chkpt = checkpoints_[i - 1];
        if (chkpt.first <= pos)
            break;
        LOG_DEBUG("ckpt at " << chkpt.first);

        position_type end = chkpt.first;
        CheckpointCallback<Tape>* cb = chkpt.second;

        // std::cout << ""
        computeAdjointsToImpl(end, start);

        resetTo(end - 1);  // removes up to and incl. the checkpoint from statements

        // keep the prevMax stored, so we can resize the der. vector on a
        // subrecording
        currentRec_->prevMax_ = currentRec_->maxDerivative_;

        cb->computeAdjoint(this);
        currentRec_->prevMax_ = INVALID_SLOT;
        // std::cout << "********Reset to " << end-1 << std::endl;

        resetTo(end - 1);  // another reset, in case the checkpoint added to the tape again
        // clearDerivativesAfter(end-1);
        start = end - 1;  // skip over the checkpoint itself (?)
        LOG_DEBUG("new start=" << start);
        // printStatus();
    }

    LOG_DEBUG("after checkpoints, we go from " << start << " to " << pos);
    if (start > pos)
        computeAdjointsToImpl(pos, start);
}

template <class T, std::size_t N>
void Tape<T, N>::computeAdjointsToImpl(position_type pos, position_type start)
{
    LOG_DEBUG("computing adj from " << start << " to " << pos);

    // go from statements, back to front (statements 0 point is added at
    // initialization - it has endpoint set to zero)

    if (pos == start)
        return;
    using s_type = typename TapeContainerTraits<T, slot_type>::statements_type;
    auto startchunk = s_type::getHighPart(start);
    auto idx = s_type::getLowPart(start);
    auto endchunk = s_type::getHighPart(pos + 1);
    auto endcidx = s_type::getLowPart(pos + 1);
    auto endidx = endcidx;
    if (startchunk != endchunk)
        endidx = 0;
    auto chunk_it = statement_.chunk_begin() + startchunk;
    auto chunk_eit = statement_.chunk_begin() + endchunk - 1;
    const int chunksz = int(s_type::chunk_size);

    for (; chunk_it != chunk_eit; --chunk_it)
    {
        if (chunk_it == chunk_eit + 1)
            endidx = endcidx;

        for (auto it = (*chunk_it) + idx, eit = (*chunk_it) + endidx; it != eit; --it)
        {
            auto st = *it;
            auto a = derivatives_[st.second];
            derivatives_[st.second] = derivative_type();
            if (a != derivative_type())
            {
                operations_.for_each(it[-1].first, st.first, [&](const T& mul, slot_type slot)
                                     { derivatives_[slot] += mul * a; });
            }
        }
        // last iteration separate
        {
            auto prevendpoint =
                endidx == 0 ? chunk_it[-1][chunksz - 1].first : chunk_it[0][endidx - 1].first;
            auto st = chunk_it[0][endidx];
            auto a = derivatives_[st.second];
            derivatives_[st.second] = derivative_type();
            if (a != derivative_type())
            {
                operations_.for_each(prevendpoint, st.first, [&](const T& mul, slot_type slot)
                                     { derivatives_[slot] += mul * a; });
            }
        }

        idx = chunksz - 1;
    }
}

template <class T, std::size_t N>
std::size_t Tape<T, N>::getMemory() const
{
    return sizeof(T) * (operations_.size() + derivatives_.size()) +
           sizeof(slot_type) * (operations_.size() +
                                // statement_endpoint_.size() + statement_slot_.size()
                                2 * statement_.size())
#ifdef XAD_TAPE_REUSE_SLOTS
           + sizeof(reusable_ranges_.front()) * reusable_ranges_.size()
#endif
           + checkpoints_.size() * sizeof(chkpt_type) +
           nestedRecordings_.size() * sizeof(nestedRecordings_.top())
        //+ sizeof(*this)
        ;
}

template <class T, std::size_t N>
void Tape<T, N>::clearDerivatives()
{
    currentRec_->derivativesInitialized_ = false;
}

template <class T, std::size_t N>
void Tape<T, N>::insertCallback(CheckpointCallback<Tape<T, N>>* cb)
{
    checkpoints_.push_back(std::make_pair(position_type(statement_.size()), cb));
    statement_.push_back(std::make_pair(size_type(operations_.size()), slot_type(INVALID_SLOT)));
}

template <class T, std::size_t N>
void Tape<T, N>::incrementAdjoint(slot_type slot, const T& x)
{
    if (slot >= derivatives_.size())
        throw OutOfRange("adjoint to be incremented is out of range");

    derivatives_[slot] += x;
}

template <class T, std::size_t N>
typename Tape<T, N>::position_type Tape<T, N>::getPosition() const
{
    return static_cast<position_type>(statement_.size() - 1);
}

template <class T, std::size_t N>
void Tape<T, N>::resetTo(position_type pos)
{
    LOG_DEBUG("resetting to " << pos);
    if (pos >= position_type(statement_.size() - 1))
        return;

    std::pair<slot_type, slot_type> st = statement_[pos];
    statement_.resize(pos + 1);
    operations_.resize(st.first);
    if (!checkpoints_.empty())
    {
        auto newend = std::upper_bound(std::begin(checkpoints_), std::end(checkpoints_), pos,
                                       [](position_type p, chkpt_type c) { return p < c.first; });
        LOG_DEBUG("removing " << std::distance(newend, checkpoints_.end()) << " checkpoints");
        checkpoints_.erase(newend, checkpoints_.end());
    }
#ifdef XAD_TAPE_REUSE_SLOTS
    if (!reusable_ranges_.empty())
    {
        auto it = reusable_ranges_.end();
        --it;
        while (!reusable_ranges_.empty() && it->second() >= currentRec_->maxDerivative_)
        {
            if (it->first() >= currentRec_->maxDerivative_)
            {
                auto it2 = it;
                --it2;
                reusable_ranges_.erase(it);
                it = it2;
            }
            else
            {
                it->second(currentRec_->maxDerivative_);
            }
        }
    }
#endif

    // clearDerivatives(getPosition());
    // std::cout << "reset to " << pos << "\n";
}

template <class T, std::size_t N>
void Tape<T, N>::pushCallback(callback_type cb)
{
    callbacks_.push_back(cb);
}

template <class T, std::size_t N>
typename Tape<T, N>::callback_type Tape<T, N>::getLastCallback()
{
    if (callbacks_.empty())
        throw OutOfRange("Empty callback stack");
    return callbacks_.back();
}

template <class T, std::size_t N>
typename Tape<T, N>::size_type Tape<T, N>::getNumCallbacks() const
{
    return Tape<T, N>::size_type(callbacks_.size());
}

template <class T, std::size_t N>
void Tape<T, N>::popCallback()
{
    if (callbacks_.empty())
        throw OutOfRange("Empty callback stack");
    callbacks_.pop_back();
}

template <class T, std::size_t N>
bool Tape<T, N>::haveCallbacks() const
{
    return !callbacks_.empty();
}

template <class T, std::size_t N>
void Tape<T, N>::clearDerivativesAfter(position_type pos)
{
    auto& st = statement_[pos];
    derivatives_.resize(st.second + 1);
    currentRec_->maxDerivative_ = st.second + 1;
}

template <class T, std::size_t N>
typename Tape<T, N>::derivative_type Tape<T, N>::getAndResetOutputAdjoint(slot_type slot)
{
    if (slot >= slot_type(derivatives_.size()))
        throw OutOfRange("Requested output slot does not exist");

    derivative_type ret = derivatives_[slot];
    derivatives_[slot] = derivative_type();
    return ret;
}

template <class T, std::size_t N>
XAD_THREAD_LOCAL Tape<T, N>* Tape<T, N>::active_tape_ = nullptr;

#define MAKE_TAPE_TLS(type)                                                                        \
    template class CheckpointCallback<Tape<type>>;                                                 \
    template class Tape<type>;

#define MAKE_TAPE_TLS_HIGHER(type)                                                                 \
    template struct type;                                                                          \
    MAKE_TAPE_TLS(type)
// 1st order
MAKE_TAPE_TLS(float)
MAKE_TAPE_TLS(double)
// 2nd order
MAKE_TAPE_TLS_HIGHER(AReal<float>)
MAKE_TAPE_TLS_HIGHER(AReal<double>)
MAKE_TAPE_TLS(XAD_SINGLE_ARG(FReal<float>))
MAKE_TAPE_TLS(XAD_SINGLE_ARG(FReal<double>))
MAKE_TAPE_TLS_HIGHER(ARealDirect<float>)
MAKE_TAPE_TLS_HIGHER(ARealDirect<double>)
MAKE_TAPE_TLS_HIGHER(FRealDirect<double>)
MAKE_TAPE_TLS_HIGHER(FRealDirect<float>)

#include <XAD/Instantiations.hpp>

#undef MAKE_STACK_TLS

}  // namespace xad

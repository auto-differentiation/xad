/*******************************************************************************

   Implementation of the Tape

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
#include <iterator>
#include <numeric>
#include <sstream>
#ifdef XAD_TAPE_CHECKS
#include <atomic>
#endif

#if 0
#define LOG_DEBUG(msg) std::cout << msg << std::endl;
#else
#define LOG_DEBUG(msg)
#endif

namespace xad
{

#ifdef XAD_TAPE_CHECKS
namespace
{
// unique, also for a tape created where a destroyed one was; 0 is no tape
unsigned int nextTapeId()
{
    static std::atomic<unsigned int> last(0);
    unsigned int id = ++last;
    return id != 0 ? id : ++last;
}
}  // namespace
#endif

template <class T, std::size_t N>
Tape<T, N>::Tape(bool activateNow)
{
#ifdef XAD_TAPE_CHECKS
    id_ = nextTapeId();
#endif

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
      positionSlots_(std::move(o.positionSlots_)),
#ifdef XAD_TAPE_REUSE_SLOTS
      reusable_ranges_(std::move(o.reusable_ranges_)),
      range_index_(std::move(o.range_index_)),
#endif
      nestedRecordings_(std::move(o.nestedRecordings_)),
      currentRec_(o.currentRec_)
{
#ifdef XAD_TAPE_CHECKS
    id_ = o.id_;
    o.id_ = nextTapeId();
#endif
#ifdef XAD_TAPE_REUSE_SLOTS
    rebindRangeIterators();
#endif
    o.resetMovedFrom();
#ifdef XAD_ENABLE_JIT
    // the JIT that deactivated o reactivates this tape instead
    if (o.jitSuspender_)
    {
        jitSuspender_ = o.jitSuspender_;
        o.jitSuspender_ = nullptr;
        *jitSuspender_ = this;
    }
#endif
    if (o.isActive())
    {
        // activate() would throw while o is active
        o.deactivate();
        active_tape_ = this;
    }
}

template <class T, std::size_t N>
Tape<T, N>& Tape<T, N>::operator=(Tape&& o) noexcept
{
    if (this == &o)
        return *this;
    operations_ = std::move(o.operations_);
    statement_ = std::move(o.statement_);
    derivatives_ = std::move(o.derivatives_);
    checkpoints_ = std::move(o.checkpoints_);
    clearCallbacks();
    callbacks_.swap(o.callbacks_);
    positionSlots_ = std::move(o.positionSlots_);
#ifdef XAD_TAPE_REUSE_SLOTS
    reusable_ranges_ = std::move(o.reusable_ranges_);
    range_index_ = std::move(o.range_index_);
#endif
    nestedRecordings_ = std::move(o.nestedRecordings_);
    currentRec_ = o.currentRec_;
#ifdef XAD_TAPE_CHECKS
    id_ = o.id_;
    o.id_ = nextTapeId();
#endif
#ifdef XAD_TAPE_REUSE_SLOTS
    rebindRangeIterators();
#endif
    o.resetMovedFrom();
#ifdef XAD_ENABLE_JIT
    if (o.jitSuspender_)
    {
        if (jitSuspender_)
            *jitSuspender_ = nullptr;
        jitSuspender_ = o.jitSuspender_;
        o.jitSuspender_ = nullptr;
        *jitSuspender_ = this;
    }
#endif
    if (o.isActive())
    {
        // activate() would throw while o is active
        o.deactivate();
        active_tape_ = this;
    }
    else
        deactivate();
    return *this;
}

// its own empty recording, so it leaves the successor alone; clearAll makes it record again
template <class T, std::size_t N>
void Tape<T, N>::resetMovedFrom()
{
    while (!nestedRecordings_.empty()) nestedRecordings_.pop();
    nestedRecordings_.push(SubRecording(this));
    currentRec_ = &nestedRecordings_.top();
}

template <class T, std::size_t N>
Tape<T, N>::~Tape()
{
    deactivate();
#ifdef XAD_ENABLE_JIT
    if (jitSuspender_)
        *jitSuspender_ = nullptr;
#endif
    clearCallbacks();
}

template <class T, std::size_t N>
void Tape<T, N>::clearAll()
{
    operations_.clear();
    statement_.clear();
    derivatives_.clear();
    checkpoints_.clear();
    positionSlots_.clear();
#ifdef XAD_TAPE_REUSE_SLOTS
    reusable_ranges_.clear();
    range_index_.clear();
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

template <class T, std::size_t N>
typename Tape<T, N>::slot_type Tape<T, N>::recordStatement(const value_type* multipliers,
                                                           const slot_type* slots, unsigned n)
{
    slot_type lhs = registerVariable();
    pushAll(multipliers, slots, n);
    pushLhs(lhs);
    return lhs;
}

template <class T, std::size_t N>
void Tape<T, N>::recordAssignment(slot_type& lhs, const value_type* multipliers,
                                  const slot_type* slots, unsigned n)
{
    pushAll(multipliers, slots, n);
    if (lhs == INVALID_SLOT)
        lhs = registerVariable();
    pushLhs(lhs);
}

#ifdef XAD_TAPE_REUSE_SLOTS
namespace
{
// More than MIN_INDEXED_RANGES reusable ranges get indexed once linear searches have passed
// RANGE_INDEX_BUILD_COST times as many in all; the index is dropped when updated more than once per
// RANGE_INDEX_UPDATE_COST ranges without being searched.
const std::size_t MIN_INDEXED_RANGES = 64;
const std::size_t RANGE_INDEX_BUILD_COST = 16;
const std::size_t RANGE_INDEX_UPDATE_COST = 32;
}  // namespace

// the first range of the current recording that ends at or after the slot
template <class T, std::size_t N>
typename Tape<T, N>::range_list::iterator Tape<T, N>::findRange(slot_type slot)
{
    auto first = currentRec_->startRange_;
    if (range_index_.empty())
    {
        auto it = first;
        std::size_t passed = 0;
        for (; it != reusable_ranges_.end() && it->second() < slot; ++it) ++passed;
        if (reusable_ranges_.size() > MIN_INDEXED_RANGES)
        {
            rangeSearchCost_ += passed;
            if (rangeSearchCost_ > RANGE_INDEX_BUILD_COST * reusable_ranges_.size())
            {
                rangeSearchCost_ = 0;
                rangeIndexUpdates_ = 0;
                for (auto r = reusable_ranges_.begin(); r != reusable_ranges_.end(); ++r)
                    range_index_.emplace_hint(range_index_.end(), RangeStart{&*r}, r);
                if (range_index_.size() != reusable_ranges_.size())  // ranges with the same start
                    range_index_.clear();
            }
        }
        return it;
    }

    rangeIndexUpdates_ = 0;
    slot_range_type probe(slot, slot);
    auto entry = range_index_.lower_bound(RangeStart{&probe});
    if (entry != range_index_.begin())
    {
        auto before = std::prev(entry)->second;
        if (before->second() >= slot)
            --entry;
    }
    auto it = entry == range_index_.end() ? reusable_ranges_.end() : entry->second;
    // only the ranges of this recording, from its first one on
    if (first == reusable_ranges_.end() || (it != reusable_ranges_.end() && *it < *first))
        return first;
    return it;
}

template <class T, std::size_t N>
XAD_FORCE_INLINE typename Tape<T, N>::range_list::iterator Tape<T, N>::insertRange(
    range_list::iterator pos, slot_type slot)
{
    auto it = reusable_ranges_.emplace(pos, slot, slot + 1);
    if (!range_index_.empty())
        indexRange(it);
    return it;
}

template <class T, std::size_t N>
XAD_FORCE_INLINE typename Tape<T, N>::range_list::iterator Tape<T, N>::eraseRange(
    range_list::iterator it)
{
    if (!range_index_.empty())
        unindexRange(it);
    return reusable_ranges_.erase(it);
}

// the index is dropped when updated more often than searched, or if a slot freed twice gives two
// ranges the same start
template <class T, std::size_t N>
void Tape<T, N>::indexRange(range_list::iterator it)
{
    if (!range_index_.emplace(RangeStart{&*it}, it).second ||
        ++rangeIndexUpdates_ > reusable_ranges_.size() / RANGE_INDEX_UPDATE_COST)
        range_index_.clear();
}

template <class T, std::size_t N>
void Tape<T, N>::unindexRange(range_list::iterator it)
{
    auto entry = range_index_.find(RangeStart{&*it});
    if (entry == range_index_.end() || entry->second != it ||
        ++rangeIndexUpdates_ > reusable_ranges_.size() / RANGE_INDEX_UPDATE_COST)
        range_index_.clear();
    else
        range_index_.erase(entry);
}

// after a move, point the outermost recording at this tape's ranges
template <class T, std::size_t N>
void Tape<T, N>::rebindRangeIterators()
{
    if (nestedRecordings_.size() != 1)
        return;
    currentRec_->startRange_ =
        reusable_ranges_.empty() ? reusable_ranges_.end() : reusable_ranges_.begin();
    currentRec_->latestRange_ = reusable_ranges_.end();
}

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
        eraseRange(currentRec_->startRange_);
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
                    eraseRange(it);
                    currentRec_->startRange_ = reusable_ranges_.end();
                }
                else
                    eraseRange(it);
            }
        }
    }
    else if (slot < currentRec_->iDerivative_)  // not at the end, nor reused by resetTo
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
                // new range at the end, the recording's first if it has none
                currentRec_->latestRange_ = insertRange(reusable_ranges_.end(), slot);
                if (currentRec_->startRange_ == reusable_ranges_.end())
                    currentRec_->startRange_ = currentRec_->latestRange_;
            }
            else
            {
                // it must fit into another range now
                auto it = findRange(slot);
                if (it != reusable_ranges_.end())
                    status = it->expand(slot);
                if (status != slot_range_type::FAILED)
                {
                    currentRec_->latestRange_ = it;
                }
                else
                {
                    // insert a new range
                    currentRec_->latestRange_ = insertRange(it, slot);
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
                // merge two ranges, erasing the first one before its start is taken over
                slot_range_type previous = *it;
                if (it == currentRec_->startRange_)
                    ++currentRec_->startRange_;
                eraseRange(it);
                currentRec_->latestRange_->joinStart(previous);
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
                eraseRange(it);
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
    positionSlots_.erase(std::lower_bound(positionSlots_.begin(), positionSlots_.end(),
                                          position_slots_type(prev.statementStartPos_, 0)),
                         positionSlots_.end());
#ifdef XAD_TAPE_REUSE_SLOTS
    for (auto range = prev.startRange_; range != reusable_ranges_.end();) range = eraseRange(range);
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
    // above the outer recording's slots, whose adjoints may be pending
    newr.iDerivative_ = (std::max)(newr.iDerivative_, newr.startDerivative_);
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
    positionSlots_.clear();
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
#ifdef XAD_ENABLE_JIT
    // slots from FIRST_JIT_SLOT on are passive, so their dependencies were not recorded
    if (currentRec_->maxDerivative_ > FIRST_JIT_SLOT)
        throw OutOfRange("too many tape slots with JIT enabled");
#endif
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

        // keep the prevMax stored, so we can resize the der. vector on a
        // subrecording
        currentRec_->prevMax_ = currentRec_->maxDerivative_;
        {
            // a nested recording keeps the callback's variables off slots with pending adjoints
            ScopedNestedRecording<Tape> nested(this);
            cb->computeAdjoint(this);
        }
        currentRec_->prevMax_ = INVALID_SLOT;
        // std::cout << "********Reset to " << end-1 << std::endl;

        // clearDerivativesAfter(end-1);
        start = end - 1;  // skip over the checkpoint itself (?)
        LOG_DEBUG("new start=" << start);
        // printStatus();
    }

    LOG_DEBUG("after checkpoints, we go from " << start << " to " << pos);
    if (start > pos)
        computeAdjointsToImpl(pos, start);
}

namespace detail
{

// zero in all orders: a zero value can still carry a tangent (FReal) or tape dependency (AReal)
template <class T>
XAD_INLINE bool isZeroAdjoint(const T& a);
template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const Vec<T, N>& a);
template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const FReal<T, N>& a);
template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const FRealDirect<T, N>& a);
template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const AReal<T, N>& a);
template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const ARealDirect<T, N>& a);

template <class T>
XAD_INLINE bool isZeroAdjoint(const T& a)
{
    return a == T();
}

template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const Vec<T, N>& a)
{
    for (std::size_t i = 0; i < N; ++i)
        if (!isZeroAdjoint(a[i]))
            return false;
    return true;
}

template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const FReal<T, N>& a)
{
    return isZeroAdjoint(a.value()) && isZeroAdjoint(a.derivative());
}

template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const FRealDirect<T, N>& a)
{
    return isZeroAdjoint(a.base());
}

template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const AReal<T, N>& a)
{
    return !a.shouldRecord() && isZeroAdjoint(a.value());
}

template <class T, std::size_t N>
XAD_INLINE bool isZeroAdjoint(const ARealDirect<T, N>& a)
{
    return isZeroAdjoint(a.base());
}

}  // namespace detail

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
            if (!detail::isZeroAdjoint(a))
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
            if (!detail::isZeroAdjoint(a))
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
    auto pos = static_cast<position_type>(statement_.size() - 1);
    while (!positionSlots_.empty() && positionSlots_.back().first >= pos) positionSlots_.pop_back();
    positionSlots_.push_back(position_slots_type(pos, currentRec_->iDerivative_));
    return pos;
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
    // reuse the slots registered after the position, if getPosition gave it out
    auto mark =
        std::lower_bound(positionSlots_.begin(), positionSlots_.end(), position_slots_type(pos, 0));
    if (mark != positionSlots_.end() && mark->first == pos &&
        pos + 1 >= currentRec_->statementStartPos_)
    {
        currentRec_->iDerivative_ = (std::min)(currentRec_->iDerivative_, mark->second);
        ++mark;
    }
    positionSlots_.erase(mark, positionSlots_.end());
#ifdef XAD_TAPE_REUSE_SLOTS
    // drop the reusable slots at or above maxDerivative_ or the slot counter
    auto limit = (std::min)(currentRec_->maxDerivative_, currentRec_->iDerivative_);
    while (!reusable_ranges_.empty() && reusable_ranges_.back().second() > limit)
    {
        auto last = std::prev(reusable_ranges_.end());
        if (last->first() < limit)
            last->second(limit);
        else
        {
            if (currentRec_->startRange_ == last)
                currentRec_->startRange_ = reusable_ranges_.end();
            if (currentRec_->latestRange_ == last)
                currentRec_->latestRange_ = reusable_ranges_.end();
            eraseRange(last);
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
void Tape<T, N>::clearCallbacks()
{
    for (auto p : callbacks_) delete p;
    callbacks_.clear();
}

template <class T, std::size_t N>
void Tape<T, N>::clearDerivativesAfter(position_type pos)
{
    // keep the slots in use at pos, which getPosition noted; the slot of the statement at pos
    // can be an older one, or none
    auto mark =
        std::lower_bound(positionSlots_.begin(), positionSlots_.end(), position_slots_type(pos, 0));
    slot_type end;
    if (mark != positionSlots_.end() && mark->first == pos)
        end = mark->second;
    else if (statement_[pos].second != INVALID_SLOT)
        end = statement_[pos].second + 1;
    else
        return;
    derivatives_.resize(end);
    currentRec_->maxDerivative_ = end;
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

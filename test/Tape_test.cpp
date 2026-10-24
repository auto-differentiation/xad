/*******************************************************************************

   Unit tests for the tape itself

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

#include <XAD/CheckpointCallback.hpp>
#include <XAD/Tape.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(Tape, isEmptyByDefault)
{
    using xad::Tape;

    EXPECT_EQ(nullptr, Tape<double>::getActive());

    {
        Tape<double> s;

        EXPECT_TRUE(s.isActive());
        ASSERT_NE(nullptr, Tape<double>::getActive());
        EXPECT_EQ(&s, Tape<double>::getActive());
    }

    EXPECT_EQ(nullptr, Tape<double>::getActive());
}

TEST(Tape, canInitializeDeactivated)
{
    using xad::Tape;
    Tape<float> s(false);

    EXPECT_FALSE(s.isActive());
    EXPECT_EQ(nullptr, Tape<float>::getActive());

    s.activate();

    EXPECT_TRUE(s.isActive());
    EXPECT_NE(nullptr, Tape<float>::getActive());
}

TEST(Tape, canActivateStatically)
{
    using xad::Tape;
    Tape<float> s(false);

    EXPECT_FALSE(s.isActive());
    EXPECT_EQ(nullptr, Tape<float>::getActive());

    xad::Tape<float>::setActive(&s);

    EXPECT_TRUE(s.isActive());
    EXPECT_NE(nullptr, Tape<float>::getActive());
}

TEST(Tape, canDeactivateGlobally)
{
    using xad::Tape;

    EXPECT_EQ(nullptr, Tape<double>::getActive());

    Tape<double> s;

    EXPECT_TRUE(s.isActive());
    Tape<double>::deactivateAll();
    EXPECT_FALSE(s.isActive());
}

TEST(Tape, isMovable)
{
    xad::Tape<double> s(false);
    xad::Tape<double> s2 = std::move(s);  // move constructor
    EXPECT_FALSE(s2.isActive());

    xad::Tape<double> s3(true);
    s3 = std::move(s2);  // move assign
    EXPECT_FALSE(s3.isActive());
}

TEST(Tape, movingActiveTapeMovesActivation)
{
    xad::Tape<double> s;
    xad::Tape<double> moved(std::move(s));
    EXPECT_TRUE(moved.isActive());
    EXPECT_FALSE(s.isActive());

    xad::Tape<double> assigned(false);
    assigned = std::move(moved);
    EXPECT_TRUE(assigned.isActive());
    EXPECT_FALSE(moved.isActive());
}

TEST(Tape, canRegisterVariables)
{
    using xad::Tape;
    Tape<double> s;

    EXPECT_EQ(0U, s.getNumVariables());

    auto slot1 = s.registerVariable();
    auto slot2 = s.registerVariable();

    EXPECT_EQ(2U, s.getNumVariables());
    EXPECT_EQ(0U, slot1);
    EXPECT_EQ(1U, slot2);
}

TEST(Tape, canUnregisterInOrder)
{
    xad::Tape<double> s;

    auto slot1 = s.registerVariable();
    auto slot2 = s.registerVariable();

    s.unregisterVariable(slot2);
    EXPECT_EQ(1U, s.getNumVariables());
    s.unregisterVariable(slot1);
    EXPECT_EQ(0U, s.getNumVariables());
}

TEST(Tape, canUnregisterOutOfOrder)
{
    xad::Tape<double> s;

    auto slot1 = s.registerVariable();
    auto slot2 = s.registerVariable();

    s.unregisterVariable(slot1);
    EXPECT_EQ(1U, s.getNumVariables());
    s.unregisterVariable(slot2);
    EXPECT_EQ(0U, s.getNumVariables());
}

#ifdef XAD_TAPE_REUSE_SLOTS

TEST(Tape, canReuseSlots)
{
    xad::Tape<double> s;

    std::vector<xad::Tape<double>::slot_type> slots(10);
    for (auto& sl : slots)
    {
        sl = s.registerVariable();
    }

    typedef xad::Tape<double>::slot_type slot_type;
    for (slot_type i = 0; i < slots.size(); ++i) EXPECT_EQ(i, slots[i]);
    EXPECT_EQ(slots.size(), s.getNumVariables());

    for (slot_type i = 3; i < 6; ++i) s.unregisterVariable(i);
    s.unregisterVariable(slots[8]);
    std::cout << s.getReusableSlotsString() << std::endl;
    // now we have 2 ranges, [3,6) and [8,9)
    EXPECT_EQ(6U, s.getNumVariables());
    EXPECT_EQ(2U, s.getNumReusableSlotSections()) << s.getReusableSlotsString();
    EXPECT_EQ(4U, s.getNumReusableSlots()) << s.getReusableSlotsString();

    // new variables should now get slot numbers 3-5 or 8
    auto s1 = s.registerVariable();
    EXPECT_TRUE((s1 >= 3 && s1 < 6) || s1 == 8)
        << "new variable not in reusable range - it is " << s1;
    EXPECT_EQ(7U, s.getNumVariables());
    EXPECT_EQ(3U, s.getNumReusableSlots());
    auto s2 = s.registerVariable();
    EXPECT_TRUE((s2 >= 3 && s2 < 6) || s2 == 8)
        << "new variable not in reusable range - it is " << s1;
    EXPECT_EQ(8U, s.getNumVariables());
    EXPECT_EQ(2U, s.getNumReusableSlots());
    EXPECT_NE(s1, s2);
    auto s3 = s.registerVariable();
    EXPECT_TRUE((s3 >= 3 && s3 < 6) || s3 == 8)
        << "new variable not in reusable range - it is " << s1;
    EXPECT_EQ(9U, s.getNumVariables());
    EXPECT_EQ(1U, s.getNumReusableSlots());
    EXPECT_NE(s3, s2);
    EXPECT_NE(s3, s1);
    auto s4 = s.registerVariable();
    EXPECT_TRUE((s4 >= 3 && s4 < 6) || s4 == 8)
        << "new variable not in reusable range - it is " << s1;
    EXPECT_EQ(10U, s.getNumVariables());
    EXPECT_EQ(0U, s.getNumReusableSlots());
    EXPECT_NE(s4, s3);
    EXPECT_NE(s4, s2);
    EXPECT_NE(s4, s1);

    EXPECT_EQ(0U, s.getNumReusableSlotSections());

    // now we filled all the ranges, so new variable should get slot 10
    auto s5 = s.registerVariable();
    EXPECT_EQ(10U, s5);
    EXPECT_EQ(11U, s.getNumVariables());
}
#endif

TEST(Tape, reusableSlotsMethodsAlwaysCallable)
{
    // Test that getNumReusableSlotSections() and getNumReusableSlots() are
    // callable regardless of XAD_TAPE_REUSE_SLOTS being defined.
    // When XAD_TAPE_REUSE_SLOTS is NOT defined, they return 1 and 0 respectively.
    // When XAD_TAPE_REUSE_SLOTS IS defined, they return actual values.
    xad::Tape<double> s;

    // These methods should always be callable
    auto sections = s.getNumReusableSlotSections();
    auto slots = s.getNumReusableSlots();

#ifdef XAD_TAPE_REUSE_SLOTS
    // With slot reuse enabled, initially no reusable slots
    EXPECT_EQ(0U, sections);
    EXPECT_EQ(0U, slots);
#else
    // Without slot reuse, returns fixed values
    EXPECT_EQ(1U, sections);
    EXPECT_EQ(0U, slots);
#endif
}

TEST(Tape, canDeriveStatements)
{
    xad::Tape<double> s;

    // putting z = x1*x2 + sin(x1);
    auto x1 = M_PI;
    auto x2 = 2.0;

    auto x1s = s.registerVariable();
    auto x2s = s.registerVariable();

    EXPECT_EQ(2U, s.getNumVariables());
    EXPECT_EQ(0U, s.getNumOperations());
    EXPECT_EQ(0U, s.getNumStatements());

    s.newRecording();
    // auto z = x1*x1 + std::sin(x1);
    auto zs = s.registerVariable();
    auto muls = {std::cos(x1), x2, x1};
    auto slots = {x1s, x1s, x2s};
    s.pushAll(muls.begin(), slots.begin(), 3);
    s.pushLhs(zs);

    EXPECT_EQ(3U, s.getNumVariables());
    EXPECT_EQ(3U, s.getNumOperations());
    EXPECT_EQ(1U, s.getNumStatements());

    // set the derivative for output (both from rvalue and lvalue)
    s.setDerivative(zs, 1.5);
    EXPECT_DOUBLE_EQ(0.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(0.0, s.getDerivative(x2s));
    EXPECT_DOUBLE_EQ(1.5, s.getDerivative(zs));

    auto outDerive = 1.0;
    s.setDerivative(zs, outDerive);
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(zs));

    // s.printStatus();

    // compute the other derivatives (adjoints)
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(M_PI, s.getDerivative(x2s));

    // s.printStatus();
}

#ifndef XAD_REDUCED_MEMORY

TEST(Tape, canPushOperationsIntoReservedSpace)
{
    xad::Tape<double> s;
    using slot_type = xad::Tape<double>::slot_type;

    auto x1s = s.registerVariable();
    auto x2s = s.registerVariable();
    s.newRecording();

    auto zs = s.registerVariable();
    std::pair<double, slot_type>* dst = s.tryReserveOperations(2);
    ASSERT_NE(nullptr, dst);
    dst[0] = {2.0, x1s};
    dst[1] = {3.0, x2s};
    s.commitOperations(2);
    s.pushLhs(zs);

    EXPECT_EQ(2U, s.getNumOperations());
    EXPECT_EQ(1U, s.getNumStatements());

    s.setDerivative(zs, 1.0);
    s.computeAdjoints();

    EXPECT_DOUBLE_EQ(2.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(3.0, s.getDerivative(x2s));
}

TEST(Tape, canCommitFewerOperationsThanReserved)
{
    xad::Tape<double> s;
    using slot_type = xad::Tape<double>::slot_type;

    auto x1s = s.registerVariable();
    s.newRecording();

    auto zs = s.registerVariable();
    std::pair<double, slot_type>* dst = s.tryReserveOperations(4);
    ASSERT_NE(nullptr, dst);
    dst[0] = {2.0, x1s};
    s.commitOperations(1);
    s.pushLhs(zs);

    EXPECT_EQ(1U, s.getNumOperations());

    s.setDerivative(zs, 1.0);
    s.computeAdjoints();

    EXPECT_DOUBLE_EQ(2.0, s.getDerivative(x1s));
}

#endif

TEST(Tape, canRestartRecording)
{
    xad::Tape<double> s;

    // putting z = x1*x2 + sin(x1);
    auto x1 = M_PI;
    auto x2 = 2.0;
    auto x1s = s.registerVariable();
    auto x2s = s.registerVariable();

    s.newRecording();
    auto zs = s.registerVariable();
    auto muls = {std::cos(x1), x2, x1};
    auto slots = {x1s, x1s, x2s};
    s.pushAll(muls.begin(), slots.begin(), 3);
    s.pushLhs(zs);
    s.setDerivative(zs, 1.0);
    // compute the other derivatives (adjoints)
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(M_PI, s.getDerivative(x2s));

    // std::cout << "Memory1: " << s.getMemory() << "\n";

    auto mem = s.getMemory();
    EXPECT_GT(mem,
              50U);  // stupid test - just to make sure that getMemory does sth

    // second new recording - keeps the variables, resets operations/statements
    s.newRecording();
    EXPECT_EQ(3U, s.getNumVariables());
    EXPECT_EQ(0U, s.getNumOperations());
    EXPECT_EQ(0U, s.getNumStatements());

    // now putting y = exp(x1) + x1 / x2;
    auto ys = s.registerVariable();
    auto muls2 = {std::exp(x1), 1.0 / x2, -x1 / (x2 * x2)};
    auto slots2 = {x1s, x1s, x2s};
    s.pushAll(muls2.begin(), slots2.begin(), 3);
    s.pushLhs(ys);
    s.setDerivative(ys, 1.0);
    s.computeAdjoints();
    // s.printStatus();
    // std::cout << "Memory2: " << s.getMemory() << "\n";

    EXPECT_DOUBLE_EQ(std::exp(x1) + 1.0 / x2, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(-x1 / (x2 * x2), s.getDerivative(x2s));

    EXPECT_LE(mem, s.getMemory());
}

TEST(Tape, canPushCombined)
{
    xad::Tape<double> s;
    // putting z = x1*x2 + sin(x1);
    auto x1 = M_PI;
    auto x2 = 2.0;
    auto x1s = s.registerVariable();
    auto x2s = s.registerVariable();

    s.newRecording();
    auto zs = s.registerVariable();
    std::array<double, 3> mul = {{std::cos(x1), x2, x1}};
    std::array<xad::Tape<double>::slot_type, 3> sl = {{x1s, x1s, x2s}};
    s.pushAll(mul.data(), sl.data(), 3);
    s.pushLhs(zs);
    s.setDerivative(zs, 1.0);
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(M_PI, s.getDerivative(x2s));
}

#ifdef XAD_TAPE_REUSE_SLOTS
TEST(Tape, restartingRecordingResetsMemory)
{
    xad::Tape<double> s;

    auto s1 = s.registerVariable();
    auto s2 = s.registerVariable();
    auto s3 = s.registerVariable();
    auto s4 = s.registerVariable();
    s.newRecording();
    EXPECT_EQ(4U, s.getNumVariables());

    auto s5 = s.registerVariable();
    EXPECT_EQ(5U, s.getNumVariables());

    s.unregisterVariable(s2);
    s.unregisterVariable(s3);
    EXPECT_EQ(3U, s.getNumVariables());

    s.unregisterVariable(s5);
    EXPECT_EQ(2U, s.getNumVariables());

    s.unregisterVariable(s1);
    EXPECT_EQ(1U, s.getNumVariables());

    s.newRecording();
    EXPECT_EQ(1U, s.getNumVariables());

    // these should fill the reusable ranges
    auto s6 = s.registerVariable();
    auto s7 = s.registerVariable();
    EXPECT_EQ(3U, s.getNumVariables());
    EXPECT_LT(s6, s4);
    EXPECT_LT(s7, s4);
    //  std::cout << s4 << ", " << s6 << ", " << s7 << std::endl;

    // std::cout << s.getReusableSlotsString() << std::endl;
    s.unregisterVariable(s4);
    // std::cout << s.getReusableSlotsString() << std::endl;
    EXPECT_EQ(2U, s.getNumVariables());
}

TEST(Tape, resetToTrimsReusableSlotsToDerivatives)
{
    xad::Tape<double> s;
    std::vector<xad::Tape<double>::slot_type> slots;
    xad::Tape<double>::position_type pos = 0;
    for (int i = 0; i < 10; ++i)
    {
        slots.push_back(s.registerVariable());
        s.pushLhs(slots.back());
        if (i == 4)
            pos = s.getPosition();
    }
    s.unregisterVariable(slots[4]);
    s.unregisterVariable(slots[5]);
    EXPECT_EQ("[4, 6), ", s.getReusableSlotsString());

    s.clearDerivativesAfter(pos);  // up to slot 4
    s.resetTo(pos);
    EXPECT_EQ("[4, 5), ", s.getReusableSlotsString());
    EXPECT_EQ(4U, s.registerVariable());
}

TEST(Tape, resetToDropsAllReusableSlotsOfTheRecording)
{
    xad::Tape<double> s;
    std::vector<xad::Tape<double>::slot_type> slots;
    xad::Tape<double>::position_type pos = 0;
    for (int i = 0; i < 10; ++i)
    {
        slots.push_back(s.registerVariable());
        s.pushLhs(slots.back());
        if (i == 4)
            pos = s.getPosition();
    }
    s.unregisterVariable(slots[7]);
    EXPECT_EQ("[7, 8), ", s.getReusableSlotsString());

    s.clearDerivativesAfter(pos);  // up to slot 4
    s.resetTo(pos);
    EXPECT_EQ("", s.getReusableSlotsString());
    EXPECT_NE(7U, s.registerVariable());
    s.unregisterVariable(slots[2]);
    EXPECT_EQ("[2, 3), ", s.getReusableSlotsString());
    EXPECT_EQ(2U, s.registerVariable());
}

TEST(Tape, nestedRecordingTakesItsReusableSlotsAlong)
{
    typedef xad::Tape<double> tape_type;
    // frees a slot in the middle of a nested recording without reusable slots of its own
    struct Callback : xad::CheckpointCallback<tape_type>
    {
        void computeAdjoint(tape_type* tape) override
        {
            xad::ScopedNestedRecording<tape_type> nested(tape);
            auto a = tape->registerVariable(), b = tape->registerVariable();
            tape->unregisterVariable(a);
            tape->unregisterVariable(b);
        }
    } cb;

    tape_type s;
    std::vector<tape_type::slot_type> live;
    for (int i = 0; i < 4; ++i)
    {
        live.push_back(s.registerVariable());
        s.pushLhs(live.back());
    }
    s.unregisterVariable(live[1]);
    live.erase(live.begin() + 1);
    s.insertCallback(&cb);
    s.setDerivative(live.back(), 1.0);
    s.computeAdjoints();

    for (int i = 0; i < 3; ++i)
    {
        auto slot = s.registerVariable();
        EXPECT_TRUE(std::find(live.begin(), live.end(), slot) == live.end()) << slot;
        live.push_back(slot);
    }
}

TEST(Tape, movedTapeReusesSlots)
{
    xad::Tape<double> s(false);
    s.registerVariable();
    auto b = s.registerVariable();
    s.registerVariable();

    xad::Tape<double> moved(std::move(s));
    moved.unregisterVariable(b);
    EXPECT_EQ("[1, 2), ", moved.getReusableSlotsString());
    EXPECT_EQ(b, moved.registerVariable());
    EXPECT_EQ(3U, moved.registerVariable());

    xad::Tape<double> assigned(false);
    assigned = std::move(moved);
    assigned.unregisterVariable(b);
    EXPECT_EQ(b, assigned.registerVariable());
    EXPECT_EQ(4U, assigned.registerVariable());
}
#endif

TEST(Tape, clearCallbacksDestroysThem)
{
    struct Callback : xad::CheckpointCallback<xad::Tape<double>>
    {
        explicit Callback(int& destroyed) : destroyed_(destroyed) {}
        ~Callback() override { ++destroyed_; }
        void computeAdjoint(xad::Tape<double>*) override {}
        int& destroyed_;
    };

    int destroyed = 0;
    {
        xad::Tape<double> s;
        s.pushCallback(new Callback(destroyed));
        s.pushCallback(new Callback(destroyed));
        s.clearCallbacks();
        EXPECT_EQ(2, destroyed);
        EXPECT_FALSE(s.haveCallbacks());
        s.pushCallback(new Callback(destroyed));
    }
    EXPECT_EQ(3, destroyed);
}

namespace
{

// portable, unlike the standard distributions
class SlotTestRng
{
  public:
    explicit SlotTestRng(std::uint64_t seed) : state_(seed) {}

    unsigned operator()(unsigned n)
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return unsigned((state_ >> 33) % n);
    }

  private:
    std::uint64_t state_;
};

// Hashes the slots handed out over a long pseudo-random sequence of register and unregister
// calls, newRecording, resetTo and clearAll, with nested recordings run by checkpoint callbacks
class SlotSequence : public xad::CheckpointCallback<xad::Tape<double>>
{
  public:
    typedef xad::Tape<double> tape_type;
    typedef tape_type::slot_type slot_type;
    typedef tape_type::position_type position_type;

    SlotSequence() : rng_(2026), hash_(14695981039346656037ULL), duplicates_(0), depth_(0) {}

    std::uint64_t run(int epochs)
    {
        std::vector<slot_type> live;
        for (int e = 0; e < epochs; ++e)
        {
            unsigned target = 50 + rng_(800);
            int ops = 100 + int(rng_(400));
            slot_type pinned = 0;
            position_type pos = 0;
            // registered after the rollback point, so gone with a resetTo to it
            std::vector<slot_type> recent;
            for (int i = 0; i < ops; ++i)
            {
                if (i == ops / 2)
                {
                    // the rollback point, kept live until the end of the epoch
                    pinned = allocate(live);
                    live.pop_back();
                    pos = tape_.getPosition();
                }
                std::size_t n = live.size() + recent.size();
                unsigned r = rng_(100);
                if (r < 2)
                    tape_.insertCallback(this);
                else if (n == 0 || r < (n < target ? 65U : 35U))
                    allocate(i < ops / 2 ? live : recent);
                else
                {
                    std::size_t k = rng_(unsigned(n));
                    if (k < live.size())
                        release(live, k);
                    else
                        release(recent, k - live.size());
                }
            }
            live.push_back(pinned);
            switch (rng_(10))
            {
                case 0:
                case 1:
                case 2:
                case 3:
                    live.insert(live.end(), recent.begin(), recent.end());
                    sweep(live);
                    tape_.newRecording();
                    break;
                case 4:
                    live.insert(live.end(), recent.begin(), recent.end());
                    sweep(live);
                    break;
                case 5:
                case 6:
                    rollBack(live, recent, pinned, pos);
                    break;
                case 7:
                    tape_.resetTo(pos);
                    releaseAll(recent);
                    break;
                case 8:
                    tape_.clearAll();
                    live.clear();
                    isLive_.clear();
                    break;
                default:
                    live.insert(live.end(), recent.begin(), recent.end());
                    break;
            }
            mix(tape_.getNumVariables());
            std::string ranges = tape_.getReusableSlotsString();
            for (std::size_t i = 0; i < ranges.size(); ++i) mix(std::uint64_t(ranges[i]));
        }
        return hash_;
    }

    int duplicates() const { return duplicates_; }

    void computeAdjoint(tape_type* tape) override
    {
        ++depth_;
        std::vector<slot_type> input;
        allocate(input);
        {
            xad::ScopedNestedRecording<tape_type> nested(tape);
            // out of order only while no outer slots are reusable
            bool anyOrder = tape->getNumReusableSlotSections() == 0;
            std::vector<slot_type> live;
            for (int i = 0, n = int(rng_(60)); i < n; ++i)
            {
                unsigned r = rng_(100);
                if (r < 5 && depth_ < 2)
                    tape->insertCallback(this);
                else if (live.empty() || r < 60)
                    allocate(live);
                else
                    release(live, anyOrder ? rng_(unsigned(live.size())) : live.size() - 1);
            }
            tape->setDerivative(allocate(live), 1.0);
            nested.computeAdjoints();
            while (!live.empty())
                release(live, anyOrder ? rng_(unsigned(live.size())) : live.size() - 1);
        }
        release(input, 0);
        --depth_;
    }

  private:
    void mix(std::uint64_t v) { hash_ = (hash_ ^ v) * 1099511628211ULL; }

    slot_type allocate(std::vector<slot_type>& live)
    {
        slot_type s = tape_.registerVariable();
        tape_.pushLhs(s);
        mix(s);
        if (s >= isLive_.size())
            isLive_.resize(s + 1);
        duplicates_ += isLive_[s] ? 1 : 0;
        isLive_[s] = true;
        live.push_back(s);
        return s;
    }

    void release(std::vector<slot_type>& live, std::size_t i)
    {
        std::swap(live[i], live.back());
        isLive_[live.back()] = false;
        tape_.unregisterVariable(live.back());
        live.pop_back();
    }

    void sweep(std::vector<slot_type>& live)
    {
        tape_.setDerivative(allocate(live), 1.0);
        tape_.computeAdjoints();
    }

    // drops the reusable slots from the rollback variable on
    void rollBack(std::vector<slot_type>& live, std::vector<slot_type>& recent, slot_type pinned,
                  position_type pos)
    {
        // freeing a variable below last keeps some reusable slots too
        for (std::size_t i = 0; i < live.size(); ++i)
        {
            if (live[i] < pinned)
            {
                release(live, i);
                tape_.clearDerivativesAfter(pos);
                break;
            }
        }
        tape_.resetTo(pos);
        releaseAll(recent);
        tape_.newRecording();
    }

    // variables going out of scope, in a random order
    void releaseAll(std::vector<slot_type>& live)
    {
        while (!live.empty()) release(live, rng_(unsigned(live.size())));
    }

    tape_type tape_;
    SlotTestRng rng_;
    std::uint64_t hash_;
    std::vector<bool> isLive_;
    int duplicates_;
    int depth_;
};

}  // namespace

TEST(Tape, slotSequenceMatchesReference)
{
    SlotSequence sequence;
#ifdef XAD_TAPE_REUSE_SLOTS
    EXPECT_EQ(5042060922808247158ULL, sequence.run(300));
#else
    EXPECT_EQ(7513799870854399479ULL, sequence.run(300));
#endif
    EXPECT_EQ(0, sequence.duplicates());
}

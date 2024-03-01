/*******************************************************************************

   Unit tests for the tape itself

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

#include <XAD/Tape.hpp>
#include <gtest/gtest.h>
#include <array>
#include <cmath>

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

TEST(Tape, isMovable)
{
    xad::Tape<double> s(false);
    xad::Tape<double> s2 = std::move(s);  // move constructor
    EXPECT_FALSE(s2.isActive());

    xad::Tape<double> s3(true);
    s3 = std::move(s2);  // move assign
    EXPECT_FALSE(s3.isActive());
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
    s.pushRhs(std::cos(x1), x1s);
    s.pushRhs(x2, x1s);
    s.pushRhs(x1, x2s);
    s.pushLhs(zs);

    EXPECT_EQ(3U, s.getNumVariables());
    EXPECT_EQ(3U, s.getNumOperations());
    EXPECT_EQ(1U, s.getNumStatements());

    // set the derivative for output
    s.setDerivative(zs, 1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(0.0, s.getDerivative(x2s));
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(zs));

    // s.printStatus();

    // compute the other derivatives (adjoints)
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, s.getDerivative(x1s));
    EXPECT_DOUBLE_EQ(M_PI, s.getDerivative(x2s));

    // s.printStatus();
}

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
    s.pushRhs(std::cos(x1), x1s);
    s.pushRhs(x2, x1s);
    s.pushRhs(x1, x2s);
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
    s.pushRhs(std::exp(x1), x1s);
    s.pushRhs(1.0 / x2, x1s);
    s.pushRhs(-x1 / (x2 * x2), x2s);
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
    s.pushAll(zs, mul.data(), sl.data(), 3);
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
#endif

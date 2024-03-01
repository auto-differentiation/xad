/*******************************************************************************

   Unit tests for To-be-recorded analysis, where only what is needed gets
   recorded on tape.

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

#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;
typedef xad::Tape<double>::slot_type slot_type;

TEST(TBR, notRegisteredOnTapeAtStart)
{
    xad::Tape<double> s;
    xad::AD a(1.0);
    xad::AD b(2.0);
    EXPECT_THAT(a.getSlot(), Eq(slot_type(xad::AD::INVALID_SLOT)));
    EXPECT_THAT(b.getSlot(), Eq(slot_type(xad::AD::INVALID_SLOT)));
}

TEST(TBR, canRegisterOnTape)
{
    xad::Tape<double> s;
    xad::AD a(1.0);
    xad::AD b(2.0);

    s.registerInput(a);
    s.registerInput(b);

    xad::AD c = b + a;

    EXPECT_EQ(a.getSlot(), 0u);
    EXPECT_EQ(b.getSlot(), 1u);
    EXPECT_EQ(c.getSlot(), 2u);
}

TEST(TBR, copiesChangeSlot)
{
    xad::Tape<double> s;
    xad::AD a(1.0);

    s.registerInput(a);

    xad::AD b = a;
    xad::AD c;
    c = b;

    EXPECT_THAT(a.getSlot(), Ne(b.getSlot()));
    EXPECT_THAT(b.getSlot(), Ne(c.getSlot()));
}

TEST(TBR, movesDoNotChangeSlotAndKeepValue)
{
    xad::Tape<double> s;
    xad::AD a(1.0);

    s.registerInput(a);
    auto slot_a = a.getSlot();

    xad::AD b = std::move(a);
    xad::AD c;
    c = std::move(b);

    EXPECT_THAT(c.getSlot(), Eq(slot_a));
    EXPECT_THAT(c.getValue(), DoubleNear(1.0, 1e-9));
}

TEST(TBR, operationsOnUnregisteredVariablesDoNotAssignSlot)
{
    xad::Tape<double> s;
    xad::AD a(1.0);
    xad::AD b = a * a;

    EXPECT_THAT(a.getSlot(), Eq(slot_type(xad::AD::INVALID_SLOT)));
    EXPECT_THAT(b.getSlot(), Eq(slot_type(xad::AD::INVALID_SLOT)));
}

TEST(TBR, canRegisterVectorsOfInputs)
{
    xad::Tape<double> t;
    std::vector<xad::AD> v(3);
    t.registerInputs(v);

    EXPECT_THAT(v[0].getSlot(), Eq(0u));
    EXPECT_THAT(v[1].getSlot(), Eq(1u));
    EXPECT_THAT(v[2].getSlot(), Eq(2u));
}

TEST(TBR, canRegisterVectorsOfInputsIterator)
{
    xad::Tape<double> t;
    std::vector<xad::AD> v(3);
    t.registerInputs(v.begin(), v.end());

    EXPECT_THAT(v[0].getSlot(), Eq(0u));
    EXPECT_THAT(v[1].getSlot(), Eq(1u));
    EXPECT_THAT(v[2].getSlot(), Eq(2u));
}

TEST(TBR, CorrectDerivativesWhenOverwriting)
{
    xad::Tape<double> t;
    xad::AD in = 2.0;
    t.registerInput(in);
    t.newRecording();
    std::vector<xad::AD> x(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        x[i] = i < 2 ? 0.0 : xad::AD(in * i);
    }
    x[0] = x[0] + x[2];
    x[1] = x[1] + x[3];
    x[0] = x[0] * x[1];
    xad::AD out = x[0];
    t.registerOutput(out);
    derivative(out) = 1.0;
    t.printStatus();
    t.computeAdjoints();
    EXPECT_THAT(value(out), DoubleNear(24.0, 1e-9));
    EXPECT_THAT(derivative(in), DoubleNear(24.0, 1e-9));
}

TEST(TBR, SettingDerivativesOfNonDependentOutputsIsOk)
{
    xad::Tape<double> t;
    xad::AD in = 2.0;
    t.registerInput(in);
    t.newRecording();
    xad::AD out = in < 0.0 ? xad::AD(2. * in) : 100.0;
    t.registerOutput(out);
    derivative(out) = 1.0;
    t.computeAdjoints();
    EXPECT_THAT(value(out), DoubleNear(100.0, 1e-9));
    EXPECT_THAT(derivative(in), DoubleNear(0.0, 1e-9));
}

TEST(TBR, AliasedUnregisteredVariableWorks)
{
    // reproducer from Libor
    xad::Tape<double> t;
    xad::AD in = 0.1234;
    t.registerInput(in);
    t.newRecording();

    xad::AD v = 0.0;
    xad::AD con1 = 0.123;
    double lam = 0.41;
    double sqez = -0.223;
    double delta = 1.2;
    xad::AD in_local = in;
    v = v + (con1 * in_local) / (1.0 + delta * in_local);
    in_local = in_local * exp(con1 * v + lam * (sqez - 0.5 * con1));
    xad::AD out = in_local;

    t.registerOutput(out);
    derivative(out) = 1.0;
    t.computeAdjoints();
    EXPECT_THAT(value(out), DoubleNear(0.109993, 1e-6));
    EXPECT_THAT(derivative(in), DoubleNear(0.892612, 1e-6));
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

TEST(TBR, AssignToSelfWorksRegistered)
{
    xad::Tape<double> t;
    xad::AD in = 2.0;
    t.registerInput(in);
    auto s = in.getSlot();

    in = in;
    EXPECT_THAT(in.getSlot(), Eq(s));
}

TEST(TBR, AssignToSelfWorksUnRegistered)
{
    xad::AD in = 2.0;
    auto s = in.getSlot();
    in = in;
    EXPECT_THAT(in.getSlot(), Eq(s));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
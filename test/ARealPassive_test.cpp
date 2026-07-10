/*******************************************************************************

   Unit tests for AReal passive-mode semantics (no tape, or unrecorded
   variables while a tape is active). These pin down the behaviour of the
   slot-before-tape-lookup fast paths in constructors, assignments, and the
   destructor.

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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <vector>

// no tape anywhere: copies, assignments, expressions, and destruction must
// work on values alone and never mark anything for recording
TEST(ARealPassive, valueSemanticsWithoutTape)
{
    ASSERT_EQ(nullptr, xad::Tape<double>::getActive());

    xad::AD a(1.5);
    xad::AD b(a);  // copy ctor
    xad::AD c;
    c = b;  // copy assignment
    EXPECT_DOUBLE_EQ(1.5, c.getValue());
    c = 2.0;  // scalar assignment
    xad::AD d = a + b * c;  // expression ctor
    d = d * 2.0 + a;        // expression assignment

    EXPECT_DOUBLE_EQ(1.5, a.getValue());
    EXPECT_DOUBLE_EQ(1.5, b.getValue());
    EXPECT_DOUBLE_EQ(2.0, c.getValue());
    EXPECT_DOUBLE_EQ(2.0 * (1.5 + 1.5 * 2.0) + 1.5, d.getValue());
    EXPECT_FALSE(a.shouldRecord());
    EXPECT_FALSE(b.shouldRecord());
    EXPECT_FALSE(c.shouldRecord());
    EXPECT_FALSE(d.shouldRecord());
}

// with a tape active, copies and expressions of unrecorded variables must
// stay unrecorded
TEST(ARealPassive, passiveOperandsStayPassiveWithActiveTape)
{
    xad::Tape<double> s;
    xad::AD p1(1.0), p2(2.0);  // never registered

    xad::AD c(p1);          // copy ctor, passive source
    xad::AD e = p1 + p2;    // expression ctor, passive operands
    xad::AD f;
    f = p2;  // copy assignment, both passive
    EXPECT_DOUBLE_EQ(2.0, f.getValue());
    f = p1 * p2 + e;  // expression assignment, all passive

    EXPECT_FALSE(c.shouldRecord());
    EXPECT_FALSE(e.shouldRecord());
    EXPECT_FALSE(f.shouldRecord());
    EXPECT_DOUBLE_EQ(1.0, c.getValue());
    EXPECT_DOUBLE_EQ(3.0, e.getValue());
    EXPECT_DOUBLE_EQ(1.0 * 2.0 + 3.0, f.getValue());
}

// copies and expressions of recorded variables must record as before
TEST(ARealPassive, recordedOperandsRecordWithActiveTape)
{
    xad::Tape<double> s;
    xad::AD x(3.0);
    s.registerInput(x);
    s.newRecording();

    xad::AD y(x);           // copy ctor of recorded variable
    xad::AD z = x * 2.0;    // expression ctor with recorded operand
    EXPECT_TRUE(y.shouldRecord());
    EXPECT_TRUE(z.shouldRecord());

    derivative(z) = 1.0;
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(2.0, derivative(x));
}

// overwriting a recorded variable with a passive value must clear its
// dependency on the inputs (this exercises the this->shouldRecord() branch
// of the assignment fast paths)
TEST(ARealPassive, assigningPassiveValueToRecordedVariableBreaksDependency)
{
    xad::Tape<double> s;
    xad::AD x(3.0);
    s.registerInput(x);
    s.newRecording();

    xad::AD passive(7.0);

    xad::AD y = x * 2.0;
    y = passive;  // copy assignment, passive rhs onto recorded lhs
    derivative(y) = 1.0;
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(0.0, derivative(x));
    EXPECT_DOUBLE_EQ(7.0, y.getValue());

    s.clearDerivatives();
    xad::AD w = x * 2.0;
    w = passive + 1.0;  // expression assignment, passive rhs onto recorded lhs
    derivative(w) = 1.0;
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(0.0, derivative(x));
    EXPECT_DOUBLE_EQ(8.0, w.getValue());

    s.clearDerivatives();
    xad::AD v = x * 2.0;
    v = 5.0;  // scalar assignment onto recorded lhs
    derivative(v) = 1.0;
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(0.0, derivative(x));
    EXPECT_DOUBLE_EQ(5.0, v.getValue());
}

// destroying registered variables when the tape has been deactivated (or
// destroyed) must be safe - the destructor checks the slot before the tape
TEST(ARealPassive, destructionAfterTapeDeactivation)
{
    xad::AD* leaked = nullptr;
    {
        xad::Tape<double> s;
        auto* x = new xad::AD(1.0);
        s.registerInput(*x);
        EXPECT_TRUE(x->shouldRecord());
        s.deactivate();
        delete x;  // slot valid, no active tape: must not crash
        s.activate();
        leaked = new xad::AD(2.0);
        s.registerInput(*leaked);
    }
    // tape destroyed entirely; variable with a valid slot outlives it
    EXPECT_EQ(nullptr, xad::Tape<double>::getActive());
    delete leaked;  // must not crash
}

// containers of passive variables with an active tape: bulk copies and
// destruction stay value-only
TEST(ARealPassive, vectorOfPassiveVariablesWithActiveTape)
{
    xad::Tape<double> s;
    std::vector<xad::AD> v(100, xad::AD(1.25));
    std::vector<xad::AD> w = v;  // copy construct all elements
    w.resize(150, xad::AD(2.5));
    double sum = 0.0;
    for (const auto& e : w) sum += e.getValue();
    EXPECT_DOUBLE_EQ(100 * 1.25 + 50 * 2.5, sum);
    for (const auto& e : w) EXPECT_FALSE(e.shouldRecord());
}

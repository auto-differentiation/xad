/*******************************************************************************

   Unit tests for To-be-recorded analysis, where only what is needed gets
   recorded on tape.

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

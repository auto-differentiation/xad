/*******************************************************************************

   Tests for exceptions.

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

TEST(Exceptions, alreadyActive)
{
    xad::Tape<double> t;

    EXPECT_THROW(xad::Tape<double>(), xad::TapeAlreadyActive);

    xad::Tape<double> t2(false);
    EXPECT_THROW(t2.activate(), xad::TapeAlreadyActive);

    t.deactivate();
    EXPECT_NO_THROW(t2.activate());
}

TEST(Exceptions, adjointsNotInitialized)
{
    xad::Tape<double> t;
    xad::AD x = 1.0;
    t.registerInput(x);
    t.newRecording();
    xad::AD y = x * x;
    EXPECT_THROW(t.computeAdjoints(), xad::DerivativesNotInitialized);
    derivative(y) = 1.0;
    t.computeAdjoints();
}

TEST(Exceptions, popCallback)
{
    xad::Tape<double> t;
    EXPECT_THROW(t.popCallback(), xad::OutOfRange);
}

TEST(Exceptions, getDerivatives)
{
    xad::Tape<double> t;
    xad::AD x = 1.0;
    t.registerInput(x);

    EXPECT_NO_THROW(t.derivative(x.getSlot()));
    EXPECT_THROW(t.derivative(12312), xad::OutOfRange);
    EXPECT_THROW(t.setDerivative(1231, 0.0), xad::OutOfRange);
}

TEST(Exceptions, checkpoints)
{
    xad::Tape<double> t;
    xad::AD x = 1.0;
    t.registerInput(x);
    t.registerOutput(x);
    x.setDerivative(1.0);

    EXPECT_NO_THROW(t.getAndResetOutputAdjoint(x.getSlot()));
    EXPECT_THROW(t.getAndResetOutputAdjoint(12312), xad::OutOfRange);
    EXPECT_NO_THROW(t.incrementAdjoint(x.getSlot(), 1.0));
    EXPECT_THROW(t.incrementAdjoint(12312, 1.0), xad::OutOfRange);
}

TEST(Exceptions, noTape)
{
    xad::AD x = 1.0;
    EXPECT_THROW(x.setDerivative(1.0), xad::NoTapeException);
    EXPECT_THROW(derivative(x) = 1.0, xad::NoTapeException);
    const xad::AD y = 1.0;
    EXPECT_THROW(y.derivative(), xad::NoTapeException);
    xad::Tape<double> t;
    EXPECT_NO_THROW(x.setDerivative(1.0));
    EXPECT_NO_THROW(derivative(x) = 1.0);
    EXPECT_NO_THROW(x.derivative());
}

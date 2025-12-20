/*******************************************************************************
 *
 *   Tutorial-style tests for JIT branching behavior.
 *
 *   These are not performance tests; they serve as a compact, executable example
 *   showing how to express conditional logic for JIT graph reuse.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2025 Xcelerit Computing Ltd.
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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>

#ifdef XAD_ENABLE_JIT

namespace
{

template <class AD>
AD piecewise_plain_if(const AD& x)
{
    // Normal C++ control flow: evaluated immediately during recording.
    if (xad::value(x) < 2.0)
        return 1.0 * x;
    return 7.0 * x;
}

template <class AD>
AD piecewise_abool_if(const AD& x)
{
    // JIT-friendly: record both branches and select at runtime.
    auto cond = xad::less(x, 2.0);
    AD t = 1.0 * x;
    AD f = 7.0 * x;
    return cond.If(t, f);
}

}  // namespace

TEST(JITTutorial, TapePlainIfReevalProducesDifferentBranches)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    auto eval = [](double x0) {
        tape_type tape;
        AD x = x0;
        tape.registerInput(x);

        tape.newRecording();
        AD y = piecewise_plain_if(x);
        tape.registerOutput(y);
        xad::derivative(y) = 1.0;
        tape.computeAdjoints();

        return std::make_pair(xad::value(y), xad::derivative(x));
    };

    auto r1 = eval(1.0);
    EXPECT_DOUBLE_EQ(1.0, r1.first);
    EXPECT_DOUBLE_EQ(1.0, r1.second);

    auto r3 = eval(3.0);
    EXPECT_DOUBLE_EQ(21.0, r3.first);
    EXPECT_DOUBLE_EQ(7.0, r3.second);
}

TEST(JITTutorial, JitPlainIfIsBakedInAtRecordTime)
{
    using AD = xad::AReal<double, 1>;
    xad::JITCompiler<double, 1> jit;

    AD x = 1.0;
    jit.registerInput(x);

    // Record at x=1.0 => takes the "<2" branch and bakes it into the graph.
    AD y = piecewise_plain_if(x);
    jit.registerOutput(y);
    jit.compile();

    double out = 0.0;

    x = 1.0;
    jit.clearDerivatives();
    jit.forward(&out, 1);
    jit.setDerivative(y.getSlot(), 1.0);
    jit.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, out);
    EXPECT_DOUBLE_EQ(1.0, jit.getDerivative(x.getSlot()));

    // Replay at x=3.0 without re-recording => still uses the recorded branch.
    x = 3.0;
    jit.clearDerivatives();
    jit.forward(&out, 1);
    jit.setDerivative(y.getSlot(), 1.0);
    jit.computeAdjoints();
    EXPECT_DOUBLE_EQ(3.0, out);
    EXPECT_DOUBLE_EQ(1.0, jit.getDerivative(x.getSlot()));
}

TEST(JITTutorial, JitABoolIfAllowsBranchToVaryPerReplay)
{
    using AD = xad::AReal<double, 1>;
    xad::JITCompiler<double, 1> jit;

    AD x = 1.0;
    jit.registerInput(x);

    AD y = piecewise_abool_if(x);
    jit.registerOutput(y);
    jit.compile();

    double out = 0.0;

    x = 1.0;
    jit.clearDerivatives();
    jit.forward(&out, 1);
    jit.setDerivative(y.getSlot(), 1.0);
    jit.computeAdjoints();
    EXPECT_DOUBLE_EQ(1.0, out);
    EXPECT_DOUBLE_EQ(1.0, jit.getDerivative(x.getSlot()));

    x = 3.0;
    jit.clearDerivatives();
    jit.forward(&out, 1);
    jit.setDerivative(y.getSlot(), 1.0);
    jit.computeAdjoints();
    EXPECT_DOUBLE_EQ(21.0, out);
    EXPECT_DOUBLE_EQ(7.0, jit.getDerivative(x.getSlot()));
}

#endif  // XAD_ENABLE_JIT



/*******************************************************************************

   Unit tests for ABool

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
#include <cmath>
#include <memory>
#include <utility>

#ifdef XAD_ENABLE_GRAPH
TEST(GraphABool, defaultConstructor)
{
    xad::ABool<double> ab;
    EXPECT_FALSE(ab.passive());
    EXPECT_FALSE(ab.hasSlot());
    // Compare with local copy to avoid ODR-use of static constexpr member
    constexpr auto invalid_slot = xad::ABool<double>::INVALID_SLOT;
    EXPECT_EQ(invalid_slot, ab.slot());
}

TEST(GraphABool, constructorFromBool)
{
    xad::ABool<double> ab_true(true);
    xad::ABool<double> ab_false(false);

    EXPECT_TRUE(ab_true.passive());
    EXPECT_FALSE(ab_false.passive());
    EXPECT_FALSE(ab_true.hasSlot());
    EXPECT_FALSE(ab_false.hasSlot());
}

TEST(GraphABool, constructorWithSlot)
{
    xad::ABool<double> ab(42, true);

    EXPECT_TRUE(ab.passive());
    EXPECT_TRUE(ab.hasSlot());
    EXPECT_EQ(42u, ab.slot());
}

TEST(GraphABool, implicitBoolConversion)
{
    xad::ABool<double> ab_true(true);
    xad::ABool<double> ab_false(false);

    // Test implicit conversion
    if (ab_true)
        EXPECT_TRUE(true);
    else
        FAIL() << "ABool(true) should convert to true";

    if (ab_false)
        FAIL() << "ABool(false) should convert to false";
    else
        EXPECT_TRUE(true);
}

TEST(GraphABool, ifWithoutGraphRecording)
{
    // Without an active graph recorder, If should use passive value
    using AD = xad::AReal<double, 1>;
    AD trueVal(10.0);
    AD falseVal(20.0);

    xad::ABool<double> cond_true(true);
    xad::ABool<double> cond_false(false);

    AD result_true = cond_true.If(trueVal, falseVal);
    AD result_false = cond_false.If(trueVal, falseVal);

    EXPECT_DOUBLE_EQ(10.0, xad::value(result_true));
    EXPECT_DOUBLE_EQ(20.0, xad::value(result_false));
}

TEST(GraphABool, staticIfWithoutGraphRecording)
{
    using AD = xad::AReal<double, 1>;
    AD trueVal(10.0);
    AD falseVal(20.0);

    xad::ABool<double> cond_true(true);
    xad::ABool<double> cond_false(false);

    AD result_true = xad::ABool<double>::If(cond_true, trueVal, falseVal);
    AD result_false = xad::ABool<double>::If(cond_false, trueVal, falseVal);

    EXPECT_DOUBLE_EQ(10.0, xad::value(result_true));
    EXPECT_DOUBLE_EQ(20.0, xad::value(result_false));
}

TEST(GraphABool, lessComparison)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    AD b = 3.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    auto cond = xad::less(a, b);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_TRUE(cond.hasSlot());  // a graph recorder is active, so slot should be set
}

TEST(GraphABool, lessComparisonWithScalar)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    recorder.registerInput(a);

    auto cond = xad::less(a, 3.0);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(GraphABool, greaterComparison)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    AD b = 3.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    auto cond = xad::greater(a, b);
    EXPECT_TRUE(cond.passive());  // 5 > 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(GraphABool, lessEqualComparison)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 3.0;
    AD b = 3.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    auto cond = xad::lessEqual(a, b);
    EXPECT_TRUE(cond.passive());  // 3 <= 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(GraphABool, greaterEqualComparison)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    AD b = 3.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    auto cond = xad::greaterEqual(a, b);
    EXPECT_TRUE(cond.passive());  // 5 >= 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(GraphABool, ifWithGraphRecording)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    recorder.registerInput(x);
    recorder.newRecording();

    AD trueVal = x * 2.0;   // 4.0
    AD falseVal = x * 3.0;  // 6.0

    auto cond = xad::less(x, 5.0);  // true for x=2
    AD result = cond.If(trueVal, falseVal);
    recorder.registerOutput(result);

    recorder.compile();
    double output;
    recorder.forward(&output);

    EXPECT_DOUBLE_EQ(4.0, output);  // x < 5, so trueVal = 2*2 = 4
}

TEST(GraphABool, ifDerivativeTrueBranch)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    recorder.registerInput(x);
    recorder.newRecording();

    AD trueVal = x * x;     // x^2, derivative = 2x
    AD falseVal = x * 3.0;  // 3x, derivative = 3

    auto cond = xad::less(x, 5.0);  // true for x=2
    AD result = cond.If(trueVal, falseVal);
    recorder.registerOutput(result);

    recorder.compile();
    recorder.setDerivative(result.getSlot(), 1.0);
    recorder.computeAdjoints();

    // Since x=2 < 5, we take the true branch (x^2)
    // d(x^2)/dx = 2x = 4
    EXPECT_NEAR(4.0, recorder.getDerivative(x.getSlot()), 1e-10);
}

TEST(GraphABool, ifDerivativeFalseBranch)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD x = 10.0;
    recorder.registerInput(x);
    recorder.newRecording();

    AD trueVal = x * x;     // x^2, derivative = 2x
    AD falseVal = x * 3.0;  // 3x, derivative = 3

    auto cond = xad::less(x, 5.0);  // false for x=10
    AD result = cond.If(trueVal, falseVal);
    recorder.registerOutput(result);

    recorder.compile();
    recorder.setDerivative(result.getSlot(), 1.0);
    recorder.computeAdjoints();

    // Since x=10 >= 5, we take the false branch (3x)
    // d(3x)/dx = 3
    EXPECT_NEAR(3.0, recorder.getDerivative(x.getSlot()), 1e-10);
}

TEST(GraphABool, comparisonWithoutGraphRecording)
{
    // Test comparison functions when no graph recorder is active
    // Should return ABool with passive value but no slot
    using AD = xad::AReal<double, 1>;

    AD a(2.0);
    AD b(3.0);

    // No graph recorder active - comparisons should work but not have slots
    auto cond = xad::less(a, b);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_FALSE(cond.hasSlot()); // No graph recorder, so no slot

    auto cond2 = xad::greater(a, 1.0);
    EXPECT_TRUE(cond2.passive());  // 2 > 1 is true
    EXPECT_FALSE(cond2.hasSlot());
}

// =============================================================================
// Additional GraphRecorder tests
// =============================================================================

#endif  // XAD_ENABLE_GRAPH

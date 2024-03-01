/*******************************************************************************

   Unit tests for math function derivatives (Part 2 - split due to long compile
   times).

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
#include <gtest/gtest.h>
#include "TestHelpers.hpp"

LOCAL_TEST_FUNCTOR1(powScalarBaseAD, pow(2.1, x))
TEST(ExpressionsMath, powScalarBaseAD)
{
    mathTest_all(0.3, std::pow(2.1, 0.3), std::log(2.1) * std::pow(2.1, 0.3),
                 std::pow(2.1, 0.3) * std::log(2.1) * std::log(2.1), powScalarBaseAD);
}

LOCAL_TEST_FUNCTOR1(powScalarBaseExpr, pow(2.1, 2.3 * x))
TEST(ExpressionsMath, powScalarBaseExpr)
{
    mathTest_all(0.3, std::pow(2.1, 2.3 * 0.3), 2.3 * std::log(2.1) * std::pow(2.1, 2.3 * 0.3),
                 std::pow(2.1, 2.3 * 0.3) * std::log(2.1) * std::log(2.1) * 2.3 * 2.3,
                 powScalarBaseExpr);
}

LOCAL_TEST_FUNCTOR1(powScalarExpAD, pow(x, 2.1))
TEST(ExpressionsMath, powScalarExpAD)
{
    mathTest_all(0.3, std::pow(0.3, 2.1), 2.1 * std::pow(0.3, 1.1), 1.1 * 2.1 * std::pow(0.3, 0.1),
                 powScalarExpAD);
}

LOCAL_TEST_FUNCTOR1(powScalarExpExpr, pow(2.3 * x, 2.1))
TEST(ExpressionsMath, powScalarExpExpr)
{
    mathTest_all(0.3, std::pow(2.3 * 0.3, 2.1), 2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1),
                 2.3 * 2.3 * 1.1 * 2.1 * std::pow(2.3 * 0.3, 0.1), powScalarExpExpr);
}

LOCAL_TEST_FUNCTOR2(powADAD, pow(x1, x2))
TEST(ExpressionsMath, powADAD)
{
    mathTest2_all(0.3, 2.1, std::pow(0.3, 2.1), 2.1 * std::pow(0.3, 1.1),         // d1
                  std::log(0.3) * std::pow(0.3, 2.1),                             // d2
                  2.1 * 1.1 * std::pow(0.3, 0.1),                                 // d11
                  2.1 * std::log(0.3) * std::pow(0.3, 1.1) + std::pow(0.3, 1.1),  // d12
                  2.1 * std::log(0.3) * std::pow(0.3, 1.1) + std::pow(0.3, 1.1),  // d21
                  std::log(0.3) * std::log(0.3) * std::pow(0.3, 2.1),             // d22
                  powADAD);
}

LOCAL_TEST_FUNCTOR2(powADExpr, pow(x1, 2.3 * x2))
TEST(ExpressionsMath, powADExpr)
{
    mathTest2_all(1.3, 2.1, std::pow(1.3, 2.3 * 2.1),
                  2.3 * 2.1 * std::pow(1.3, 2.3 * 2.1 - 1.0),                             // d1
                  2.3 * std::log(1.3) * std::pow(1.3, 2.3 * 2.1),                         // d2
                  2.3 * 2.1 * (2.3 * 2.1 - 1.0) * std::pow(1.3, 2.3 * 2.1 - 2.0),         // d11
                  2.3 * std::pow(1.3, 2.3 * 2.1 - 1) * (2.3 * std::log(1.3) * 2.1 + 1.),  // d12
                  2.3 * std::pow(1.3, 2.3 * 2.1 - 1) * (2.3 * std::log(1.3) * 2.1 + 1.),  // d21
                  2.3 * std::log(1.3) * 2.3 * std::log(1.3) * std::pow(1.3, 2.3 * 2.1),   // d22
                  powADExpr);
}

LOCAL_TEST_FUNCTOR2(powExprAD, pow(2.3 * x1, x2))
TEST(ExpressionsMath, powExprAD)
{
    mathTest2_all(
        0.3, 2.1, std::pow(2.3 * 0.3, 2.1), 2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1),
        std::log(2.3 * 0.3) * std::pow(2.3 * 0.3, 2.1),
        2.3 * 2.1 * 2.3 * 1.1 * std::pow(2.3 * 0.3, 0.1),  // d11
        2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1) * std::log(2.3 * .3) +
            2.3 * std::pow(2.3 * 0.3,
                           1.1),  // 1./.3*std::pow(2.3*.3,2.1)*(std::log(2.3*.3)*2.1+1), // d12
        2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1) * std::log(2.3 * .3) +
            2.3 * std::pow(2.3 * 0.3, 1.1),                                    // d21
        std::log(2.3 * 0.3) * std::log(2.3 * 0.3) * std::pow(2.3 * 0.3, 2.1),  // d22
        powExprAD);
}

LOCAL_TEST_FUNCTOR2(powExprExpr, pow(1.2 * x1, 2.3 * x2))
TEST(ExpressionsMath, powExprExpr)
{
    mathTest2_all(
        0.3, 2.1, std::pow(1.2 * 0.3, 2.3 * 2.1),
        1.2 * 2.3 * 2.1 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0),
        2.3 * std::log(1.2 * 0.3) * std::pow(1.2 * 0.3, 2.3 * 2.1),
        1.2 * (2.3 * 2.1 - 1.0) * 1.2 * 2.3 * 2.1 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 2.0),  // d11
        1.2 * 2.3 * 2.3 * 2.1 * std::log(1.2 * .3) * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0) +
            1.2 * 2.3 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0),
        1.2 * 2.3 * 2.3 * 2.1 * std::log(1.2 * .3) * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0) +
            1.2 * 2.3 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0),
        2.3 * std::log(1.2 * 0.3) * 2.3 * std::log(1.2 * 0.3) *
            std::pow(1.2 * 0.3, 2.3 * 2.1),  // d22
        powExprExpr);
}

LOCAL_TEST_FUNCTOR1(pownAD, pown(x, 2))
TEST(ExpressionsMath, pownAD) { mathTest_all_aad(0.3, std::pow(0.3, 2), 2. * 0.3, 2., pownAD); }

LOCAL_TEST_FUNCTOR1(pownExpr, pown(2.3 * x, 2))
TEST(ExpressionsMath, pownExpr)
{
    mathTest_all_aad(0.3, std::pow(2.3 * 0.3, 2), 2.3 * 2 * 2.3 * 0.3, 2.3 * 2. * 2.3, pownExpr);
}

LOCAL_TEST_FUNCTOR1(pown1AD, pow(x, 2))
TEST(ExpressionsMath, pown1AD)
{
    mathTest_all(0.3, std::pow(0.3, 2), 2 * std::pow(0.3, 1), 2. * std::pow(0.3, 0), pown1AD);
}

LOCAL_TEST_FUNCTOR1(pown1Expr, pow(2.3 * x, 2))
TEST(ExpressionsMath, pown1Expr)
{
    mathTest_all(0.3, std::pow(2.3 * 0.3, 2), 2.3 * 2 * std::pow(2.3 * 0.3, 1),
                 2.3 * 2.3 * 2 * 1 * std::pow(2.3 * 0.3, 0), pown1Expr);
}

LOCAL_TEST_FUNCTOR1(sqrtAD, sqrt(x))
TEST(ExpressionsMath, sqrtAD)
{
    mathTest_all(0.3, std::sqrt(0.3), 0.5 / std::sqrt(0.3), -0.5 * 0.5 / std::pow(0.3, 3. / 2.),
                 sqrtAD);
}

LOCAL_TEST_FUNCTOR1(sqrtExpr, sqrt(2.3 * x))
TEST(ExpressionsMath, sqrtExpr)
{
    mathTest_all(0.3, std::sqrt(2.3 * 0.3), 2.3 * 0.5 / std::sqrt(2.3 * 0.3),
                 2.3 * 0.5 * 2.3 * -0.5 / std::pow(2.3 * 0.3, 3. / 2.), sqrtExpr);
}

LOCAL_TEST_FUNCTOR1(absAD, abs(x))
TEST(ExpressionsMath, absAD)
{
    mathTest_all(0.3, std::abs(0.3), 1.0, 0.0, absAD);
    mathTest_all(-0.3, std::abs(-0.3), -1.0, 0.0, absAD);
    mathTest_all(0.0, std::abs(0.0), 0.0, 0.0, absAD);
}

LOCAL_TEST_FUNCTOR1(absExpr, abs(2.3 * x))
TEST(ExpressionsMath, absExpr)
{
    mathTest_all(0.3, std::abs(2.3 * 0.3), 2.3, 0.0, absExpr);
    mathTest_all(-0.3, std::abs(2.3 * -0.3), -2.3, 0.0, absExpr);
    mathTest_all(0.0, std::abs(2.3 * 0.0), 0.0, 0.0, absExpr);
}

LOCAL_TEST_FUNCTOR1(fabsAD, fabs(x))
TEST(ExpressionsMath, fabsAD)
{
    mathTest_all(0.3, std::fabs(0.3), 1.0, 0.0, fabsAD);
    mathTest_all(-0.3, std::fabs(-0.3), -1.0, 0.0, fabsAD);
    mathTest_all(0.0, std::fabs(0.0), 0.0, 0.0, fabsAD);
}

LOCAL_TEST_FUNCTOR1(fabsExpr, fabs(2.3 * x))
TEST(ExpressionsMath, fabsExpr)
{
    mathTest_all(0.3, std::fabs(2.3 * 0.3), 2.3, 0.0, fabsExpr);
    mathTest_all(-0.3, std::fabs(2.3 * -0.3), -2.3, 0.0, fabsExpr);
    mathTest_all(0.0, std::fabs(2.3 * 0.0), 0.0, 0.0, fabsExpr);
}

LOCAL_TEST_FUNCTOR1(sabsAD, smooth_abs(x))
TEST(ExpressionsMath, sabsAD)
{
    double c = 0.001;
    mathTest_all_aad(0.3, std::abs(0.3), 1.0, 0.0, sabsAD);
    mathTest_all_aad(-0.3, std::abs(-0.3), -1.0, 0.0, sabsAD);
    mathTest_all_aad(0.0, std::abs(0.0), 0.0, 4. / c, sabsAD);
}

LOCAL_TEST_FUNCTOR1(sabsExpr, smooth_abs(2.3 * x))
TEST(ExpressionsMath, sabsExpr)
{
    mathTest_all_aad(0.3, std::abs(2.3 * 0.3), 2.3, 0.0, sabsExpr);
    mathTest_all_aad(-0.3, std::abs(2.3 * -0.3), -2.3, 0.0, sabsExpr);
    double c = 0.001;
    mathTest_all_aad(0.0, std::abs(2.3 * 0.0), 0.0, 2.3 * 2.3 * 4. / c, sabsExpr);
}

LOCAL_TEST_FUNCTOR2(sabsADAD, smooth_abs(x1, x2))
TEST(ExpressionsMath, sabsADAD)
{
    mathTest2_all_aad(0.3, 0.001, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sabsADAD);
    mathTest2_all_aad(-0.3, 0.001, 0.3, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sabsADAD);
    mathTest2_all_aad(0.0, 0.001, 0.0, 0.0, 0.0, 4. / 0.001, 0.0, 0.0, 0.0, sabsADAD);
}

LOCAL_TEST_FUNCTOR2(sabsADExpr, smooth_abs(x1, 2.3 * x2))
TEST(ExpressionsMath, sabsADExpr)
{
    mathTest2_all_aad(0.3, 0.001, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sabsADExpr);
    mathTest2_all_aad(-0.3, 0.001, 0.3, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sabsADExpr);
    mathTest2_all_aad(0.0, 0.001, 0.0, 0.0, 0.0, 4. / 2.3 / 0.001, 0.0, 0.0, 0.0, sabsADExpr);
}

LOCAL_TEST_FUNCTOR2(sabsExprAD, smooth_abs(2.3 * x1, x2))
TEST(ExpressionsMath, sabsExprAD)
{
    mathTest2_all_aad(0.3, 0.001, 2.3 * 0.3, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, sabsExprAD);
    mathTest2_all_aad(-0.3, 0.001, 2.3 * 0.3, -2.3, 0.0, 0.0, 0.0, 0.0, 0.0, sabsExprAD);
    mathTest2_all_aad(0.0, 0.001, 0.0, 0.0, 0.0, 2.3 * 2.3 * 4. / 0.001, 0.0, 0.0, 0.0, sabsExprAD);
}

LOCAL_TEST_FUNCTOR2(sabsExprExpr, smooth_abs(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMath, sabsExprExpr)
{
    mathTest2_all_aad(0.3, 0.001, 2.3 * 0.3, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, sabsExprExpr);
    mathTest2_all_aad(-0.3, 0.001, 2.3 * 0.3, -2.3, 0.0, 0.0, 0.0, 0.0, 0.0, sabsExprExpr);
    mathTest2_all_aad(0.0, 0.001, 0.0, 0.0, 0.0, 2.3 * 2.3 * 4. / 2.3 / 0.001, 0.0, 0.0, 0.0,
                      sabsExprExpr);
}

LOCAL_TEST_FUNCTOR1(floorAD, floor(x))
LOCAL_TEST_FUNCTOR1(ceilAD, ceil(x))
LOCAL_TEST_FUNCTOR1(truncAD, trunc(x))
LOCAL_TEST_FUNCTOR1(roundAD, round(x))
TEST(ExpressionsMath, ceil_floor_trunc_round_AD)
{
    using xad::round;
    using xad::trunc;
    mathTest_all(1.3, std::floor(1.3), 0.0, 0.0, floorAD);
    mathTest_all(1.3, std::ceil(1.3), 0.0, 0.0, ceilAD);
    mathTest_all(1.3, trunc(1.3), 0.0, 0.0, truncAD);
    mathTest_all(1.3, round(1.3), 0.0, 0.0, roundAD);
    mathTest_all(1.0, std::floor(1.0), 0.0, 0.0, floorAD);
    mathTest_all(1.0, std::ceil(1.0), 0.0, 0.0, ceilAD);
    mathTest_all(1.0, trunc(1.0), 0.0, 0.0, truncAD);
    mathTest_all(1.0, round(1.0), 0.0, 0.0, roundAD);
}

LOCAL_TEST_FUNCTOR1(floorExpr, floor(2.3 * x))
LOCAL_TEST_FUNCTOR1(ceilExpr, ceil(2.3 * x))
LOCAL_TEST_FUNCTOR1(truncExpr, trunc(2.3 * x))
LOCAL_TEST_FUNCTOR1(roundExpr, round(2.3 * x))
TEST(ExpressionsMath, ceil_floor_trunc_round_Expr)
{
    using xad::round;
    using xad::trunc;

    mathTest_all(1.3, std::floor(2.3 * 1.3), 0.0, 0.0, floorExpr);
    mathTest_all(1.3, std::ceil(2.3 * 1.3), 0.0, 0.0, ceilExpr);
    mathTest_all(1.3, trunc(2.3 * 1.3), 0.0, 0.0, truncExpr);
    mathTest_all(1.3, round(2.3 * 1.3), 0.0, 0.0, roundExpr);
}

#ifndef __FAST_MATH__
TEST(ExpressionsMath, isnan_inf_finite)
{
    xad::Tape<double> s;
    xad::AD x1 = 1.2, x2 = std::numeric_limits<double>::infinity(),
            x3 = std::numeric_limits<double>::quiet_NaN(), x4 = 0.0;
    EXPECT_FALSE(isinf(x1));
    EXPECT_FALSE(isinf(x1 * 2.3));
    EXPECT_TRUE(isinf(x2));
    EXPECT_TRUE(isinf(x2 * 2.3));
    EXPECT_FALSE(isnan(x1));
    EXPECT_FALSE(isnan(x1 * 2.3));
    EXPECT_FALSE(isnan(x2));
    EXPECT_TRUE(isnan(x3));
    EXPECT_TRUE(isfinite(x1));
    EXPECT_TRUE(isfinite(x1 * 2.3));
    EXPECT_FALSE(isfinite(x2));
    EXPECT_FALSE(isfinite(x2 * x3));
    // EXPECT_FALSE(__isinf(x1));
    // EXPECT_FALSE(__isinf(x1*2.3));
    // EXPECT_TRUE(__isinf(x2));
    // EXPECT_TRUE(__isinf(x2*2.3));
    // EXPECT_FALSE(__isnan(x1));
    // EXPECT_FALSE(__isnan(x1*2.3));
    // EXPECT_FALSE(__isnan(x2));
    // EXPECT_TRUE(__isnan(x3));
    // EXPECT_TRUE(__isfinite(x1));
    // EXPECT_TRUE(__isfinite(x1*2.3));
    // EXPECT_FALSE(__isfinite(x2));
    // EXPECT_FALSE(__isfinite(x2*x3));
    EXPECT_EQ(FP_NORMAL, fpclassify(x1));
    EXPECT_EQ(FP_INFINITE, fpclassify(x2));
    EXPECT_EQ(FP_NAN, fpclassify(x3));
    EXPECT_EQ(FP_ZERO, fpclassify(0.0 * x1));
    EXPECT_FALSE(signbit(x1));
    EXPECT_TRUE(signbit(-x1));
    EXPECT_FALSE(signbit(x4));
    EXPECT_TRUE(signbit(-x4));
}

TEST(ExpressionsMath, isnan_inf_finite_fwd)
{
    xad::FAD x1 = 1.2, x2 = std::numeric_limits<double>::infinity(),
             x3 = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(isinf(x1));
    EXPECT_FALSE(isinf(x1 * 2.3));
    EXPECT_TRUE(isinf(x2));
    EXPECT_TRUE(isinf(x2 * 2.3));
    EXPECT_FALSE(isnan(x1));
    EXPECT_FALSE(isnan(x1 * 2.3));
    EXPECT_FALSE(isnan(x2));
    EXPECT_TRUE(isnan(x3));
    EXPECT_TRUE(isfinite(x1));
    EXPECT_TRUE(isfinite(x1 * 2.3));
    EXPECT_FALSE(isfinite(x2));
    EXPECT_FALSE(isfinite(x2 * x3));
    // EXPECT_FALSE(__isinf(x1));
    // EXPECT_FALSE(__isinf(x1*2.3));
    // EXPECT_TRUE(__isinf(x2));
    // EXPECT_TRUE(__isinf(x2*2.3));
    // EXPECT_FALSE(__isnan(x1));
    // EXPECT_FALSE(__isnan(x1*2.3));
    // EXPECT_FALSE(__isnan(x2));
    // EXPECT_TRUE(__isnan(x3));
    // EXPECT_TRUE(__isfinite(x1));
    // EXPECT_TRUE(__isfinite(x1*2.3));
    // EXPECT_FALSE(__isfinite(x2));
    // EXPECT_FALSE(__isfinite(x2*x3));
    EXPECT_EQ(FP_NORMAL, fpclassify(x1));
    EXPECT_EQ(FP_INFINITE, fpclassify(x2));
    EXPECT_EQ(FP_NAN, fpclassify(x3));
    EXPECT_EQ(FP_ZERO, fpclassify(0.0 * x1));
}
#endif

// TODO: not in docs that fmod may cause lookup problems with VS 2012 or below
LOCAL_TEST_FUNCTOR2(fmodAD, xad::fmod(x1, x2))
LOCAL_TEST_FUNCTOR1(fmodADScalar, xad::fmod(x, 0.5))
LOCAL_TEST_FUNCTOR1(fmodScalarAD, xad::fmod(1.3, x))
TEST(ExpressionsMath, fmod_AD)
{
    mathTest2_all(1.3, 0.5, std::fmod(1.3, 0.5), 1.0, -std::floor(1.3 / 0.5), 0.0, 0.0, 0.0, 0.0,
                  fmodAD);
    mathTest_all(1.3, std::fmod(1.3, 0.5), 1.0, 0.0, fmodADScalar);
    mathTest_all(0.5, std::fmod(1.3, 0.5), -std::floor(1.3 / 0.5), 0.0, fmodScalarAD);
}

LOCAL_TEST_FUNCTOR2(fmodExprExpr, xad::fmod(2.3 * x1, 2.3 * x2))
LOCAL_TEST_FUNCTOR2(fmodADExpr, xad::fmod(x1, 2.3 * x2))
LOCAL_TEST_FUNCTOR2(fmodExprAD, xad::fmod(2.3 * x1, x2))
LOCAL_TEST_FUNCTOR1(fmodExprScalar, xad::fmod(2.3 * x, 0.5))
LOCAL_TEST_FUNCTOR1(fmodScalarExpr, xad::fmod(1.3, 2.3 * x))
TEST(ExpressionsMath, fmod_Expr)
{
    mathTest2_all(1.3, 0.5, std::fmod(2.3 * 1.3, 2.3 * 0.5), 2.3, -2.3 * std::floor(1.3 / 0.5), 0.0,
                  0.0, 0.0, 0.0, fmodExprExpr);
    mathTest2_all(1.3, 0.5, std::fmod(1.3, 2.3 * 0.5), 1.0, -2.3 * std::floor(1.3 / 0.5 / 2.3), 0.0,
                  0.0, 0.0, 0.0, fmodADExpr);
    mathTest2_all(1.3, 0.5, std::fmod(2.3 * 1.3, 0.5), 2.3, -std::floor(2.3 * 1.3 / 0.5), 0.0, 0.0,
                  0.0, 0.0, fmodExprAD);
    mathTest_all(1.3, std::fmod(2.3 * 1.3, 0.5), 2.3, 0.0, fmodExprScalar);
    mathTest_all(0.5, std::fmod(1.3, 2.3 * 0.5), -2.3 * std::floor(1.3 / 0.5 / 2.3), 0.0,
                 fmodScalarExpr);
}

LOCAL_TEST_FUNCTOR2(remainderAD, remainder(x1, x2))
LOCAL_TEST_FUNCTOR1(remainderADScalar, remainder(x, 0.5))
LOCAL_TEST_FUNCTOR1(remainderScalarAD, remainder(1.3, x))
TEST(ExpressionsMath, remainder_AD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest2_all(1.3, 0.5, res, 1.0, -double(n), 0.0, 0.0, 0.0, 0.0, remainderAD);
    mathTest_all(1.3, res, 1.0, 0.0, remainderADScalar);
    mathTest_all(0.5, res, -double(n), 0.0, remainderScalarAD);
}

LOCAL_TEST_FUNCTOR2(remainderExprExpr, remainder(2.3 * x1, 2.3 * x2))
LOCAL_TEST_FUNCTOR2(remainderADExpr, remainder(x1, 2.3 * x2))
LOCAL_TEST_FUNCTOR2(remainderExprAD, remainder(2.3 * x1, x2))
LOCAL_TEST_FUNCTOR1(remainderExprScalar, remainder(2.3 * x, 0.5))
LOCAL_TEST_FUNCTOR1(remainderScalarExpr, remainder(1.3, 2.3 * x))
TEST(ExpressionsMath, remainder_Expr)
{
    int n1, n2, n3;
    auto r1 = std::remquo(2.3 * 1.3, 2.3 * 0.5, &n1);
    auto r2 = std::remquo(1.3, 2.3 * 0.5, &n2);
    auto r3 = std::remquo(2.3 * 1.3, 0.5, &n3);
    mathTest2_all(1.3, 0.5, r1, 2.3, -2.3 * double(n1), 0.0, 0.0, 0.0, 0.0, remainderExprExpr);
    mathTest2_all(1.3, 0.5, r2, 1.0, -2.3 * double(n2), 0.0, 0.0, 0.0, 0.0, remainderADExpr);
    mathTest2_all(1.3, 0.5, r3, 2.3, -double(n3), 0.0, 0.0, 0.0, 0.0, remainderExprAD);
    mathTest_all(1.3, r3, 2.3, 0.0, remainderExprScalar);
    mathTest_all(0.5, r2, -2.3 * double(n2), 0.0, remainderScalarExpr);
}

// in Windows, remquo will always need to be qualified due to a
// global template definition in VC++'s library which will always
// conflict. It works with ADL in linux though...
int rmqn_ = 0;
LOCAL_TEST_FUNCTOR2(remquoAD, xad::remquo(x1, x2, &rmqn_))
LOCAL_TEST_FUNCTOR1(remquoADScalar, xad::remquo(x, 0.5, &rmqn_))
LOCAL_TEST_FUNCTOR1(remquoScalarAD, xad::remquo(1.3, x, &rmqn_))
TEST(ExpressionsMath, remquo_AD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest2_all_aad(1.3, 0.5, res, 1.0, -double(n), 0.0, 0.0, 0.0, 0.0, remquoAD);
    EXPECT_EQ(n, rmqn_);
    rmqn_ = 0;
    mathTest_all_aad(1.3, res, 1.0, 0.0, remquoADScalar);
    EXPECT_EQ(n, rmqn_);
    rmqn_ = 0;
    mathTest_all_aad(0.5, res, -double(n), 0.0, remquoScalarAD);
    EXPECT_EQ(n, rmqn_);
    rmqn_ = 0;
}

LOCAL_TEST_FUNCTOR2(remquoExprExpr, xad::remquo(2.3 * x1, 2.3 * x2, &rmqn_))
LOCAL_TEST_FUNCTOR2(remquoADExpr, xad::remquo(x1, 2.3 * x2, &rmqn_))
LOCAL_TEST_FUNCTOR2(remquoExprAD, xad::remquo(2.3 * x1, x2, &rmqn_))
LOCAL_TEST_FUNCTOR1(remquoExprScalar, xad::remquo(2.3 * x, 0.5, &rmqn_))
LOCAL_TEST_FUNCTOR1(remquoScalarExpr, xad::remquo(1.3, 2.3 * x, &rmqn_))
TEST(ExpressionsMath, remquo_Expr)
{
    int n1, n2, n3;
    auto r1 = std::remquo(2.3 * 1.3, 2.3 * 0.5, &n1);
    auto r2 = std::remquo(1.3, 2.3 * 0.5, &n2);
    auto r3 = std::remquo(2.3 * 1.3, 0.5, &n3);
    mathTest2_all_aad(1.3, 0.5, r1, 2.3, -2.3 * double(n1), 0.0, 0.0, 0.0, 0.0, remquoExprExpr);
    EXPECT_EQ(n1, rmqn_);
    rmqn_ = 0;
    mathTest2_all_aad(1.3, 0.5, r2, 1.0, -2.3 * double(n2), 0.0, 0.0, 0.0, 0.0, remquoADExpr);
    EXPECT_EQ(n2, rmqn_);
    rmqn_ = 0;
    mathTest2_all_aad(1.3, 0.5, r3, 2.3, -double(n3), 0.0, 0.0, 0.0, 0.0, remquoExprAD);
    EXPECT_EQ(n3, rmqn_);
    rmqn_ = 0;
    mathTest_all_aad(1.3, r3, 2.3, 0.0, remquoExprScalar);
    EXPECT_EQ(n3, rmqn_);
    rmqn_ = 0;
    mathTest_all_aad(0.5, r2, -2.3 * double(n2), 0.0, remquoScalarExpr);
    EXPECT_EQ(n2, rmqn_);
    rmqn_ = 0;
}

LOCAL_TEST_FUNCTOR2(atan2AD, xad::atan2(x1, x2))
LOCAL_TEST_FUNCTOR1(atan2ADScalar, xad::atan2(x, 0.5))
LOCAL_TEST_FUNCTOR1(atan2ScalarAD, xad::atan2(0.3, x))
TEST(ExpressionsMath, atan2_AD)
{
    mathTest2_all(0.3, 0.5, std::atan2(0.3, 0.5),
                  0.5 / (0.3 * 0.3 + 0.5 * 0.5),                                        // d1
                  -0.3 / (0.3 * 0.3 + 0.5 * 0.5),                                       // d2
                  -2. * 0.5 * 0.3 / (0.3 * 0.3 + 0.5 * 0.5) / (0.3 * 0.3 + 0.5 * 0.5),  // d11
                  -(.5 * .5 - .3 * .3) / (.3 * .3 + .5 * .5) / (.3 * .3 + .5 * .5),     // d12
                  -(.5 * .5 - .3 * .3) / (.3 * .3 + .5 * .5) / (.3 * .3 + .5 * .5),     // d21
                  2. * .3 * .5 / (.3 * .3 + .5 * .5) / (.3 * .3 + .5 * .5),             // d22
                  atan2AD);

    mathTest_all(0.3, std::atan2(0.3, 0.5), 0.5 / (0.3 * 0.3 + 0.5 * 0.5),
                 -2. * 0.5 * 0.3 / (0.3 * 0.3 + 0.5 * 0.5) / (0.3 * 0.3 + 0.5 * 0.5),
                 atan2ADScalar);

    mathTest_all(0.5, std::atan2(0.3, 0.5), -0.3 / (0.3 * 0.3 + 0.5 * 0.5),
                 2. * .3 * .5 / (.3 * .3 + .5 * .5) / (.3 * .3 + .5 * .5), atan2ScalarAD);
}

LOCAL_TEST_FUNCTOR2(atan2ExprExpr, xad::atan2(1.3 * x1, 1.3 * x2))
LOCAL_TEST_FUNCTOR1(atan2ExprScalar, xad::atan2(1.3 * x, 0.5))
LOCAL_TEST_FUNCTOR1(atan2ScalarExpr, xad::atan2(0.3, 1.3 * x))
TEST(ExpressionsMath, atan2_Expr)
{
    mathTest2_all(0.3, 0.5, std::atan2(1.3 * 0.3, 1.3 * 0.5),
                  1.3 * 0.5 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),   // d1
                  -1.3 * 0.3 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),  // d2
                  -2. * 1.3 * 1.3 * 1.3 * 1.3 * .5 * 0.3 /
                      (1.3 * 1.3 * 0.3 * 0.3 + 1.3 * 1.3 * 0.5 * 0.5) /
                      (1.3 * 1.3 * 0.3 * 0.3 + 1.3 * 1.3 * 0.5 * 0.5),               // d11
                  -(.5 * .5 - .3 * .3) / (.5 * .5 + .3 * .3) / (.5 * .5 + .3 * .3),  // d12
                  -(.5 * .5 - .3 * .3) / (.5 * .5 + .3 * .3) / (.5 * .5 + .3 * .3),  // d21
                  2. * .3 * .5 / (.5 * .5 + .3 * .3) / (.5 * .5 + .3 * .3),          // d22
                  atan2ExprExpr);
    mathTest_all(0.3, std::atan2(1.3 * 0.3, 0.5), 1.3 * 0.5 / (1.3 * 1.3 * 0.3 * 0.3 + 0.5 * 0.5),
                 -2. * 1.3 * 1.3 * 1.3 * .5 * .3 / (1.3 * 1.3 * .3 * .3 + .5 * .5) /
                     (1.3 * 1.3 * .3 * .3 + .5 * .5),
                 atan2ExprScalar);
    mathTest_all(0.5, std::atan2(0.3, 1.3 * 0.5), -1.3 * 0.3 / (0.3 * 0.3 + 1.3 * 1.3 * 0.5 * 0.5),
                 2. * 1.3 * 1.3 * 1.3 * .3 * .5 / (1.3 * 1.3 * .5 * .5 + .3 * .3) /
                     (1.3 * 1.3 * .5 * .5 + .3 * .3),
                 atan2ScalarExpr);
}

LOCAL_TEST_FUNCTOR2(hypotAD, xad::hypot(x1, x2))
LOCAL_TEST_FUNCTOR1(hypotADScalar, xad::hypot(x, 0.5))
LOCAL_TEST_FUNCTOR1(hypotScalarAD, xad::hypot(0.3, x))
TEST(ExpressionsMath, hypot_AD)
{
    mathTest2_all(0.3, 0.5, std::hypot(0.3, 0.5), 0.3 / std::hypot(0.3, 0.5),  // d1
                  0.5 / std::hypot(0.3, 0.5),                                  // d2
                  0.5 * 0.5 / std::pow(std::hypot(0.3, 0.5), 3),               // d11
                  -(0.5 * 0.3) / std::pow(std::hypot(0.3, 0.5), 3),            // d12
                  -(0.5 * 0.3) / std::pow(std::hypot(0.3, 0.5), 3),            // d21
                  0.3 * 0.3 / std::pow(std::hypot(0.3, 0.5), 3),               // d22
                  hypotAD);

    mathTest_all(0.3, std::hypot(0.3, 0.5), 0.3 / std::hypot(0.3, 0.5),
                 0.5 * 0.5 / std::pow(std::hypot(0.3, 0.5), 3), hypotADScalar);

    mathTest_all(0.5, std::hypot(0.3, 0.5), 0.5 / std::hypot(0.3, 0.5),
                 0.3 * 0.3 / std::pow(std::hypot(0.3, 0.5), 3), hypotScalarAD);
}

LOCAL_TEST_FUNCTOR2(hypotExprExpr, xad::hypot(1.3 * x1, 1.3 * x2))
LOCAL_TEST_FUNCTOR1(hypotExprScalar, xad::hypot(1.3 * x, 0.5))
LOCAL_TEST_FUNCTOR1(hypotScalarExpr, xad::hypot(0.3, 1.3 * x))
TEST(ExpressionsMath, hypot_Expr)
{
    mathTest2_all(
        0.3, 0.5, std::hypot(1.3 * 0.3, 1.3 * 0.5),
        1.3 * 1.3 * 0.3 / std::hypot(1.3 * 0.3, 1.3 * 0.5),                                 // d1
        1.3 * 1.3 * 0.5 / std::hypot(1.3 * 0.3, 1.3 * 0.5),                                 // d2
        1.3 * 1.3 * 1.3 * 0.5 * 1.3 * 0.5 / std::pow(std::hypot(1.3 * 0.3, 1.3 * 0.5), 3),  // d11
        -(1.3 * 1.3 * 1.3 * 0.5 * 1.3 * 0.3) /
            std::pow(std::hypot(1.3 * 0.3, 1.3 * 0.5), 3),  // d12
        -(1.3 * 1.3 * 1.3 * 0.5 * 1.3 * 0.3) /
            std::pow(std::hypot(1.3 * 0.3, 1.3 * 0.5), 3),                                  // d21
        1.3 * 1.3 * 1.3 * 0.3 * 1.3 * 0.3 / std::pow(std::hypot(1.3 * 0.3, 1.3 * 0.5), 3),  // d22
        hypotExprExpr);
    mathTest_all(0.3, std::hypot(1.3 * 0.3, 0.5), 1.3 * 1.3 * 0.3 / std::hypot(1.3 * 0.3, 0.5),
                 1.3 * 1.3 * 0.5 * 0.5 / std::pow(std::hypot(1.3 * 0.3, 0.5), 3), hypotExprScalar);
    mathTest_all(0.5, std::hypot(0.3, 1.3 * 0.5), 1.3 * 1.3 * 0.5 / std::hypot(0.3, 1.3 * 0.5),
                 1.3 * 1.3 * 0.3 * 0.3 / std::pow(std::hypot(0.3, 1.3 * 0.5), 3), hypotScalarExpr);
}

LOCAL_TEST_FUNCTOR1(cbrtAD, cbrt(x))
TEST(ExpressionsMath, cbrtAD)
{
    using xad::cbrt;
    mathTest_all(1.3, cbrt(1.3), 1.0 / 3.0 / std::pow(1.3, 2.0 / 3.0),
                 -2. / 9. / std::pow(1.3, 5. / 3.), cbrtAD);
}

LOCAL_TEST_FUNCTOR1(cbrtExpr, cbrt(2.1 * x))
TEST(ExpressionsMath, cbrtExpr)
{
    using xad::cbrt;
    mathTest_all(1.3, cbrt(2.1 * 1.3), 2.1 / 3.0 / std::pow(2.1 * 1.3, 2.0 / 3.0),
                 -2. * 2.1 * 2.1 / 9. / std::pow(2.1 * 1.3, 5. / 3.), cbrtExpr);
}

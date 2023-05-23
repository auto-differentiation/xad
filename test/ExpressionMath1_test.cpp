/*******************************************************************************

   Unit tests for math function derivatives (Part 1 - split due to long compile
   times).

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2023 Xcelerit Computing Ltd.

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

LOCAL_TEST_FUNCTOR1(degreesAD, degrees(x))
TEST(ExpressionsMath, degreesAD)
{
    mathTest_all_aad(3.141592653589793238462643, 180.0,
                     57.2957795130823208767981548141051703324054725, 0.0, degreesAD);
}

LOCAL_TEST_FUNCTOR1(degreesExpr, degrees(0.5 * x))
TEST(ExpressionsMath, degreesExpr)
{
    mathTest_all_aad(3.141592653589793238462643, 90.0,
                     0.5 * 57.2957795130823208767981548141051703324054725, 0.0, degreesExpr);
}

LOCAL_TEST_FUNCTOR1(radiansAD, radians(x));
TEST(ExpressionsMath, radiansAD)
{
    mathTest_all_aad(180.0, 3.141592653589793238462643,
                     0.0174532925199432957692369076848861271344287, 0.0, radiansAD);
}

LOCAL_TEST_FUNCTOR1(radiansExpr, radians(2.0 * x))
TEST(ExpressionsMath, radiansExpr)
{
    mathTest_all_aad(180.0, 2.0 * 3.141592653589793238462643,
                     2.0 * 0.0174532925199432957692369076848861271344287, 0.0, radiansExpr);
}

LOCAL_TEST_FUNCTOR1(cosAD, cos(x))
TEST(ExpressionsMath, cosAD)
{
    mathTest_all(1.0, std::cos(1.0), -std::sin(1.0), -std::cos(1.0), cosAD);
}

LOCAL_TEST_FUNCTOR1(cosExpr, cos(2.3 * x))
TEST(ExpressionsMath, cosExpr)
{
    mathTest_all(1.0, std::cos(2.3), -2.3 * std::sin(2.3), -2.3 * 2.3 * cos(2.3), cosExpr);
}

LOCAL_TEST_FUNCTOR1(sinAD, sin(x))
TEST(ExpressionsMath, sinAD)
{
    mathTest_all(1.0, std::sin(1.0), std::cos(1.0), -std::sin(1.0), sinAD);
}

LOCAL_TEST_FUNCTOR1(sinExpr, sin(2.3 * x))
TEST(ExpressionsMath, sinExpr)
{
    mathTest_all(1.0, std::sin(2.3), 2.3 * std::cos(2.3), -2.3 * 2.3 * std::sin(2.3), sinExpr);
}

LOCAL_TEST_FUNCTOR1(expAD, exp(x))
TEST(ExpressionsMath, expAD)
{
    mathTest_all(1.0, std::exp(1.0), std::exp(1.0), std::exp(1.0), expAD);
}

LOCAL_TEST_FUNCTOR1(expExpr, exp(2.3 * x))
TEST(ExpressionsMath, expExpr)
{
    mathTest_all(1.0, std::exp(2.3), 2.3 * std::exp(2.3), 2.3 * 2.3 * std::exp(2.3), expExpr);
}

LOCAL_TEST_FUNCTOR1(logAD, log(x))
TEST(ExpressionsMath, logAD)
{
    mathTest_all(1.3, std::log(1.3), 1.0 / 1.3, -1.0 / (1.3 * 1.3), logAD);
}

LOCAL_TEST_FUNCTOR1(logExpr, log(2.3 * x))
TEST(ExpressionsMath, logExpr) { mathTest_all(1.0, std::log(2.3), 1.0, -1.0, logExpr); }

LOCAL_TEST_FUNCTOR1(log10AD, log10(x))
TEST(ExpressionsMath, log10AD)
{
    mathTest_all(1.3, std::log10(1.3), 1.0 / std::log(10.0) / 1.3,
                 -1.0 / std::log(10.0) / (1.3 * 1.3), log10AD);
}

LOCAL_TEST_FUNCTOR1(log10Expr, log10(2.3 * x))
TEST(ExpressionsMath, log10Expr)
{
    mathTest_all(1.0, std::log10(2.3), 2.3 / std::log(10.0) / 2.3,
                 -2.3 * 2.3 / std::log(10.0) / (2.3 * 2.3), log10Expr);
}

LOCAL_TEST_FUNCTOR1(log2AD, log2(x))
TEST(ExpressionsMath, log2AD)
{
    using xad::log2;
    mathTest_all(1.3, log2(1.3), 1.0 / log(2.0) / 1.3, -1.0 / std::log(2.0) / (1.3 * 1.3), log2AD);
}

LOCAL_TEST_FUNCTOR1(log2Expr, log2(2.3 * x))
TEST(ExpressionsMath, log2Expr)
{
    using xad::log2;
    mathTest_all(1.3, log2(2.3 * 1.3), 2.3 / log(2.0) / 1.3 / 2.3,
                 -2.3 * 2.3 / std::log(2.0) / (1.3 * 2.3 * 1.3 * 2.3), log2Expr);
}

LOCAL_TEST_FUNCTOR1(asinAD, asin(x))
TEST(ExpressionsMath, asinAD)
{
    mathTest_all(0.3, std::asin(0.3), 1.0 / std::sqrt(1.0 - 0.3 * 0.3),
                 0.3 / std::pow(1 - 0.3 * 0.3, 3.0 / 2.0), asinAD);
}

LOCAL_TEST_FUNCTOR1(asinExpr, asin(2.3 * x))
TEST(ExpressionsMath, asinExpr)
{
    mathTest_all(0.1, std::asin(0.1 * 2.3), 2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 2.3 * 2.3 * 2.3 * 0.1 / std::pow(1 - 2.3 * 2.3 * 0.1 * 0.1, 3.0 / 2.0), asinExpr);
}

LOCAL_TEST_FUNCTOR1(acosAD, acos(x))
TEST(ExpressionsMath, acosAD)
{
    mathTest_all(0.3, std::acos(0.3), -1.0 / std::sqrt(1.0 - 0.3 * 0.3),
                 -0.3 / std::pow(1.0 - 0.3 * 0.3, 3.0 / 2.0), acosAD);
}

LOCAL_TEST_FUNCTOR1(acosExpr, acos(2.3 * x))
TEST(ExpressionsMath, acosExpr)
{
    mathTest_all(0.1, std::acos(0.1 * 2.3), -2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 -2.3 * 2.3 * 2.3 * 0.1 / std::pow(1 - 2.3 * 2.3 * 0.1 * 0.1, 3.0 / 2.0), acosExpr);
}

LOCAL_TEST_FUNCTOR1(atanAD, atan(x))
TEST(ExpressionsMath, atanAD)
{
    mathTest_all(0.3, std::atan(0.3), 1.0 / (1.0 + 0.3 * 0.3),
                 -2.0 * 0.3 / (0.3 * 0.3 + 1) / (0.3 * 0.3 + 1), atanAD);
}

LOCAL_TEST_FUNCTOR1(atanExpr, atan(2.3 * x))
TEST(ExpressionsMath, atanExpr)
{
    mathTest_all(0.1, std::atan(0.1 * 2.3), 2.3 / (1.0 + 2.3 * 2.3 * 0.1 * 0.1),
                 -2.0 * 2.3 * 2.3 * 2.3 * 0.1 / (2.3 * 2.3 * 0.1 * 0.1 + 1.0) /
                     (2.3 * 2.3 * 0.1 * 0.1 + 1.0),
                 atanExpr);
}

LOCAL_TEST_FUNCTOR1(sinhAD, sinh(x))
TEST(ExpressionsMath, sinhAD)
{
    mathTest_all(0.3, std::sinh(0.3), std::cosh(0.3), std::sinh(0.3), sinhAD);
}

LOCAL_TEST_FUNCTOR1(sinhExpr, sinh(2.3 * x))
TEST(ExpressionsMath, sinhExpr)
{
    mathTest_all(0.1, std::sinh(0.1 * 2.3), 2.3 * std::cosh(2.3 * 0.1),
                 2.3 * 2.3 * std::sinh(2.3 * 0.1), sinhExpr);
}

LOCAL_TEST_FUNCTOR1(coshAD, cosh(x))
TEST(ExpressionsMath, coshAD)
{
    mathTest_all(0.3, std::cosh(0.3), std::sinh(0.3), std::cosh(0.3), coshAD);
}

LOCAL_TEST_FUNCTOR1(coshExpr, cosh(2.3 * x))
TEST(ExpressionsMath, coshExpr)
{
    mathTest_all(0.3, std::cosh(2.3 * 0.3), 2.3 * std::sinh(2.3 * 0.3),
                 2.3 * 2.3 * std::cosh(2.3 * 0.3), coshExpr);
}

LOCAL_TEST_FUNCTOR1(expm1AD, expm1(x))
TEST(ExpressionsMath, expm1AD)
{
    using xad::expm1;
    mathTest_all(0.3, expm1(0.3), std::exp(0.3), std::exp(0.3), expm1AD);
}

LOCAL_TEST_FUNCTOR1(expm1Expr, expm1(2.3 * x))
TEST(ExpressionsMath, expm1Expr)
{
    using xad::expm1;
    mathTest_all(0.3, expm1(2.3 * 0.3), 2.3 * std::exp(2.3 * 0.3), 2.3 * 2.3 * std::exp(2.3 * 0.3),
                 expm1Expr);
}

LOCAL_TEST_FUNCTOR1(exp2AD, exp2(x))
TEST(ExpressionsMath, exp2AD)
{
    using xad::exp2;
    mathTest_all(0.3, exp2(0.3), std::log(2.0) * exp2(0.3),
                 std::log(2.0) * std::log(2.0) * exp2(0.3), exp2AD);
}

LOCAL_TEST_FUNCTOR1(exp2Expr, exp2(2.3 * x))
TEST(ExpressionsMath, exp2Expr)
{
    using xad::exp2;
    mathTest_all(0.3, exp2(2.3 * 0.3), 2.3 * std::log(2.0) * exp2(2.3 * 0.3),
                 2.3 * 2.3 * std::log(2.0) * std::log(2.0) * exp2(2.3 * 0.3), exp2Expr);
}

LOCAL_TEST_FUNCTOR1(log1pAD, log1p(x))
TEST(ExpressionsMath, log1pAD)
{
    using xad::log1p;
    mathTest_all(0.3, log1p(0.3), 1.0 / (1.0 + 0.3), -1.0 / (0.3 + 1.0) / (0.3 + 1.0), log1pAD);
}

LOCAL_TEST_FUNCTOR1(log1pExpr, log1p(2.3 * x))
TEST(ExpressionsMath, log1pExpr)
{
    using xad::log1p;
    mathTest_all(0.3, log1p(2.3 * 0.3), 2.3 / (1.0 + 2.3 * 0.3),
                 -2.3 * 2.3 / (2.3 * 0.3 + 1.0) / (2.3 * 0.3 + 1.0), log1pExpr);
}

LOCAL_TEST_FUNCTOR1(asinhAD, asinh(x))
TEST(ExpressionsMath, asinhAD)
{
    using xad::asinh;
    mathTest_all(0.3, asinh(0.3), 1.0 / std::sqrt(1.0 + 0.3 * 0.3),
                 -0.3 / std::pow(0.3 * .3 + 1.0, 3.0 / 2.0), asinhAD);
}

LOCAL_TEST_FUNCTOR1(asinhExpr, asinh(2.3 * x))
TEST(ExpressionsMath, asinhExpr)
{
    using xad::asinh;
    mathTest_all(0.3, asinh(2.3 * 0.3), 2.3 / std::sqrt(1.0 + 2.3 * 2.3 * 0.3 * 0.3),
                 -2.3 * 2.3 * 2.3 * 0.3 / std::pow(2.3 * 2.3 * 0.3 * 0.3 + 1.0, 3. / 2.),
                 asinhExpr);
}

LOCAL_TEST_FUNCTOR1(acoshAD, acosh(x))
TEST(ExpressionsMath, acoshAD)
{
    using xad::acosh;
    mathTest_all(1.3, acosh(1.3), 1.0 / std::sqrt(1.3 * 1.3 - 1.0),
                 -1.3 / std::pow(1.3 * 1.3 - 1., 3. / 2.), acoshAD);
}

LOCAL_TEST_FUNCTOR1(acoshExpr, acosh(2.3 * x))
TEST(ExpressionsMath, acoshExpr)
{
    using xad::acosh;
    mathTest_all(1.3, acosh(2.3 * 1.3), 2.3 / std::sqrt(2.3 * 2.3 * 1.3 * 1.3 - 1.0),
                 -2.3 * 2.3 * 2.3 * 1.3 / std::pow(2.3 * 2.3 * 1.3 * 1.3 - 1., 3. / 2.), acoshExpr);
}

LOCAL_TEST_FUNCTOR1(atanhAD, atanh(x))
TEST(ExpressionsMath, atanhAD)
{
    using xad::atanh;
    mathTest_all(0.3, atanh(0.3), 1.0 / (1.0 - 0.3 * 0.3),
                 2. * 0.3 / (0.3 * 0.3 - 1.) / (0.3 * 0.3 - 1.), atanhAD);
}

LOCAL_TEST_FUNCTOR1(atanhExpr, atanh(2.3 * x))
TEST(ExpressionsMath, atanhExpr)
{
    using xad::atanh;
    mathTest_all(
        0.3, atanh(2.3 * 0.3), 2.3 / (1.0 - 2.3 * 2.3 * 0.3 * 0.3),
        2. * 2.3 * 2.3 * 2.3 * 0.3 / (2.3 * 2.3 * 0.3 * 0.3 - 1.) / (2.3 * 2.3 * 0.3 * 0.3 - 1.),
        atanhExpr);
}

LOCAL_TEST_FUNCTOR1(erfAD, erf(x))
TEST(ExpressionsMath, erfAD)
{
    mathTest_all(0.3, std::erf(0.3), 2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3),
                 -4. * 0.3 * exp(-0.3 * 0.3) / std::sqrt(M_PI), erfAD);
}

LOCAL_TEST_FUNCTOR1(erfExpr, erf(2.3 * x))
TEST(ExpressionsMath, erfExpr)
{
    mathTest_all(
        0.3, std::erf(2.3 * 0.3), 2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3),
        -4. * 2.3 * 2.3 * 2.3 * 0.3 * std::exp(-2.3 * 2.3 * 0.3 * 0.3) / std::sqrt(M_PI), erfExpr);
}

LOCAL_TEST_FUNCTOR1(erfcAD, erfc(x))
TEST(ExpressionsMath, erfcAD)
{
    mathTest_all(0.3, std::erfc(0.3), -2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3),
                 4. * 0.3 * std::exp(-0.3 * 0.3) / std::sqrt(M_PI), erfcAD);
}

LOCAL_TEST_FUNCTOR1(erfcExpr, erfc(2.3 * x))
TEST(ExpressionsMath, erfcExpr)
{
    mathTest_all(
        0.3, std::erfc(2.3 * 0.3), -2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3),
        4. * 2.3 * 2.3 * 2.3 * 0.3 * std::exp(-2.3 * 2.3 * 0.3 * 0.3) / std::sqrt(M_PI), erfcExpr);
}

LOCAL_TEST_FUNCTOR1(tanhAD, tanh(x))
TEST(ExpressionsMath, tanhAD)
{
    mathTest_all(0.3, std::tanh(0.3), 1.0 - std::tanh(0.3) * std::tanh(0.3),
                 -2. / std::cosh(0.3) / std::cosh(0.3) * std::tanh(.3), tanhAD);
}

LOCAL_TEST_FUNCTOR1(tanhExpr, tanh(2.3 * x))
TEST(ExpressionsMath, tanhExpr)
{
    mathTest_all(
        0.3, std::tanh(2.3 * 0.3), 2.3 * (1.0 - std::tanh(2.3 * 0.3) * std::tanh(2.3 * 0.3)),
        -2. * 2.3 * 2.3 / std::cosh(2.3 * 0.3) / std::cosh(2.3 * 0.3) * std::tanh(2.3 * 0.3),
        tanhExpr);
}

LOCAL_TEST_FUNCTOR1(tanAD, tan(x))
TEST(ExpressionsMath, tanAD)
{
    mathTest_all(0.3, std::tan(0.3), 1.0 / std::cos(0.3) / std::cos(0.3),
                 2. / std::cos(0.3) / std::cos(0.3) * std::tan(0.3), tanAD);
}

LOCAL_TEST_FUNCTOR1(tanExpr, tan(2.3 * x))
TEST(ExpressionsMath, tanExpr)
{
    mathTest_all(0.3, std::tan(2.3 * 0.3), 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3),
                 2. * 2.3 * 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3) * std::tan(2.3 * 0.3),
                 tanExpr);
}

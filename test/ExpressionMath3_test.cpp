/*******************************************************************************

   Unit tests for math function derivatives (Part 3 - split due to long compile
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

LOCAL_TEST_FUNCTOR2(nextafterADAD, nextafter(x1, x2))
TEST(ExpressionsMath, nextafterADAD)
{
    mathTest2_all(0.1, 0.2, std::nextafter(0.1, 0.2), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, nextafterADAD);
    mathTest2_all(-0.1, -0.2, std::nextafter(-0.1, -0.2), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterADAD);
}

LOCAL_TEST_FUNCTOR2(nextafterADExpr, nextafter(x1, 2.3 * x2))
TEST(ExpressionsMath, nextafterADExpr)
{
    mathTest2_all(0.1, 0.2, std::nextafter(0.1, 2.3 * 0.2), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterADExpr);
    mathTest2_all(-0.1, -0.2, std::nextafter(-0.1, 2.3 * -0.2), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterADExpr);
}

LOCAL_TEST_FUNCTOR2(nextafterExprAD, nextafter(2.3 * x1, x2))
TEST(ExpressionsMath, nextafterExprAD)
{
    mathTest2_all(0.1, 0.2, std::nextafter(2.3 * 0.1, 0.2), 2.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterExprAD);
    mathTest2_all(-0.1, -0.2, std::nextafter(2.3 * -0.1, -0.2), 2.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterExprAD);
}

LOCAL_TEST_FUNCTOR2(nextafterExprExpr, nextafter(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMath, nextafterExprExpr)
{
    mathTest2_all(0.1, 0.2, std::nextafter(2.3 * 0.1, 2.3 * 0.2), 2.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterExprExpr);
    mathTest2_all(-0.1, -0.2, std::nextafter(2.3 * -0.1, 2.3 * -0.2), 2.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                  nextafterExprExpr);
}

LOCAL_TEST_FUNCTOR1(nextafterScalarAD, nextafter(0.1, x))
TEST(ExpressionsMath, nextafterScalarAD)
{
    mathTest_all(0.2, std::nextafter(0.1, 0.2), 0.0, 0.0, nextafterScalarAD);
}

LOCAL_TEST_FUNCTOR1(nextafterADScalar, nextafter(x, 0.2))
TEST(ExpressionsMath, nextafterADScalar)
{
    mathTest_all(0.1, std::nextafter(0.1, 0.2), 1.0, 0.0, nextafterADScalar);
}

LOCAL_TEST_FUNCTOR1(nextafterScalarExpr, nextafter(0.1, x * 2.3))
TEST(ExpressionsMath, nextafterScalarExpr)
{
    mathTest_all(0.2, std::nextafter(0.1, 0.2 * 2.3), 0.0, 0.0, nextafterScalarExpr);
}

LOCAL_TEST_FUNCTOR1(nextafterExprScalar, nextafter(x * 2.3, 0.2))
TEST(ExpressionsMath, nextafterExprScalar)
{
    mathTest_all(0.1, std::nextafter(0.1 * 2.3, 0.2), 2.3, 0.0, nextafterExprScalar);
}

LOCAL_TEST_FUNCTOR1(scalbnAD, scalbn(x, 2))
TEST(ExpressionsMath, scalbnAD)
{
    mathTest_all(0.1, std::scalbn(0.1, 2), std::pow(double(FLT_RADIX), 2.0), 0.0, scalbnAD);
}

LOCAL_TEST_FUNCTOR1(scalbnExpr, scalbn(x * 2.3, 2))
TEST(ExpressionsMath, scalbnExpr)
{
    mathTest_all(0.1, std::scalbn(0.1 * 2.3, 2), std::pow(double(FLT_RADIX), 2.0) * 2.3, 0.0,
                 scalbnExpr);
}

LOCAL_TEST_FUNCTOR2(maxADAD, max(x1, x2))
LOCAL_TEST_FUNCTOR2(fmaxADAD, fmax(x1, x2))
TEST(ExpressionsMath, maxADAD)
{
    mathTest2_all(0.3, 0.7, 0.7, 0, 1.0, 0.0, 0.0, 0.0, 0.0, maxADAD);
    mathTest2_all(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, maxADAD);
    mathTest2_all(1.7, 1.7, 1.7, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, maxADAD);
    mathTest2_all(0.3, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, fmaxADAD);
    mathTest2_all(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, fmaxADAD);
    mathTest2_all(1.7, 1.7, 1.7, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, fmaxADAD);
}

LOCAL_TEST_FUNCTOR2(maxADExpr, max(x1, 2.3 * x2))
TEST(ExpressionsMath, maxADExpr)
{
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxADExpr);
    mathTest2_all(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, maxADExpr);
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxADExpr);
    mathTest2_all(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, maxADExpr);
}

LOCAL_TEST_FUNCTOR2(maxExprAD, max(2.3 * x1, x2))
TEST(ExpressionsMath, maxExprAD)
{
    mathTest2_all(0.3, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, maxExprAD);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprAD);
    mathTest2_all(0.3, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, maxExprAD);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprAD);
}

LOCAL_TEST_FUNCTOR2(maxExprExpr, max(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMath, maxExprExpr)
{
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr);
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr);
}

LOCAL_TEST_FUNCTOR2(maxExprExpr2, max(2.3 * x1, 2.3 * x2 + 0.0))
TEST(ExpressionsMath, maxExprExpr2)
{
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr2);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr2);
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr2);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr2);
}

LOCAL_TEST_FUNCTOR2(maxExprExpr3, max(2.3 * x1 + 0.0, 2.3 * x2))
TEST(ExpressionsMath, maxExprExpr3)
{
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr3);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr3);
    mathTest2_all(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, maxExprExpr3);
    mathTest2_all(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, maxExprExpr3);
}

LOCAL_TEST_FUNCTOR1(maxScalarAD, max(0.7, x))
TEST(ExpressionsMath, maxScalarAD)
{
    mathTest_all(1.1, 1.1, 1.0, 0.0, maxScalarAD);
    mathTest_all(0.6, 0.7, 0.0, 0.0, maxScalarAD);
}

LOCAL_TEST_FUNCTOR1(maxADScalar, max(x, 0.7))
TEST(ExpressionsMath, maxADScalar)
{
    mathTest_all(1.1, 1.1, 1.0, 0.0, maxADScalar);
    mathTest_all(0.6, 0.7, 0.0, 0.0, maxADScalar);
}

LOCAL_TEST_FUNCTOR1(maxScalarExpr, max(0.7, 2.0 * x))
TEST(ExpressionsMath, maxScalarExpr)
{
    mathTest_all(1.1, 2.0 * 1.1, 2.0, 0.0, maxScalarExpr);
    mathTest_all(0.3, 0.7, 0.0, 0.0, maxScalarExpr);
}

LOCAL_TEST_FUNCTOR1(maxExprScalar, max(2.0 * x, 0.7))
TEST(ExpressionsMath, maxExprScalar)
{
    mathTest_all(1.1, 2.0 * 1.1, 2.0, 0.0, maxExprScalar);
    mathTest_all(0.3, 0.7, 0.0, 0.0, maxExprScalar);
}

LOCAL_TEST_FUNCTOR2(smaxADAD, smooth_max(x1, x2))
TEST(ExpressionsMath, smaxADAD)
{
    mathTest2_all_aad(0.3, 0.7, 0.7, 0, 1.0, 0.0, 0.0, 0.0, 0.0, smaxADAD);
    mathTest2_all_aad(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, smaxADAD);
    double c = 0.001;
    mathTest2_all_aad(1.7, 1.7, 1.7, 0.5, 0.5, 4. / 2. / c, -2. / c, -2. / c, 4. / 2. / c,
                      smaxADAD);
}

LOCAL_TEST_FUNCTOR2(smaxADExpr, smooth_max(x1, 2.0 * x2))
TEST(ExpressionsMath, smaxADExpr)
{
    mathTest2_all_aad(0.3, 0.7, 2.0 * 0.7, 0, 2.0, 0.0, 0.0, 0.0, 0.0, smaxADExpr);
    mathTest2_all_aad(1.7, -0.7, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, smaxADExpr);
    double c = 0.001;
    mathTest2_all_aad(2.0, 1.0, 2.0, 0.5, 1.0, 4. / 2. / c, 2. * -2. / c, 2. * -2. / c,
                      2. * 2. * 4. / 2. / c, smaxADExpr);
}

LOCAL_TEST_FUNCTOR2(smaxExprAD, smooth_max(2.0 * x1, x2))
TEST(ExpressionsMath, smaxExprAD)
{
    mathTest2_all_aad(0.3, 0.7, 0.7, 0, 1.0, 0.0, 0.0, 0.0, 0.0, smaxExprAD);
    mathTest2_all_aad(1.7, -0.7, 2.0 * 1.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, smaxExprAD);
    double c = 0.001;
    mathTest2_all_aad(1.0, 2.0, 2.0, 1.0, 0.5, 2. * 2. * 4. / 2. / c, 2. * -2. / c, 2. * -2. / c,
                      4. / 2. / c, smaxExprAD);
}

LOCAL_TEST_FUNCTOR2(smaxExprExpr, smooth_max(2.0 * x1, 2.0 * x2))
TEST(ExpressionsMath, smaxExprExpr)
{
    mathTest2_all_aad(0.3, 0.7, 2.0 * 0.7, 0, 2.0, 0.0, 0.0, 0.0, 0.0, smaxExprExpr);
    mathTest2_all_aad(1.7, -0.7, 2.0 * 1.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, smaxExprExpr);
    double c = 0.001;
    mathTest2_all_aad(1.0, 1.0, 2.0, 1.0, 1.0, 2. * 2. * 4. / 2. / c, 2. * 2. * -2. / c,
                      2. * 2. * -2. / c, 2. * 2. * 4. / 2. / c, smaxExprExpr);
}

LOCAL_TEST_FUNCTOR1(smaxScalarAD, smooth_max(0.7, x))
TEST(ExpressionsMath, smaxScalarAD)
{
    mathTest_all_aad(1.1, 1.1, 1.0, 0.0, smaxScalarAD);
    mathTest_all_aad(0.6, 0.7, 0.0, 0.0, smaxScalarAD);
}

LOCAL_TEST_FUNCTOR1(smaxScalarExpr, smooth_max(2.0, 2.0 * x))
TEST(ExpressionsMath, smaxScalarExpr)
{
    double c = 0.001;
    mathTest_all_aad(1.1, 2.2, 2.0, 0.0, smaxScalarExpr);
    mathTest_all_aad(1.0, 2.0, 1.0, 2. * 2. * 4. / 2. / c, smaxScalarExpr);
    mathTest_all_aad(0.3, 2.0, 0.0, 0.0, smaxScalarExpr);
}

LOCAL_TEST_FUNCTOR1(smaxADScalar, smooth_max(x, 0.7))
TEST(ExpressionsMath, smaxADScalar)
{
    mathTest_all_aad(1.1, 1.1, 1.0, 0.0, smaxADScalar);
    mathTest_all_aad(0.6, 0.7, 0.0, 0.0, smaxADScalar);
}

LOCAL_TEST_FUNCTOR1(smaxExprScalar, smooth_max(2.0 * x, 2.0))
TEST(ExpressionsMath, smaxExprScalar)
{
    double c = 0.001;
    mathTest_all_aad(1.1, 2.2, 2.0, 0.0, smaxExprScalar);
    mathTest_all_aad(1.0, 2.0, 1.0, 2. * 2. * 4. / 2. / c, smaxExprScalar);
    mathTest_all_aad(0.3, 2.0, 0.0, 0.0, smaxExprScalar);
}

LOCAL_TEST_FUNCTOR2(minADAD, min(x1, x2))
TEST(ExpressionsMath, minADAD)
{
    mathTest2_all(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, minADAD);
    mathTest2_all(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minADAD);
    mathTest2_all(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, minADAD);
    mathTest2_all(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minADAD);

    xad::Tape<double> s;
    xad::AD x1 = 1.0, x2 = 2.3;
    static_assert((std::is_same<decltype(min(x1, x2)),
                                xad::BinaryExpr<double, xad::min_op<double>, xad::ADVar<double>,
                                                xad::ADVar<double> > >::value),
                  "AD type not wrapped");
}

LOCAL_TEST_FUNCTOR2(minADExpr, min(x1, 2.3 * x2))
TEST(ExpressionsMath, minADExpr)
{
    mathTest2_all(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, minADExpr);
    mathTest2_all(1.7, -0.7, -0.7 * 2.3, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, minADExpr);
    mathTest2_all(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, minADExpr);
    mathTest2_all(1.7, -0.7, -0.7 * 2.3, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, minADExpr);
}

LOCAL_TEST_FUNCTOR2(minExprAD, min(2.3 * x1, x2))
TEST(ExpressionsMath, minExprAD)
{
    mathTest2_all(0.5, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minExprAD);
    mathTest2_all(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minExprAD);
    mathTest2_all(0.5, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minExprAD);
    mathTest2_all(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, minExprAD);
}

LOCAL_TEST_FUNCTOR2(minExprExpr, min(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMath, minExprExpr)
{
    mathTest2_all(0.3, 0.7, 2.3 * 0.3, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, minExprExpr);
    mathTest2_all(1.7, -0.7, 2.3 * -0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, minExprExpr);
    mathTest2_all(0.3, 0.7, 2.3 * 0.3, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, minExprExpr);
    mathTest2_all(1.7, -0.7, 2.3 * -0.7, 0.0, 2.3, 0.0, 0.0, 0.0, 0.0, minExprExpr);
}

LOCAL_TEST_FUNCTOR1(minScalarAD, min(0.7, x))
TEST(ExpressionsMath, minScalarAD)
{
    mathTest_all(1.1, 0.7, 0.0, 0.0, minScalarAD);
    mathTest_all(0.6, 0.6, 1.0, 0.0, minScalarAD);
}

LOCAL_TEST_FUNCTOR1(minADScalar, min(x, 0.7))
TEST(ExpressionsMath, minADScalar)
{
    mathTest_all(1.1, 0.7, 0.0, 0.0, minADScalar);
    mathTest_all(0.6, 0.6, 1.0, 0.0, minADScalar);
}

LOCAL_TEST_FUNCTOR1(minScalarExpr, min(0.7, 2.0 * x))
TEST(ExpressionsMath, minScalarExpr)
{
    mathTest_all(1.1, 0.7, 0.0, 0.0, minScalarExpr);
    mathTest_all(0.3, 2.0 * 0.3, 2.0, 0.0, minScalarExpr);
}

LOCAL_TEST_FUNCTOR1(minExprScalar, min(2.0 * x, 0.7))
TEST(ExpressionsMath, minExprScalar)
{
    mathTest_all(1.1, 0.7, 0.0, 0.0, minExprScalar);
    mathTest_all(0.3, 2.0 * 0.3, 2.0, 0.0, minExprScalar);
}

LOCAL_TEST_FUNCTOR2(sminADAD, smooth_min(x1, x2))
TEST(ExpressionsMath, sminADAD)
{
    mathTest2_all_aad(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sminADAD);
    mathTest2_all_aad(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, sminADAD);
    double c = 0.001;
    mathTest2_all_aad(1.7, 1.7, 1.7, 0.5, 0.5, -2. / c, 2. / c, 2. / c, -2. / c, sminADAD);
}

LOCAL_TEST_FUNCTOR2(sminADExpr, smooth_min(x1, 2.0 * x2))
TEST(ExpressionsMath, sminADExpr)
{
    mathTest2_all_aad(0.3, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, sminADExpr);
    mathTest2_all_aad(1.7, -0.7, 2.0 * -0.7, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, sminADExpr);
    double c = 0.001;
    mathTest2_all_aad(2.0, 1.0, 2.0, 0.5, 1.0, -2. / c, 2. * 2. / c, 2. * 2. / c, 2. * 2. * -2. / c,
                      sminADExpr);
}

LOCAL_TEST_FUNCTOR2(sminExprAD, smooth_min(2.0 * x1, x2))
TEST(ExpressionsMath, sminExprAD)
{
    mathTest2_all_aad(0.3, 0.7, 2.0 * 0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, sminExprAD);
    mathTest2_all_aad(1.7, -0.7, -0.7, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, sminExprAD);
    double c = 0.001;
    mathTest2_all_aad(1.0, 2.0, 2.0, 1.0, 0.5, 2. * 2. * -2. / c, 2. * 2. / c, 2. * 2. / c, -2. / c,
                      sminExprAD);
}

LOCAL_TEST_FUNCTOR2(sminExprExpr, smooth_min(2.0 * x1, 2.0 * x2))
TEST(ExpressionsMath, sminExprExpr)
{
    mathTest2_all_aad(0.3, 0.7, 2.0 * 0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, sminExprExpr);
    mathTest2_all_aad(1.7, -0.7, 2.0 * -0.7, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, sminExprExpr);
    double c = 0.001;
    mathTest2_all_aad(1.0, 1.0, 2.0, 1.0, 1.0, 2. * 2. * -2. / c, 2. * -2. * -2. / c,
                      2. * -2. * -2. / c, 2. * 2. * -2. / c, sminExprExpr);
}

LOCAL_TEST_FUNCTOR1(sminScalarAD, smooth_min(0.7, x))
TEST(ExpressionsMath, sminScalarAD)
{
    mathTest_all_aad(1.1, 0.7, 0.0, 0.0, sminScalarAD);
    mathTest_all_aad(0.6, 0.6, 1.0, 0.0, sminScalarAD);
}

LOCAL_TEST_FUNCTOR1(sminScalarExpr, smooth_min(2.0, 2.0 * x))
TEST(ExpressionsMath, sminScalarExpr)
{
    mathTest_all_aad(1.1, 2.0, 0.0, 0.0, sminScalarExpr);
    mathTest_all_aad(0.9, 1.8, 2.0, 0.0, sminScalarExpr);
    double c = 0.001;
    mathTest_all_aad(1.0, 2.0, 1.0, -2. * 2. * 2. / c, sminScalarExpr);
}

LOCAL_TEST_FUNCTOR1(sminADScalar, smooth_min(x, 0.7))
TEST(ExpressionsMath, sminADScalar)
{
    mathTest_all_aad(1.1, 0.7, 0.0, 0.0, sminADScalar);
    mathTest_all_aad(0.6, 0.6, 1.0, 0.0, sminADScalar);
}

LOCAL_TEST_FUNCTOR1(sminExprScalar, smooth_min(2.0 * x, 2.0))
TEST(ExpressionsMath, sminExprScalar)
{
    mathTest_all_aad(1.1, 2.0, 0.0, 0.0, sminExprScalar);
    mathTest_all_aad(0.9, 1.8, 2.0, 0.0, sminExprScalar);
    double c = 0.001;
    mathTest_all_aad(1.0, 2.0, 1.0, -2. * 2. * 2. / c, sminExprScalar);
}

// make sure that max/min in std namespace are still working for integer and
// other arguments
TEST(ExpressionsMath, maxMinExplicitRealAD)
{
    xad::AD x = 10.0, y = 8.0;
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(x, y)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(x, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(x, y * 1.0)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(x, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(x * 1.0, y)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(x * 1.0, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(x * 1.0, y * 1.0)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(x * 1.0, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(1.0, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(1.0, y)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(1.0, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(1.0, y * 1.0)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(y, 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(y, 1.0)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::AD>(y * 1.0, 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::AD>(y * 1.0, 1.0)), 1.0, 1e-9);
}

TEST(ExpressionsMath, maxMinExplicitRealFAD)
{
    xad::FAD x = 10.0, y = 8.0;
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(x, y)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(x, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(x, y * 1.0)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(x, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(x * 1.0, y)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(x * 1.0, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(x * 1.0, y * 1.0)), 10.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(x * 1.0, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(1.0, y)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(1.0, y)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(1.0, y * 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(1.0, y * 1.0)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(y, 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(y, 1.0)), 1.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::max<xad::FAD>(y * 1.0, 1.0)), 8.0, 1e-9);
    EXPECT_NEAR(xad::value(xad::min<xad::FAD>(y * 1.0, 1.0)), 1.0, 1e-9);
}

TEST(ExpressionsMath, maxMinForIntegers)
{
    int x = 10, y = 8;
    EXPECT_EQ(xad::max(x, y), x);
    EXPECT_EQ(xad::max(y, x), x);
    EXPECT_EQ(xad::min(x, y), y);
    EXPECT_EQ(xad::min(y, x), y);
}

TEST(ExpressionsMath, maxMinForIntegersExplicit)
{
    int x = 10;
    long long y = 8;
    EXPECT_EQ(xad::max<long long>(x, y), x);
    EXPECT_EQ(xad::max<long long>(y, x), x);
    EXPECT_EQ(xad::min<long long>(x, y), y);
    EXPECT_EQ(xad::min<long long>(y, x), y);
}

// temporarily disable double->int conversion warning, as we're testing just that below
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#if (__GNUC__ < 5) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wconversion"
#else
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-conversion"
#endif

TEST(ExpressionsMath, maxMinForIntegersDoubleExplicit)
{
    int x = 10;
    double y = 8.;
    EXPECT_EQ(xad::max<long long>(x, y), x);
    EXPECT_EQ(xad::max<long long>(y, x), x);
    EXPECT_EQ(xad::min<long long>(x, y), static_cast<long long>(y));
    EXPECT_EQ(xad::min<long long>(y, x), static_cast<long long>(y));
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

// ldexp(x, a) = x1 * 2^a
LOCAL_TEST_FUNCTOR1(ldexpAD, ldexp(x, 3))
TEST(ExpressionsMath, ldexpAD) { mathTest_all(1.1, 1.1 * 8, 8.0, 0.0, ldexpAD); }

LOCAL_TEST_FUNCTOR1(ldexpExpr, ldexp(2.0 * x, 3))
TEST(ExpressionsMath, ldexpExp) { mathTest_all(1.1, 2.2 * 8, 16.0, 0.0, ldexpExpr); }

struct testFunctor_frexpAD
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(x, &exponent);
    }
} frexpAD;
int testFunctor_frexpAD::exponent = 0;

TEST(ExpressionsMath, frexpAD)
{
    mathTest_all(1024.0, 0.5, 1.0 / (1 << 11), 0.0, frexpAD);
    EXPECT_EQ(testFunctor_frexpAD::exponent, 11);
}

struct testFunctor_frexpExpr
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(1.0 * x, &exponent);
    }
} frexpExpr;
int testFunctor_frexpExpr::exponent = 0;

TEST(ExpressionsMath, frexpExpr)
{
    mathTest_all(1024.0, 0.5, 1.0 / (1 << 11), 0.0, frexpExpr);
    EXPECT_EQ(testFunctor_frexpExpr::exponent, 11);
}

struct testFunctor_modfADScalar
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        return modf(x, &ipart);
    }
} modfADScalar;
double testFunctor_modfADScalar::ipart = 0.0;

TEST(ExpressionsMath, modfADScalar)
{
    mathTest_all(1.2, 0.2, 1.0, 0.0, modfADScalar);
    EXPECT_NEAR(testFunctor_modfADScalar::ipart, 1.0, 1e-9);
    mathTest_all(790.185598, 790.185598 - 790.0, 1.0, 0.0, modfADScalar);
    EXPECT_NEAR(testFunctor_modfADScalar::ipart, 790.0, 1e-9);
    mathTest_all(-790.185598, -790.185598 + 790.0, 1.0, 0.0, modfADScalar);
    EXPECT_NEAR(testFunctor_modfADScalar::ipart, -790.0, 1e-9);
}

struct testFunctor_modfADAD
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        T ipt;
        T ret = modf(x, &ipt);
        ipart = xad::value(xad::value(ipt));
        return ret;
    }
} modfADAD;
double testFunctor_modfADAD::ipart = 0.0;

TEST(ExpressionsMath, modfADAD)
{
    mathTest_all(1.2, 0.2, 1.0, 0.0, modfADAD);
    EXPECT_NEAR(testFunctor_modfADAD::ipart, 1.0, 1e-9);
}

struct testFunctor_modfExprScalar
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        return modf(x * 1.0, &ipart);
    }
} modfExprScalar;
double testFunctor_modfExprScalar::ipart = 0.0;

TEST(ExpressionsMath, modfExprScalar)
{
    mathTest_all(1.2, 0.2, 1.0, 0.0, modfExprScalar);
    EXPECT_NEAR(testFunctor_modfExprScalar::ipart, 1.0, 1e-9);
}

struct testFunctor_copysignScalar1
{
    explicit testFunctor_copysignScalar1(double op1) : op1_(op1) {}
    double op1_ = 0.0;
    template <class T>
    double operator()(const T& x) const
    {
        return copysign(op1_, x);
    }
};

TEST(ExpressionsMath, copysignScalarAD)
{
    mathTest_all(1.2, 42.2, 0.0, 0.0, testFunctor_copysignScalar1(42.2));
    mathTest_all(-1.2, -42.2, 0.0, 0.0, testFunctor_copysignScalar1(42.2));
}

struct testFunctor_copysignScalar2
{
    explicit testFunctor_copysignScalar2(double op2) : op2_(op2) {}
    double op2_ = 0.0;
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, op2_);
    }
};

TEST(ExpressionsMath, copysignADScalar)
{
    mathTest_all(1.2, 1.2, 1.0, 0.0, testFunctor_copysignScalar2(5.9));
    mathTest_all(1.2, 1.2, 1.0, 0.0, testFunctor_copysignScalar2(0.0));
    mathTest_all(1.2, -1.2, -1.0, 0.0, testFunctor_copysignScalar2(-5.9));
    mathTest_all(1.2, -1.2, -1.0, 0.0, testFunctor_copysignScalar2(-0.0000001));
}

struct testFunctor_copysignAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, x);
    }
} copysignAD;

TEST(ExpressionsMath, copysignADAD)
{
    mathTest_all(1.2, 1.2, 1.0, 0.0, copysignAD);
    mathTest_all(-1.2, -1.2, 1.0, 0.0, copysignAD);
}

struct testFunctor_copysignADExpr
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, -x);
    }
} copysignADExpr;

TEST(ExpressionsMath, copysignADExpr) { mathTest_all(1.2, -1.2, -1.0, 0.0, copysignADExpr); }

struct testFunctor_copysignExprAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, x);
    }
} copysignExprAD;

TEST(ExpressionsMath, copysignExprAD) { mathTest_all(1.2, 1.2, 1.0, 0.0, copysignExprAD); }

struct testFunctor_copysignExprExpr
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, -x);
    }
} copysignExprExpr;

TEST(ExpressionsMath, copysignExprExpr) { mathTest_all(1.2, -1.2, -1.0, 0.0, copysignExprExpr); }

TEST(ExpressionsMath, copysignADQuantLibReal) {
    xad::AD x(1.2);
    xad::AD y(-0.5);

    auto result = copysign(x, y);

    EXPECT_EQ(xad::value(result), -1.2);
}
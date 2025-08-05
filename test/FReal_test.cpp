/*******************************************************************************

   Unit tests for Vector Forward Mode

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

#include <XAD/Literals.hpp>
#include <XAD/Vec.hpp>
#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

///////// FReal<Scalar, N> /////////

TEST(FRealTest, CanConstructFRealVec)
{
    using AD = xad::FReal<double, 2>;
    AD a;

    EXPECT_THAT(value(a), DoubleEq(0.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(0.0), DoubleEq(0.0)));
}

TEST(FRealTest, CanAssignToFRealVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(1.0, {1.0, 0.0});

    EXPECT_THAT(value(a), DoubleEq(1.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(1.0), DoubleEq(0.0)));
}

TEST(FRealTest, CanCopyConstructVec)
{
    using AD = xad::FReal<double, 2>;
    AD x0 = 2.0;
    derivative(x0) = {1.0, 0.0};

    AD x1(x0);

    EXPECT_THAT(value(x1), value(x0));
    EXPECT_THAT(derivative(x0), ElementsAre(DoubleEq(1.0), DoubleEq(0.0)));
    EXPECT_THAT(derivative(x1), ElementsAre(DoubleEq(1.0), DoubleEq(0.0)));
}

TEST(FRealTest, CanAssignValueToExistingObjectVec)
{
    using AD = xad::FReal<double, 2>;
    AD x0 = 2.0;
    derivative(x0) = {1.0, 0.0};
    x0 = 4.0;
    EXPECT_THAT(value(x0), DoubleEq(4.0));
    EXPECT_THAT(derivative(x0), ElementsAre(DoubleEq(0.0), DoubleEq(0.0)));
}

TEST(FRealTest, ConstructWithExpressionVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(3.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD z(a * b);
    EXPECT_THAT(value(z), DoubleEq(6.0));
    EXPECT_THAT(derivative(z), ElementsAre(value(b), value(a)));
}

TEST(FRealTest, AssignExpressionVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(3.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});
    AD z;

    z = a * b;
    EXPECT_THAT(value(z), DoubleEq(6.0));
    EXPECT_THAT(derivative(z), ElementsAre(value(b), value(a)));
}

TEST(FRealTest, CanSetDerivativeVec)
{
    double x0 = 1.0;
    double x1 = 2.0;

    using AD = xad::FReal<double>;
    AD x0_ad = x0;
    AD x1_ad = x1;

    derivative(x0_ad) = 1.0;
    derivative(x1_ad) = 0.0;

    EXPECT_THAT(derivative(x0_ad), DoubleEq(1.0));
    EXPECT_THAT(derivative(x1_ad), DoubleEq(0.0));
}

TEST(FRealTest, CanDoAdditionVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(2.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = a + b;

    EXPECT_THAT(value(x), DoubleEq(4.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(1.0)));
}

TEST(FRealTest, CanDoSubtractionVec)
{
    using AD = xad::FReal<double, 2>;
    AD a = 5.0;
    AD b = 2.0;

    derivative(a) = {1.0, 0.0};
    derivative(b) = {0.0, 1.0};

    AD x = a - b;

    EXPECT_THAT(value(x), DoubleEq(3.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(-1.0)));
}

TEST(FRealTest, CanDoMultiplicationVec)
{
    using AD = xad::FReal<double, 2>;
    AD a = 2.0;
    AD b = 3.0;

    derivative(a) = {1.0, 0.0};
    derivative(b) = {0.0, 1.0};

    AD x = a * b;

    EXPECT_THAT(value(x), DoubleEq(6.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(3.0), DoubleEq(2.0)));
}

TEST(FRealTest, CanDoDivisionVec)
{
    using AD = xad::FReal<double, 2>;
    AD a = 6.0;
    AD b = 1.0;

    derivative(a) = {1.0, 0.0};
    derivative(b) = {0.0, 1.0};

    AD x = a / b;

    EXPECT_THAT(value(x), DoubleEq(6.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(-6.0)));
}

TEST(FRealTest, AdditionOperatorVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(2.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    a += b;
    EXPECT_THAT(value(a), DoubleEq(4.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(1.0), DoubleEq(1.0)));
}

TEST(FRealTest, SubtractionOperatorVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(5.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    a -= b;
    EXPECT_THAT(value(a), DoubleEq(3.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(1.0), DoubleEq(-1.0)));
}

TEST(FRealTest, MultiplicationOperatorVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(2.0, {1.0, 0.0});
    AD b(3.0, {0.0, 1.0});

    a *= b;
    EXPECT_THAT(value(a), DoubleEq(6.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(3.0), DoubleEq(2.0)));
}

TEST(FRealTest, DivisionOperatorVec)
{
    using AD = xad::FReal<double, 2>;
    AD a(6.0, {1.0, 0.0});
    AD b(1.0, {0.0, 1.0});

    a /= b;
    EXPECT_THAT(value(a), DoubleEq(6.0));
    EXPECT_THAT(derivative(a), ElementsAre(DoubleEq(1.0), DoubleEq(-6.0)));
}

TEST(FRealTest, CanCompareVec)
{
    using AD = xad::FReal<double, 2>;
    AD a = 6.0;
    AD b = 1.0;

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b < a);
    EXPECT_FALSE(a == b);
}

TEST(FRealTest, CanCompareWithScalarVec)
{
    using AD = xad::FReal<double, 2>;
    AD a = 6.0;

    EXPECT_TRUE(a != 1.0);
    EXPECT_TRUE(a >= 1.0);
    EXPECT_TRUE(a > 1.0);
    EXPECT_TRUE(1.0 <= a);
    EXPECT_TRUE(1.0 < a);
    EXPECT_FALSE(a == 1.0);
}

TEST(FRealTest, CanDoPowVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(1.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = xad::pow(a, b);

    EXPECT_THAT(value(x), DoubleEq(1.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(2.0), DoubleEq(0.0)));
}

TEST(FRealTest, CanDoMaxOPVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(1.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = xad::max(a, b);

    EXPECT_THAT(value(x), value(b));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(0.0), DoubleEq(1.0)));
}

TEST(FRealTest, CanDoMinOPVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(1.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = xad::min(a, b);

    EXPECT_THAT(value(x), value(a));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(0.0)));
}

TEST(FRealTest, CanDoFmodVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(6.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = xad::fmod(a, b);

    EXPECT_THAT(value(x), DoubleEq(0.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(-3.0)));
}

TEST(FRealTest, CanDoRemainderVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(6.0, {1.0, 0.0});
    AD b(2.0, {0.0, 1.0});

    AD x = xad::remainder(a, b);

    EXPECT_THAT(value(x), DoubleEq(0.0));
    EXPECT_THAT(derivative(x), ElementsAre(DoubleEq(1.0), DoubleEq(-3.0)));
}

TEST(FRealTest, NegateVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(6.0, {1.0, 2.0});
    AD b = -(a);

    EXPECT_THAT(value(b), DoubleEq(-6.0));
    EXPECT_THAT(derivative(b), ElementsAre(DoubleEq(-1.0), DoubleEq(-2.0)));
}

TEST(FRealTest, unaryVec)
{
    using AD = xad::FReal<double, 2>;

    AD a(6.0, {1.0, 0.0});
    AD b = ++a;

    EXPECT_THAT(value(b), DoubleEq(7.0));
    EXPECT_THAT(derivative(b), ElementsAre(DoubleEq(1.0), DoubleEq(0.0)));
}

///////// FReal<Scalar> /////////

TEST(FRealTest, CanConstructFReal)
{
    using AD = xad::FReal<double>;
    AD a;

    EXPECT_THAT(value(a), DoubleEq(0.0));
    EXPECT_THAT(derivative(a), DoubleEq(0.0));
}

TEST(FRealTest, CanAssignToFReal)
{
    using AD = xad::FReal<double>;
    AD a(1.0, 1.0);

    EXPECT_THAT(value(a), DoubleEq(1.0));
    EXPECT_THAT(derivative(a), DoubleEq(1.0));
}

TEST(FRealTest, CanCopyConstruct)
{
    using AD = xad::FReal<double>;
    AD x0 = 2.0;
    derivative(x0) = 1.0;

    AD x1(x0);

    EXPECT_THAT(value(x1), value(x0));
    EXPECT_THAT(derivative(x0), derivative(x1));
}

TEST(FRealTest, CanAssignValueToExistingObject)
{
    using AD = xad::FReal<double>;
    AD x0(2.0, 1.0);
    x0 = 4.0;
    EXPECT_THAT(value(x0), DoubleEq(4.0));
    EXPECT_THAT(derivative(x0), DoubleEq(0.0));
}

TEST(FRealTest, ConstructWithExpression)
{
    using AD = xad::FReal<double>;

    AD a(3.0, 1.0);
    AD b(2.0, 0.0);

    AD z(a * b);
    EXPECT_THAT(value(z), DoubleEq(6.0));
    EXPECT_THAT(derivative(z), value(b));
}

TEST(FRealTest, AssignExpression)
{
    using AD = xad::FReal<double>;

    AD a(3.0, 1.0);
    AD b(2.0, 0.0);
    AD z;

    z = a * b;
    EXPECT_THAT(value(z), DoubleEq(6.0));
    EXPECT_THAT(derivative(z), value(b));
}

TEST(FRealTest, CanSetDerivative)
{
    double x0 = 1.0;
    double x1 = 2.0;

    using AD = xad::FReal<double, 4>;
    AD x0_ad = x0;
    AD x1_ad = x1;

    derivative(x0_ad) = {1.0, 0.0, 0.0, 0.0};
    derivative(x1_ad) = {0.0, 1.0, 0.0, 0.0};

    EXPECT_THAT(derivative(x0_ad)[0], DoubleEq(1.0));
    EXPECT_THAT(derivative(x1_ad)[1], DoubleEq(1.0));
}

TEST(FRealTest, CanDoAddition)
{
    using AD = xad::FReal<double>;
    AD a(2.0, 1.0);
    AD b(2.0, 0.0);

    AD x = a + b;

    EXPECT_THAT(value(x), DoubleEq(4.0));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, CanDoSubtraction)
{
    using AD = xad::FReal<double>;
    AD a = 5.0;
    AD b = 2.0;

    derivative(a) = 1.0;
    derivative(b) = 0.0;

    AD x = a - b;

    EXPECT_THAT(value(x), DoubleEq(3.0));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, CanDoMultiplication)
{
    using AD = xad::FReal<double>;
    AD a = 2.0;
    AD b = 3.0;

    derivative(a) = 1.0;
    derivative(b) = 0.0;

    AD x = a * b;

    EXPECT_THAT(value(x), DoubleEq(6.0));
    EXPECT_THAT(derivative(x), DoubleEq(3.0));
}

TEST(FRealTest, CanDoDivision)
{
    using AD = xad::FReal<double>;
    AD a = 6.0;
    AD b = 1.0;

    derivative(a) = 1.0;
    derivative(b) = 0.0;

    AD x = a / b;

    EXPECT_THAT(value(x), DoubleEq(6.0));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, AdditionOperator)
{
    using AD = xad::FReal<double>;
    AD a(2.0, 1.0);
    AD b(2.0, 0.0);

    a += b;
    EXPECT_THAT(value(a), DoubleEq(4.0));
    EXPECT_THAT(derivative(a), DoubleEq(1.0));
}

TEST(FRealTest, SubtractionOperator)
{
    using AD = xad::FReal<double>;
    AD a(5.0, 1.0);
    AD b(2.0, 0.0);

    a -= b;
    EXPECT_THAT(value(a), DoubleEq(3.0));
    EXPECT_THAT(derivative(a), DoubleEq(1.0));
}

TEST(FRealTest, MultiplicationOperator)
{
    using AD = xad::FReal<double>;
    AD a(2.0, 1.0);
    AD b(3.0, 0.0);

    a *= b;
    EXPECT_THAT(value(a), DoubleEq(6.0));
    EXPECT_THAT(derivative(a), DoubleEq(3.0));
}

TEST(FRealTest, DivisionOperator)
{
    using AD = xad::FReal<double>;
    AD a(6.0, 1.0);
    AD b(1.0, 0.0);

    a /= b;
    EXPECT_THAT(value(a), DoubleEq(6.0));
    EXPECT_THAT(derivative(a), DoubleEq(1.0));
}

TEST(FRealTest, CanCompare)
{
    using AD = xad::FReal<double>;
    AD a = 6.0;
    AD b = 1.0;

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b < a);
    EXPECT_FALSE(a == b);
}

TEST(FRealTest, CanCompareWithScalar)
{
    using AD = xad::FReal<double>;
    AD a = 6.0;

    EXPECT_TRUE(a != 1.0);
    EXPECT_TRUE(a >= 1.0);
    EXPECT_TRUE(a > 1.0);
    EXPECT_TRUE(1.0 <= a);
    EXPECT_TRUE(1.0 < a);
    EXPECT_FALSE(a == 1.0);
}

TEST(FRealTest, CanDoPow)
{
    using AD = xad::FReal<double>;

    AD a(1.0, 1.0);
    AD b(2.0, 0.0);

    AD x = xad::pow(a, b);

    EXPECT_THAT(value(x), DoubleEq(1.0));
    EXPECT_THAT(derivative(x), DoubleEq(2.0));
}

TEST(FRealTest, CanDoMaxOP)
{
    using AD = xad::FReal<double>;

    AD a(1.0, 1.0);
    AD b(2.0, 0.0);

    AD x = xad::max(a, b);

    EXPECT_THAT(value(x), value(b));
    EXPECT_THAT(derivative(x), DoubleEq(0.0));
}

TEST(FRealTest, CanDoMinOP)
{
    using AD = xad::FReal<double>;

    AD a(1.0, 1.0);
    AD b(2.0, 0.0);

    AD x = xad::min(a, b);

    EXPECT_THAT(value(x), value(a));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, CanDoFmod)
{
    using AD = xad::FReal<double>;

    AD a(6.0, 1.0);
    AD b(2.0, 0.0);

    AD x = xad::fmod(a, b);

    EXPECT_THAT(value(x), DoubleEq(0.0));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, CanDoRemainder)
{
    using AD = xad::FReal<double>;

    AD a(6.0, 1.0);
    AD b(2.0, 0.0);

    AD x = xad::remainder(a, b);

    EXPECT_THAT(value(x), DoubleEq(0.0));
    EXPECT_THAT(derivative(x), DoubleEq(1.0));
}

TEST(FRealTest, UnaryOp)
{
    using AD = xad::FReal<double>;

    AD a(6.0, 1.0);
    AD b = ++a;

    EXPECT_THAT(value(b), DoubleEq(7.0));
    EXPECT_THAT(derivative(b), DoubleEq(1.0));
}

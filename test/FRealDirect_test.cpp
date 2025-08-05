/*******************************************************************************

   Tests for FRealDirect .

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

#include <XAD/FRealDirect.hpp>

using namespace ::testing;

TEST(FRealDirectTest, hasInitialValueOfZero)
{
    xad::FRealDirect<double> obj;
    EXPECT_THAT(obj.value(), DoubleEq(0.0));
    EXPECT_THAT(obj.derivative(), DoubleEq(0.0));
}

TEST(FRealDirectTest, ConstructWithValue)
{
    xad::FRealDirect<double> obj(42., 4.2);
    EXPECT_THAT(obj.value(), DoubleEq(42.0));
    EXPECT_THAT(obj.derivative(), DoubleEq(4.2));
}

TEST(FRealDirectTest, CanCopyConstruct)
{
    xad::FRealDirect<double> obj(1337, 2);
    xad::FRealDirect<double> cp(obj);
    EXPECT_THAT(obj.value(), DoubleEq(1337.0));
    EXPECT_THAT(obj.derivative(), DoubleEq(2.0));
}

TEST(FRealDirectTest, CanGetValue)
{
    xad::FRealDirect<double> obj(1337, 2);
    EXPECT_THAT(obj.value(), DoubleEq(1337.));
    EXPECT_THAT(value(obj), DoubleEq(1337.));
    EXPECT_THAT(obj.derivative(), DoubleEq(2.));
}

TEST(FRealDirectTest, CanGetValueAndDerivativeByReference)
{
    xad::FRealDirect<double> obj(1, 1);
    EXPECT_THAT(obj.value(), DoubleEq(1));
    EXPECT_THAT(obj.derivative(), DoubleEq(1));
    obj.derivative() = 42.0;
    obj.value() = 42.0;
    EXPECT_THAT(obj.value(), DoubleEq(42.0));
    EXPECT_THAT(obj.derivative(), DoubleEq(42.));
}

TEST(FRealDirectTest, TestValueAndDerivativeGetterAndSetter)
{
    xad::FRealDirect<double> obj(42, 42);
    EXPECT_THAT(obj.getValue(), DoubleEq(42.));
    EXPECT_THAT(obj.getDerivative(), DoubleEq(42.));
    obj.setDerivative(1.0);
    EXPECT_THAT(obj.derivative(), DoubleEq(1.));
}

TEST(FRealDirectTest, CanGetValueUsingGlobalValue)
{
    xad::FRealDirect<double> obj(1337, 2);
    EXPECT_THAT(obj.value(), DoubleEq(1337.));
    EXPECT_THAT(obj.derivative(), DoubleEq(2.));
}

TEST(FRealDirectTest, CanAssignObject)
{
    xad::FRealDirect<double> obj(1337, 2);
    xad::FRealDirect<double> newObj = obj;
    EXPECT_THAT(newObj.value(), DoubleEq(1337.));
    EXPECT_THAT(newObj.derivative(), DoubleEq(2.));
}

TEST(FRealDirectTest, CanPerformAddition)
{
    xad::FRealDirect<double> obj1(2, 7);
    xad::FRealDirect<double> obj2(3, 2);

    xad::FRealDirect<double> obj3 = obj1 + obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));
    EXPECT_THAT(obj3.derivative(), DoubleEq(9));

    xad::FRealDirect<double> x = 2.0;

    auto y1 = x + 2;
    auto y2 = x + 2.0;
    auto y3 = 2 + x;
    auto y4 = 2.0 + x;

    EXPECT_THAT(y1.value(), DoubleEq(4));
    EXPECT_THAT(y2.value(), DoubleEq(4));
    EXPECT_THAT(y3.value(), DoubleEq(4));
    EXPECT_THAT(y4.value(), DoubleEq(4));
}

TEST(FRealDirectTest, CanPerformSubtraction)
{
    xad::FRealDirect<double> obj1(5, 7);
    xad::FRealDirect<double> obj2(3, 2);

    xad::FRealDirect<double> obj3 = obj1 - obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(2));
    EXPECT_THAT(obj3.derivative(), DoubleEq(5));

    xad::FRealDirect<double> x = 4.0;

    auto y1 = x - 2;
    auto y2 = x - 2.0;
    auto y3 = 2 - x;
    auto y4 = 2.0 - x;

    EXPECT_THAT(y1.value(), DoubleEq(2));
    EXPECT_THAT(y2.value(), DoubleEq(2));
    EXPECT_THAT(y3.value(), DoubleEq(-2));
    EXPECT_THAT(y4.value(), DoubleEq(-2));
}

TEST(FRealDirectTest, CanPerformDivision)
{
    xad::FRealDirect<double> obj1(5, 1);
    xad::FRealDirect<double> obj2(1, 0);

    xad::FRealDirect<double> obj3 = obj1 / obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));
    EXPECT_THAT(obj3.derivative(), DoubleEq(1));

    xad::FRealDirect<double> x(5, 1);

    auto y1 = x / 1;
    auto y2 = x / 1.0;
    auto y3 = 5 / x;
    auto y4 = 5.0 / x;

    EXPECT_THAT(y1.value(), DoubleEq(5));
    EXPECT_THAT(y1.derivative(), DoubleEq(1));
    EXPECT_THAT(y2.value(), DoubleEq(5));
    EXPECT_THAT(y2.derivative(), DoubleEq(1));
    EXPECT_THAT(y3.value(), DoubleEq(1));
    EXPECT_THAT(y4.value(), DoubleEq(1));
}

TEST(FRealDirectTest, CanPerformMultiplication)
{
    xad::FRealDirect<double> obj1(5, 1);
    xad::FRealDirect<double> obj2(1, 0);

    xad::FRealDirect<double> obj3 = obj1 * obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));
    EXPECT_THAT(obj3.derivative(), DoubleEq(1));

    xad::FRealDirect<double> x(5, 1);

    auto y1 = x * 1;
    auto y2 = x * 1.0;
    auto y3 = 1 * x;
    auto y4 = 1.0 * x;

    EXPECT_THAT(y1.value(), DoubleEq(5));
    EXPECT_THAT(y1.derivative(), DoubleEq(1));
    EXPECT_THAT(y2.value(), DoubleEq(5));
    EXPECT_THAT(y2.derivative(), DoubleEq(1));
    EXPECT_THAT(y3.value(), DoubleEq(5));
    EXPECT_THAT(y3.derivative(), DoubleEq(1));
    EXPECT_THAT(y4.value(), DoubleEq(5));
    EXPECT_THAT(y4.derivative(), DoubleEq(1));
}

TEST(FRealDirectTest, simpleMathTest)
{
    xad::FRealDirect<double> ob(3, 2);
    xad::FRealDirect<double> ob2(0, 0);

    xad::FRealDirect<double> ob3 = xad::min(ob, ob2);

    EXPECT_THAT(ob3.value(), 0);
    EXPECT_THAT(ob3.derivative(), 0);
}

TEST(FRealDirectTest, canAddValueToTheInstance2)
{
    xad::FRealDirect<double> obj1(2, 7);
    xad::FRealDirect<double> obj2(3, 2);

    obj1 += obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));
    EXPECT_THAT(obj1.derivative(), DoubleEq(9));

    xad::FRealDirect<double> x = 2;

    x += 2;
    EXPECT_THAT(x.value(), DoubleEq(4));
    x += 2.0;
    EXPECT_THAT(x.value(), DoubleEq(6));
}

TEST(FRealDirectTest, CanBeSelfSubstracted2)
{
    xad::FRealDirect<double> obj1(5, 7);
    xad::FRealDirect<double> obj2(3, 2);

    obj1 -= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(2));
    EXPECT_THAT(obj1.derivative(), DoubleEq(5));

    xad::FRealDirect<double> x = 6;

    x -= 2;
    EXPECT_THAT(x.value(), DoubleEq(4));
    x -= 2.0;
    EXPECT_THAT(x.value(), DoubleEq(2));
}

TEST(FRealDirectTest, CanMultiplyByItself2)
{
    xad::FRealDirect<double> obj1(5, 1);
    xad::FRealDirect<double> obj2(1, 0);

    obj1 *= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));
    EXPECT_THAT(obj1.derivative(), DoubleEq(1));

    xad::FRealDirect<double> x = 5;

    x *= 2;
    EXPECT_THAT(x.value(), DoubleEq(10));
    x *= 2.0;
    EXPECT_THAT(x.value(), DoubleEq(20));
}

TEST(FRealDirectTest, CanDivideByItself2)
{
    xad::FRealDirect<double> obj1(5, 1);
    xad::FRealDirect<double> obj2(1, 0);

    obj1 /= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));
    EXPECT_THAT(obj1.derivative(), DoubleEq(1));

    xad::FRealDirect<double> x = 5;

    x /= 1;
    EXPECT_THAT(x.value(), DoubleEq(5));
    x /= 1.0;
    EXPECT_THAT(x.value(), DoubleEq(5));
}

TEST(FRealDirectTest, CanBeNegated2)
{
    xad::FRealDirect<double> obj1(5, 1);
    xad::FRealDirect<double> obj2 = -obj1;

    EXPECT_THAT(obj2.value(), DoubleEq(-5));
    EXPECT_THAT(obj2.derivative(), DoubleEq(-1));
}

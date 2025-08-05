/*******************************************************************************

   Tests for ARealDirect .

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

#include <XAD/ARealDirect.hpp>
#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

TEST(ARealDirectTest, hasInitialValueOfZero)
{
    xad::ARealDirect<double> obj;
    EXPECT_THAT(obj.value(), DoubleEq(0.0));
}

TEST(ARealDirectTest, ConstructWithValue)
{
    xad::ARealDirect<double> obj(42);
    EXPECT_THAT(obj.value(), DoubleEq(42.0));
}

TEST(ARealDirectTest, CanCopyConstruct)
{
    xad::ARealDirect<double> obj(1337);
    xad::ARealDirect<double> cp(obj);
    EXPECT_THAT(obj.value(), DoubleEq(1337.0));
}

TEST(ARealDirectTest, CanGetValue)
{
    xad::ARealDirect<double> obj(1337);
    EXPECT_THAT(obj.value(), DoubleEq(1337.));
    EXPECT_THAT(value(obj), DoubleEq(1337.));
}

TEST(ARealDirectTest, CanUpdateValueAndDerivative)
{
    xad::ARealDirect<double> obj;
    EXPECT_THAT(obj.value(), DoubleEq(0.));
    obj.value() = 42.0;
    EXPECT_THAT(obj.value(), DoubleEq(42.0));
}

TEST(ARealDirectTest, CanGetValueUsingGlobalValue)
{
    xad::ARealDirect<double> obj(1337);
    EXPECT_THAT(obj.value(), DoubleEq(1337.));
}

TEST(ARealDirectTest, CanAssignObject)
{
    xad::ARealDirect<double> obj(1337);
    xad::ARealDirect<double> newObj = obj;
    EXPECT_THAT(obj.value(), DoubleEq(1337.));
}

TEST(ARealDirectTest, BasicTest)
{
    xad::Tape<double> tape;
    xad::AReal<double> x1 = 0., x2 = 2.0;
    tape.registerInput(x1);
    tape.registerInput(x2);
    tape.newRecording();
    xad::AReal<double> y = sin(x1) + x1 * x2;
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();

    EXPECT_THAT(y.value(), DoubleEq(0));
    EXPECT_THAT(y.derivative(), DoubleEq(0));
    EXPECT_THAT(x1.derivative(), DoubleEq(3));  // dy/dx1
    EXPECT_THAT(x2.derivative(), DoubleEq(0));  // dy/dx2
}

TEST(ARealDirectTest, CanPerformAddition)
{
    xad::ARealDirect<double> obj1(2);
    xad::ARealDirect<double> obj2(3);

    xad::ARealDirect<double> obj3 = obj1 + obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 2.0;

    auto y1 = x + 2;
    auto y2 = x + 2.0;
    auto y3 = 2 + x;
    auto y4 = 2.0 + x;

    EXPECT_THAT(y1.value(), DoubleEq(4));
    EXPECT_THAT(y2.value(), DoubleEq(4));
    EXPECT_THAT(y3.value(), DoubleEq(4));
    EXPECT_THAT(y4.value(), DoubleEq(4));
}

TEST(ARealDirectTest, CanPerformSubtraction)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(3);

    xad::ARealDirect<double> obj3 = obj1 - obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(2));

    xad::ARealDirect<double> x = 4.0;

    auto y1 = x - 2;
    auto y2 = x - 2.0;
    auto y3 = 2 - x;
    auto y4 = 2.0 - x;

    EXPECT_THAT(y1.value(), DoubleEq(2));
    EXPECT_THAT(y2.value(), DoubleEq(2));
    EXPECT_THAT(y3.value(), DoubleEq(-2));
    EXPECT_THAT(y4.value(), DoubleEq(-2));
}

TEST(ARealDirectTest, CanPerformDivision)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(1);

    xad::ARealDirect<double> obj3 = obj1 / obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 5;

    auto y1 = x / 1;
    auto y2 = x / 1.0;
    auto y3 = 5 / x;
    auto y4 = 5.0 / x;

    EXPECT_THAT(y1.value(), DoubleEq(5));
    EXPECT_THAT(y2.value(), DoubleEq(5));
    EXPECT_THAT(y3.value(), DoubleEq(1));
    EXPECT_THAT(y4.value(), DoubleEq(1));
}

TEST(ARealDirectTest, CanPerformMultiplication)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(1);

    xad::ARealDirect<double> obj3 = obj1 * obj2;
    EXPECT_THAT(obj3.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 5;

    auto y1 = x * 1;
    auto y2 = x * 1.0;
    auto y3 = 1 * x;
    auto y4 = 1.0 * x;

    EXPECT_THAT(y1.value(), DoubleEq(5));
    EXPECT_THAT(y2.value(), DoubleEq(5));
    EXPECT_THAT(y3.value(), DoubleEq(5));
    EXPECT_THAT(y4.value(), DoubleEq(5));
}

TEST(FRealDirectTest, canAddValueToTheInstance)
{
    xad::ARealDirect<double> obj1(2);
    xad::ARealDirect<double> obj2(3);

    obj1 += obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 2;

    x += 2;
    EXPECT_THAT(x.value(), DoubleEq(4));
    x += 2.0;
    EXPECT_THAT(x.value(), DoubleEq(6));
}

TEST(FRealDirectTest, CanBeSelfSubstracted)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(3);

    obj1 -= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(2));

    xad::ARealDirect<double> x = 6;

    x -= 2;
    EXPECT_THAT(x.value(), DoubleEq(4));
    x -= 2.0;
    EXPECT_THAT(x.value(), DoubleEq(2));
}

TEST(FRealDirectTest, CanMultiplyByItself)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(1);

    obj1 *= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 5;

    x *= 2;
    EXPECT_THAT(x.value(), DoubleEq(10));
    x *= 2.0;
    EXPECT_THAT(x.value(), DoubleEq(20));
}

TEST(FRealDirectTest, CanDivideByItself)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2(1);

    obj1 /= obj2;
    EXPECT_THAT(obj1.value(), DoubleEq(5));

    xad::ARealDirect<double> x = 5;

    x /= 1;
    EXPECT_THAT(x.value(), DoubleEq(5));
    x /= 1.0;
    EXPECT_THAT(x.value(), DoubleEq(5));
}

TEST(FRealDirectTest, CanBeNegated)
{
    xad::ARealDirect<double> obj1(5);
    xad::ARealDirect<double> obj2 = -obj1;

    EXPECT_THAT(obj2.value(), DoubleEq(-5));
}

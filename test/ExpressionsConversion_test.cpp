/*******************************************************************************

   Unit tests for derivatives of arithmetic and logical expressions which
   require type conversions. This is separated into a new translation unit
   since type conversion warnings should be disabled in the compiler.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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
#include <gmock/gmock.h>

using namespace ::testing;

TEST(Expressions, canCompareOtherTypes)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;

    EXPECT_TRUE(x1 == 2);
    EXPECT_TRUE(x1 != 3);
    EXPECT_TRUE(x1 < 2.2f);
    EXPECT_TRUE(x1 <= 2.0f);
    EXPECT_TRUE(x1 > short(-1));
    EXPECT_TRUE(x1 >= 1ULL);

    EXPECT_TRUE(2 == x1);
    EXPECT_TRUE(3 != x1);
    EXPECT_TRUE(2.2f > x1);
    EXPECT_TRUE(2.0f >= x1);
    EXPECT_TRUE(short(-1) < x1);
    EXPECT_TRUE(1ULL <= x1);
}

TEST(Expressions, canCompareOtherTypesFwd)
{
    xad::FAD x1 = 2.0;

    EXPECT_TRUE(x1 == 2);
    EXPECT_TRUE(x1 != 3);
    EXPECT_TRUE(x1 < 2.2f);
    EXPECT_TRUE(x1 <= 2.0f);
    EXPECT_TRUE(x1 > short(-1));
    EXPECT_TRUE(x1 >= 1ULL);

    EXPECT_TRUE(2 == x1);
    EXPECT_TRUE(3 != x1);
    EXPECT_TRUE(2.2f > x1);
    EXPECT_TRUE(2.0f >= x1);
    EXPECT_TRUE(short(-1) < x1);
    EXPECT_TRUE(1ULL <= x1);
}

enum TestEnum
{
    TEST_VAL0 = 0,
    TEST_VAL1 = 1
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wenum-float-conversion"
#endif

TEST(Expressions, canImplicitlyConvertEnums)
{
    // these expressions should purely compile
    xad::AD a = TEST_VAL1;
    xad::AD b = a + 2.0 * TEST_VAL1;
    xad::AD c = a * TEST_VAL1 + b;
    xad::AD d = xad::pow(a, TEST_VAL1);

    xad::AD e = a + TEST_VAL1 * 2.0;
    xad::AD f = TEST_VAL1 * a + b;
    xad::AD g = xad::pow(TEST_VAL1, a);

    EXPECT_DOUBLE_EQ(value(a), 1.0);
    EXPECT_DOUBLE_EQ(value(b), 3.0);
    EXPECT_DOUBLE_EQ(value(c), 4.0);
    EXPECT_DOUBLE_EQ(value(d), 1.0);
    EXPECT_DOUBLE_EQ(value(e), 3.0);
    EXPECT_DOUBLE_EQ(value(f), 4.0);
    EXPECT_DOUBLE_EQ(value(g), 1.0);
}

TEST(Expressions, canIncrementWithEnums)
{
    xad::AD x = 1.0;
    x += TEST_VAL1;

    EXPECT_DOUBLE_EQ(value(x), 2.0);
}

TEST(Expressions, canDecrementWithEnums)
{
    xad::AD x = 1.0;
    x -= TEST_VAL1;

    EXPECT_DOUBLE_EQ(value(x), 0.0);
}

TEST(Expressions, canMultiplyWithEnums)
{
    xad::AD x = 1.0;
    x *= TEST_VAL1;

    EXPECT_DOUBLE_EQ(value(x), 1.0);
}

TEST(Expressions, canDivideWithEnums)
{
    xad::AD x = 1.0;
    x /= TEST_VAL1;

    EXPECT_DOUBLE_EQ(value(x), 1.0);
}

TEST(Expressions, canCompareToEnums)
{
    xad::AD a = TEST_VAL1;
    xad::AD b = TEST_VAL0;

    EXPECT_TRUE(a == TEST_VAL1);
    EXPECT_TRUE(a * 1.0 == TEST_VAL1);
    EXPECT_FALSE(b == TEST_VAL1);
    EXPECT_FALSE(b * 1.0 == TEST_VAL1);
    EXPECT_TRUE(TEST_VAL1 == a);
    EXPECT_TRUE(TEST_VAL1 == a * 1.0);
    EXPECT_FALSE(TEST_VAL1 == b);
    EXPECT_FALSE(TEST_VAL1 == b * 1.0);

    EXPECT_FALSE(a != TEST_VAL1);
    EXPECT_FALSE(a * 1.0 != TEST_VAL1);
    EXPECT_TRUE(b != TEST_VAL1);
    EXPECT_TRUE(b * 1.0 != TEST_VAL1);
    EXPECT_FALSE(TEST_VAL1 != a);
    EXPECT_FALSE(TEST_VAL1 != a * 1.0);
    EXPECT_TRUE(TEST_VAL1 != b);
    EXPECT_TRUE(TEST_VAL1 != b * 1.0);

    EXPECT_TRUE(a > TEST_VAL0);
    EXPECT_FALSE(b > TEST_VAL0);
    EXPECT_TRUE(a >= TEST_VAL0);
    EXPECT_TRUE(a >= TEST_VAL1);
    EXPECT_TRUE(b >= TEST_VAL0);
    EXPECT_FALSE(b >= TEST_VAL1);
    EXPECT_TRUE(a * 1.0 > TEST_VAL0);
    EXPECT_FALSE(b * 1.0 > TEST_VAL0);
    EXPECT_TRUE(a * 1.0 >= TEST_VAL0);
    EXPECT_TRUE(a * 1.0 >= TEST_VAL1);
    EXPECT_TRUE(b * 1.0 >= TEST_VAL0);
    EXPECT_FALSE(b * 1.0 >= TEST_VAL1);

    EXPECT_FALSE(a < TEST_VAL0);
    EXPECT_FALSE(b < TEST_VAL0);
    EXPECT_FALSE(a <= TEST_VAL0);
    EXPECT_TRUE(a <= TEST_VAL1);
    EXPECT_TRUE(b <= TEST_VAL0);
    EXPECT_TRUE(b <= TEST_VAL1);
    EXPECT_FALSE(a * 1.0 < TEST_VAL0);
    EXPECT_FALSE(b * 1.0 < TEST_VAL0);
    EXPECT_FALSE(a * 1.0 <= TEST_VAL0);
    EXPECT_TRUE(a * 1.0 <= TEST_VAL1);
    EXPECT_TRUE(b * 1.0 <= TEST_VAL0);
    EXPECT_TRUE(b * 1.0 <= TEST_VAL1);

    EXPECT_FALSE(TEST_VAL0 > a);
    EXPECT_FALSE(TEST_VAL0 > b);
    EXPECT_FALSE(TEST_VAL0 >= a);
    EXPECT_TRUE(TEST_VAL1 >= a);
    EXPECT_TRUE(TEST_VAL0 >= b);
    EXPECT_TRUE(TEST_VAL1 >= b);
    EXPECT_FALSE(TEST_VAL0 > a * 1.0);
    EXPECT_FALSE(TEST_VAL0 > b * 1.0);
    EXPECT_FALSE(TEST_VAL0 >= a * 1.0);
    EXPECT_TRUE(TEST_VAL1 >= a * 1.0);
    EXPECT_TRUE(TEST_VAL0 >= b * 1.0);
    EXPECT_TRUE(TEST_VAL1 >= b * 1.0);

    EXPECT_TRUE(TEST_VAL0 < a);
    EXPECT_FALSE(TEST_VAL0 < b);
    EXPECT_TRUE(TEST_VAL0 <= a);
    EXPECT_TRUE(TEST_VAL1 <= a);
    EXPECT_TRUE(TEST_VAL0 <= b);
    EXPECT_FALSE(TEST_VAL1 <= b);
    EXPECT_TRUE(TEST_VAL0 < a * 1.0);
    EXPECT_FALSE(TEST_VAL0 < b * 1.0);
    EXPECT_TRUE(TEST_VAL0 <= a * 1.0);
    EXPECT_TRUE(TEST_VAL1 <= a * 1.0);
    EXPECT_TRUE(TEST_VAL0 <= b * 1.0);
    EXPECT_FALSE(TEST_VAL1 <= b * 1.0);
}

struct TestConv
{
    operator xad::AD() const { return x_; }
    xad::AD x_;
};

TEST(Expressions, canImplicitlyConvertClasses)
{
    TestConv x{2.0};
    xad::AD a = x;
    xad::AD c = a * x;
    xad::AD d = x * a;
    xad::AD e = pow(a, x);

    EXPECT_DOUBLE_EQ(value(a), 2.0);
    EXPECT_DOUBLE_EQ(value(c), 4.0);
    EXPECT_DOUBLE_EQ(value(d), 4.0);
    EXPECT_DOUBLE_EQ(value(e), 4.0);
}

TEST(Expressions, canIncrementWithConvertClasses)
{
    xad::AD x = 1.0;
    TestConv c{1.0};
    x += c;

    EXPECT_DOUBLE_EQ(value(x), 2.0);
}

TEST(Expressions, canDecrementWithConvertClasses)
{
    xad::AD x = 1.0;
    TestConv c{1.0};
    x -= c;

    EXPECT_DOUBLE_EQ(value(x), 0.0);
}

TEST(Expressions, canMultiplyWithConvertClasses)
{
    xad::AD x = 1.0;
    TestConv c{1.0};
    x *= c;

    EXPECT_DOUBLE_EQ(value(x), 1.0);
}

TEST(Expressions, canDivideWithConvertClasses)
{
    xad::AD x = 1.0;
    TestConv c{1.0};
    x /= c;

    EXPECT_DOUBLE_EQ(value(x), 1.0);
}

TEST(Expressions, canCompareToConvertibleClasses)
{
    TestConv x{2.0};
    xad::AD a = 2.0;
    xad::AD b = 1.0;

    EXPECT_TRUE(a == x);
    EXPECT_TRUE(a * 1.0 == x);
    EXPECT_FALSE(b == x);
    EXPECT_FALSE(b * 1.0 == x);
    EXPECT_FALSE(a != x);
    EXPECT_FALSE(a * 1.0 != x);
    EXPECT_TRUE(b != x);
    EXPECT_TRUE(b * 1.0 != x);
    EXPECT_TRUE(x == a);
    EXPECT_TRUE(x == a * 1.0);
    EXPECT_FALSE(x == b);
    EXPECT_FALSE(x == b * 1.0);
    EXPECT_FALSE(x != a);
    EXPECT_FALSE(x != a * 1.0);
    EXPECT_TRUE(x != b);
    EXPECT_TRUE(x != b * 1.0);

    // we leave out the < > <= >= here, as they are implemented exactly the same
    // as == and !=, using the same macro.
}

TEST(Expressions, canImplicitlyConvertToBoolean)
{
    xad::AD zero = 0.0;
    xad::AD one = 1.0;

    EXPECT_TRUE(bool(one));
    EXPECT_FALSE(bool(zero));

    EXPECT_FALSE(bool(zero * one));
    EXPECT_TRUE(bool(one + 1.2 - pown(one, 2)));

    if (zero)
        FAIL() << "zero evaluated to true in if";
    if (!one)
        FAIL() << "one evaluated to false in if";
    if (zero * one)
        FAIL() << "zero  in expression evaluated to true in if";
    if (one + 1.2 - pown(one, 2))
    {
    }
    else
        FAIL() << "long expression evaluated to false in if";
}

#ifdef XAD_ALLOW_INT_CONVERSION

template <class T>
class ExpressionsIntConversion : public ::testing::Test
{
};

typedef ::testing::Types<char, unsigned char, signed char, short, unsigned short, int, unsigned int,
                         long, unsigned long, long long, unsigned long long>
    int_test_types;

TYPED_TEST_SUITE(ExpressionsIntConversion, int_test_types);

TYPED_TEST(ExpressionsIntConversion, canConvertARealToIntegers)
{ 
    xad::AReal<double> x = 42.0;
    
    TypeParam i = static_cast<TypeParam>(x);
    TypeParam j = (TypeParam)x;
    TypeParam k = TypeParam(x);

    EXPECT_THAT(i, Eq(j));
    EXPECT_THAT(i, Eq(k));
    EXPECT_THAT(j, Eq(k));
}

TYPED_TEST(ExpressionsIntConversion, canConvertFRealToIntegers)
{
    xad::FReal<double> x = 42.0;

    TypeParam i = static_cast<TypeParam>(x);
    TypeParam j = (TypeParam)x;
    TypeParam k = TypeParam(x);

    EXPECT_THAT(i, Eq(j));
    EXPECT_THAT(i, Eq(k));
    EXPECT_THAT(j, Eq(k));
}

TYPED_TEST(ExpressionsIntConversion, canConvertARealExprToIntegers)
{ 
    xad::AReal<double> x = 42.0;
    
    TypeParam i = static_cast<TypeParam>(floor(x));
    TypeParam j = (TypeParam)floor(x);
    TypeParam k = TypeParam(floor(x));

    EXPECT_THAT(i, Eq(j));
    EXPECT_THAT(i, Eq(k));
    EXPECT_THAT(j, Eq(k));
}

TYPED_TEST(ExpressionsIntConversion, canConvertFRealExprToIntegers)
{
    xad::FReal<double> x = 42.0;

    TypeParam i = static_cast<TypeParam>(floor(x));
    TypeParam j = (TypeParam)floor(x);
    TypeParam k = TypeParam(floor(x));

    EXPECT_THAT(i, Eq(j));
    EXPECT_THAT(i, Eq(k));
    EXPECT_THAT(j, Eq(k));
}

#endif
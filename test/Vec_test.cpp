/*******************************************************************************

   Unit tests for the Vec type.

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

#include <XAD/Vec.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

TEST(Vec, can_be_initialized)
{
    xad::Vec<int, 4> v = {-7, 2, 3, 5};
    EXPECT_THAT(v, ElementsAre(-7, 2, 3, 5));
}

TEST(Vec, can_be_updated)
{
    xad::Vec<int, 4> v = {1, 2, 3, 4};
    v[0] = 8;
    v[1] = 9;
    v[2] = 10;
    v[3] = 11;
    EXPECT_THAT(v, ElementsAre(8, 9, 10, 11));
}

TEST(Vec, check_size)
{
    xad::Vec<int, 4> v = {-7, 2, 3, 5};
    EXPECT_THAT(v.size(), 4);
}

TEST(Vec, is_empty)
{
    xad::Vec<double, 0> v1 = {};
    xad::Vec<double, 3> v2 = {0.1, 0.2, 0.3};

    EXPECT_THAT(v1.empty(), true);
    EXPECT_THAT(v2.empty(), false);
}

TEST(Vec, check_assign)
{
    xad::Vec<double, 4> v1 = {1.0, 2.0, 3.0, 4.0};
    v1 = 7.0;
    EXPECT_THAT(v1, ElementsAre(7.0, 7.0, 7.0, 7.0));
}

TEST(Vec, equality_to_self)
{
    xad::Vec<double, 4> v1 = {1.0, 2.0, 4.0, 8.0};
    xad::Vec<double, 4> v2 = {1.0, 2.0, 4.0, 8.0};
    xad::Vec<double, 4> v3 = {3.0, 7.0, 5.0, 8.0};
    EXPECT_THAT(v1 == v2, true);
    EXPECT_THAT(v1 == v3, false);
    EXPECT_THAT(v1 != v2, false);
    EXPECT_THAT(v1 != v3, true);
}

TEST(Vec, equality_to_scalar)
{
    xad::Vec<double, 4> v1 = {7.0, 7.0, 7.0, 7.0};
    EXPECT_THAT(v1 == 7.0, true);
    EXPECT_THAT(v1 == 4.0, false);
    EXPECT_THAT(v1 != 7.0, false);
    EXPECT_THAT(v1 != 3.0, true);
}

TEST(Vec, can_be_self_added)
{
    xad::Vec<double, 5> v1 = {1, 2, 3, 4, 5};
    xad::Vec<double, 5> v2 = {1, 0, -1, -2, -3};
    v1 += v2;
    EXPECT_THAT(v1 == 2, true);
}

TEST(Vec, can_be_added)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v += 10.0;
    EXPECT_THAT(v, ElementsAre(11.0, 12.0, 13.0, 14.0, 15.0));
}

TEST(Vec, can_be_left_added)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = v + 10.0;
    EXPECT_THAT(v, ElementsAre(11.0, 12.0, 13.0, 14.0, 15.0));
}

TEST(Vec, can_be_right_added)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = 10.0 + v;
    EXPECT_THAT(v, ElementsAre(11.0, 12.0, 13.0, 14.0, 15.0));
}

TEST(Vec, can_be_self_substracted)
{
    xad::Vec<double, 5> v1 = {1, 2, 3, 4, 5};
    xad::Vec<double, 5> v2 = {1, 2, 3, 4, 5};
    v1 -= v2;
    EXPECT_THAT(v1 == 0, true);
}

TEST(Vec, can_be_substracted)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v -= 10.0;
    EXPECT_THAT(v, ElementsAre(-9.0, -8.0, -7.0, -6.0, -5.0));
}

TEST(Vec, can_be_left_substracted)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = v - 10.0;
    EXPECT_THAT(v, ElementsAre(-9.0, -8.0, -7.0, -6.0, -5.0));
}

TEST(Vec, can_be_right_substracted)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = 10.0 - v;
    EXPECT_THAT(v, ElementsAre(9.0, 8.0, 7.0, 6.0, 5.0));
}

TEST(Vec, can_be_self_multiplied)
{
    xad::Vec<double, 5> v1 = {1, 2, 3, 4, 5};
    xad::Vec<double, 5> v2 = {1, 2, 3, 4, 5};
    v1 *= v2;
    EXPECT_THAT(v1, ElementsAre(1, 4, 9, 16, 25));
}

TEST(Vec, can_be_multiplied)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v *= 10.0;
    EXPECT_THAT(v, ElementsAre(10.0, 20.0, 30.0, 40.0, 50.0));
}

TEST(Vec, can_be_left_multiplied)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = v * 10.0;
    EXPECT_THAT(v, ElementsAre(10.0, 20.0, 30.0, 40.0, 50.0));
}

TEST(Vec, can_be_right_multiplied)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = 10.0 * v;
    EXPECT_THAT(v, ElementsAre(10.0, 20.0, 30.0, 40.0, 50.0));
}

TEST(Vec, can_be_self_divided)
{
    xad::Vec<double, 5> v1 = {10, 20, 30, 40, 50};
    xad::Vec<double, 5> v2 = {1, 2, 3, 4, 5};
    v1 /= v2;
    EXPECT_THAT(v1 == 10, true);
}

TEST(Vec, can_be_divided)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v /= 10.0;
    EXPECT_THAT(v, ElementsAre(0.1, 0.2, 0.3, 0.4, 0.5));
}

TEST(Vec, can_be_left_divided)
{
    xad::Vec<double, 5> v = {1, 2, 3, 4, 5};
    v = v / 10.0;
    EXPECT_THAT(v, ElementsAre(0.1, 0.2, 0.3, 0.4, 0.5));
}

TEST(Vec, can_be_right_divided)
{
    xad::Vec<double, 4> v = {1, 2, 4, 5};
    v = 10.0 / v;
    EXPECT_THAT(v, ElementsAre(10.0, 5.0, 2.5, 2.0));
}

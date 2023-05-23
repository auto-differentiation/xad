/*******************************************************************************

   Unit tests for stream operations on literals.

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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace ::testing;

template <class T>
class Streams : public ::testing::Test
{
};

typedef ::testing::Types<xad::AReal<double>, xad::FReal<double>, xad::AReal<float>,
                         xad::FReal<float>, xad::AReal<xad::AReal<double>>,
                         xad::AReal<xad::FReal<double>>, xad::FReal<xad::AReal<double>>,
                         xad::FReal<xad::FReal<double>>>
    test_types;

TYPED_TEST_SUITE(Streams, test_types);

TYPED_TEST(Streams, CanWriteToOstream)
{
    std::stringstream sstr;
    TypeParam in = 1.25;
    sstr << in;
    EXPECT_THAT(sstr.str(), Eq("1.25"));
}

TYPED_TEST(Streams, CanWriteExpressionToOstream)
{
    std::stringstream sstr;
    TypeParam in = 1.25;
    sstr << (in * 1.0);
    EXPECT_THAT(sstr.str(), Eq("1.25"));
}

TYPED_TEST(Streams, CanReadFromIstream)
{
    std::stringstream sstr;
    TypeParam out;
    sstr << "1.25";
    sstr >> out;
    EXPECT_THAT(double(xad::value(xad::value(out))), DoubleNear(1.25, 1e-9));
}

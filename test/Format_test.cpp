/*******************************************************************************

   Unit tests for std::format support.

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
#include <XAD/StdCompatibility.hpp>
#include <gtest/gtest.h>
#include <limits>

#if (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || \
    (!defined(_MSVC_LANG) && __cplusplus >= 202002L)
#if defined(__has_include) && __has_include(<format>)
#include <format>
#endif
#endif

#if defined(__cpp_lib_format)
TEST(StdFormat, ARealMatchesDouble)
{
    xad::AReal<double> value = 1.25;

    EXPECT_EQ(std::format("{}", value), std::format("{}", 1.25));
    EXPECT_EQ(std::format("{:.4f}", value), std::format("{:.4f}", 1.25));
    EXPECT_EQ(std::format("{:e}", value), std::format("{:e}", 1.25));
    EXPECT_EQ(std::format("{:10.2f}", value), std::format("{:10.2f}", 1.25));
    EXPECT_EQ(std::format("{:*<10.2f}", value), std::format("{:*<10.2f}", 1.25));
}

TEST(StdFormat, FRealMatchesDouble)
{
    xad::FReal<double> value = 1.25;

    EXPECT_EQ(std::format("{:.4f}", value), std::format("{:.4f}", 1.25));
    EXPECT_EQ(std::format("{:e}", value), std::format("{:e}", 1.25));
    EXPECT_EQ(std::format("{:10.2f}", value), std::format("{:10.2f}", 1.25));
    EXPECT_EQ(std::format("{:*<10.2f}", value), std::format("{:*<10.2f}", 1.25));
}

TEST(StdFormat, NestedARealFRealMatchesInnermostDouble)
{
    xad::AReal<xad::FReal<double>> value = 1.25;

    EXPECT_EQ(std::format("{:.4f}", value), std::format("{:.4f}", 1.25));
    EXPECT_EQ(std::format("{:e}", value), std::format("{:e}", 1.25));
    EXPECT_EQ(std::format("{:10.2f}", value), std::format("{:10.2f}", 1.25));
    EXPECT_EQ(std::format("{:*<10.2f}", value), std::format("{:*<10.2f}", 1.25));
}

TEST(StdFormat, NestedFRealARealMatchesInnermostDouble)
{
    xad::FReal<xad::AReal<double>> value = 1.25;

    EXPECT_EQ(std::format("{:.4f}", value), std::format("{:.4f}", 1.25));
}

TEST(StdFormat, FloatUnderlyingScalar)
{
    xad::AReal<float> active = 1.25f;
    xad::FReal<float> forward = 1.25f;

    EXPECT_EQ(std::format("{}", active), std::format("{}", 1.25f));
    EXPECT_EQ(std::format("{:.4f}", forward), std::format("{:.4f}", 1.25f));
}

TEST(StdFormat, SpecialFloatingPointValuesMatchScalars)
{
    const double positiveInfinity = std::numeric_limits<double>::infinity();
    const double negativeInfinity = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    xad::AReal<double> activePositiveInfinity = positiveInfinity;
    xad::AReal<double> activeNegativeInfinity = negativeInfinity;
    xad::AReal<double> activeNaN = nan;

    EXPECT_EQ(std::format("{}", activePositiveInfinity),
              std::format("{}", positiveInfinity));
    EXPECT_EQ(std::format("{}", activeNegativeInfinity),
              std::format("{}", negativeInfinity));
    EXPECT_EQ(std::format("{}", activeNaN), std::format("{}", nan));
}

TEST(StdFormat, StdCompatibilityArithmeticTraitsDoNotInterfere)
{
    xad::AReal<double> active = 1.25;
    xad::FReal<double> forward = 1.25;

    static_assert(std::is_arithmetic<decltype(active)>::value,
                  "StdCompatibility should make AReal arithmetic");
    static_assert(std::is_arithmetic<decltype(forward)>::value,
                  "StdCompatibility should make FReal arithmetic");
    EXPECT_EQ(std::format("{}", active), std::format("{}", 1.25));
    EXPECT_EQ(std::format("{}", forward), std::format("{}", 1.25));
}
#endif

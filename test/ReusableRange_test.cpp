/*******************************************************************************

   Test for ReusableRange class, used in Tape to keep track of slots that can be
   re-used.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include <XAD/ReusableRange.hpp>

using namespace ::testing;

TEST(ReusableRange, DefaultIsClosed)
{
    xad::ReusableRange<unsigned> r;

    EXPECT_THAT(r.isClosed(), true);
    EXPECT_THAT(r.size(), 0u);
}

TEST(ReusableRange, SizeIsOk)
{
    xad::ReusableRange<unsigned> r(5, 7);

    EXPECT_THAT(r.isClosed(), false);
    EXPECT_THAT(r.size(), 2U);
}

TEST(ReusableRange, ComparingByStart)
{
    xad::ReusableRange<unsigned> r(5, 7), r2(4, 5), r3(8, 20);

    EXPECT_THAT(r2, Lt(r));
    EXPECT_THAT(r2, Lt(r3));
    EXPECT_THAT(r, Lt(r3));
}

TEST(ReusableRange, CanBeSorted)
{
    xad::ReusableRange<unsigned> r(5, 7), r2(4, 5), r3(8, 20);
    std::vector<xad::ReusableRange<unsigned>> v = {r, r2, r3};
    std::sort(v.begin(), v.end());

    EXPECT_THAT(v, ElementsAre(r2, r, r3));
}

TEST(ResuableRange, IsInRange)
{
    xad::ReusableRange<unsigned> r(5, 7);

    EXPECT_THAT(r.isInRange(3), false);
    EXPECT_THAT(r.isInRange(5), true);
    EXPECT_THAT(r.isInRange(6), true);
    EXPECT_THAT(r.isInRange(7), false);
}

TEST(ReusableRange, CanInsert)
{
    xad::ReusableRange<unsigned> r(5, 7);
    auto s = r.insert();

    EXPECT_THAT(s, 5u);
    EXPECT_THAT(r.first(), 6u);
    EXPECT_THAT(r.isClosed(), false);
}

TEST(ReusableRange, InsertionClosesRange)
{
    xad::ReusableRange<unsigned> r(6, 7);
    auto s = r.insert();

    EXPECT_THAT(s, 6u);
    EXPECT_THAT(r.isClosed(), true);
}

TEST(ReusableRange, ExpandSuccessEnd)
{
    xad::ReusableRange<unsigned> r(5, 7);

    auto ret = r.expand(7);
    EXPECT_THAT(ret, xad::ReusableRange<unsigned>::END);
    EXPECT_THAT(r.size(), 3u);
    EXPECT_THAT(r.second(), 8u);
}

TEST(ReusableRange, ExpandSuccessStart)
{
    xad::ReusableRange<unsigned> r(5, 7);

    auto ret = r.expand(4);
    EXPECT_THAT(ret, xad::ReusableRange<unsigned>::START);
    EXPECT_THAT(r.size(), 3u);
    EXPECT_THAT(r.first(), 4u);
    EXPECT_THAT(r.second(), 7u);
}

TEST(ReusableRange, ExpandFail)
{
    xad::ReusableRange<unsigned> r(5, 7);

    EXPECT_THAT(r.expand(2), xad::ReusableRange<unsigned>::FAILED);
    EXPECT_THAT(r.expand(6), xad::ReusableRange<unsigned>::FAILED);
    EXPECT_THAT(r.expand(9), xad::ReusableRange<unsigned>::FAILED);

    EXPECT_THAT(r.size(), 2u);
    EXPECT_THAT(r.first(), 5u);
    EXPECT_THAT(r.second(), 7u);
}

TEST(ResuableRange, JoinEnd)
{
    xad::ReusableRange<unsigned> r(5, 7), r2(7, 12);

    EXPECT_THAT(r.isJoinableEnd(r2), true);
    EXPECT_THAT(r.isJoinableStart(r2), false);
    EXPECT_THAT(r.isJoinable(r2), xad::ReusableRange<unsigned>::END);
    auto out = r.joinEnd(r2);
    EXPECT_THAT(out, Eq(r));
    EXPECT_THAT(out.size(), 7u);
    EXPECT_THAT(out.first(), 5u);
    EXPECT_THAT(out.second(), 12u);
}

TEST(ResuableRange, JoinStart)
{
    xad::ReusableRange<unsigned> r(5, 7), r2(7, 12);

    EXPECT_THAT(r2.isJoinableStart(r), true);
    EXPECT_THAT(r2.isJoinableEnd(r), false);
    EXPECT_THAT(r2.isJoinable(r), xad::ReusableRange<unsigned>::START);
    auto out = r2.joinStart(r);
    EXPECT_THAT(out.size(), 7u);
    EXPECT_THAT(out, Eq(r2));
    EXPECT_THAT(out.first(), 5u);
    EXPECT_THAT(out.second(), 12u);
}

TEST(ReusableRange, NotJoinable)
{
    xad::ReusableRange<unsigned> r(5, 7), r2(9, 12);

    EXPECT_THAT(r.isJoinable(r2), xad::ReusableRange<unsigned>::FAILED);
}

TEST(ReusableRange, OutputToStream)
{
    xad::ReusableRange<unsigned> r(5, 7);
    std::stringstream sstr;
    sstr << r;
    EXPECT_THAT(sstr.str(), Eq("[5, 7)"));
}

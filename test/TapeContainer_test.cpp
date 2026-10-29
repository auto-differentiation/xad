/*******************************************************************************

   General unit tests for the tape container (whatever it is typedef-ed to).

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

#include "XAD/TapeContainer.hpp"

#include <gmock/gmock.h>

using namespace testing;

TEST(TapeContainer, statements)
{
    // currently, it's a redefine of ChunkContainer, which we have a test for already
    typedef typename xad::TapeContainerTraits<int, int>::statements_type container;

    container sc;

    EXPECT_EQ(0U, sc.size());
    EXPECT_TRUE(sc.empty());

    // we only require push_backs to be successful after reserving
    sc.reserve(2);

    sc.emplace_back(2, 0);
    sc.emplace_back(3, 1);
    EXPECT_THAT(sc.size(), Eq(2));
    EXPECT_THAT(sc.empty(), IsFalse());
    EXPECT_THAT(sc[0], Pair(2, 0));
    EXPECT_THAT(sc[1], Pair(3, 1));
}

TEST(TapeContainer, operations)
{
    // currently, it's a redefine of ChunkContainer, which we have a test for already
    typedef typename xad::TapeContainerTraits<int, int>::operations_type container;

    container sc;

    EXPECT_EQ(0U, sc.size());
    EXPECT_TRUE(sc.empty());

    // we only require push_backs to be successful after reserving
    sc.reserve(2);

    sc.push_back(2, 0);
    sc.push_back(3, 1);
    EXPECT_THAT(sc.size(), Eq(2));
    EXPECT_THAT(sc.empty(), IsFalse());
    EXPECT_THAT(sc[0], Pair(2, 0));
    EXPECT_THAT(sc[1], Pair(3, 1));
}

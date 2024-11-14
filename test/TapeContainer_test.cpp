/*******************************************************************************

   General unit tests for the tape container (whatever it is typedef-ed to).

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

#include "XAD/TapeContainer.hpp"

#include <gtest/gtest.h>

TEST(TapeContainer, basic)
{
    // currently, it's a redefine of ChunkContainer, which we have a test for already
    typedef xad::TapeContainerTraits<int>::type container;

    container sc;

    EXPECT_EQ(0U, sc.size());
    EXPECT_TRUE(sc.empty());

    // we only require push_backs to be successful after reserving
    sc.reserve(2);

    sc.push_back_reserved(2);
    sc.push_back_reserved(3);
    EXPECT_EQ(2U, sc.size());
    EXPECT_FALSE(sc.empty());
    EXPECT_EQ(2, sc[0]);
    EXPECT_EQ(3, sc[1]);
}

/*******************************************************************************

   Unit tests for gaplist container in tape.

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

#include <XAD/TapeGapList.hpp>

#include <gtest/gtest.h>

TEST(TapeGapList, basic)
{
    typedef xad::TapeGapListTraits<int>::type type;

    type l;

    EXPECT_EQ(0U, l.size());

    l.push_back(10);
    l.push_back(11);
    l.push_front(3);
    l.push_front(2);
    l.push_front(1);

    EXPECT_EQ(5U, l.size());
    EXPECT_EQ(1, l.front());
    l.pop_front();
    EXPECT_EQ(2, l.front());
    EXPECT_EQ(11, l.back());
    l.pop_back();
    EXPECT_EQ(10, l.back());
}

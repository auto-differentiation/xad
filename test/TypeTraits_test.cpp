/*******************************************************************************

   Tests for helper traits.

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

#include <XAD/TypeTraits.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <list>

TEST(TypeTraits, HasBeginVectorOfVectorsIterator)
{
    using It = std::vector<std::vector<int>>::iterator;
    EXPECT_TRUE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginVectorOfVectors)
{
    using It = std::vector<std::vector<int>>;
    EXPECT_FALSE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginListIterator)
{
    using It = std::list<int>::iterator;
    EXPECT_TRUE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginList)
{
    using It = std::list<int>;
    EXPECT_FALSE(xad::detail::has_begin<It>::value);
}

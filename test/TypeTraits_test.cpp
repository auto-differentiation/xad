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
#include <XAD/XAD.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <list>

TEST(TypeTraits, HasBeginVectorOfVectorsIteratorDereferenced)
{
    std::vector<std::vector<int>> t(0);
    auto row = t.begin();
    EXPECT_TRUE(xad::detail::has_begin<decltype(*row)>::value);
}

TEST(TypeTraits, HasBeginVectorOfVectors)
{
    using It = std::vector<std::vector<int>>;
    EXPECT_TRUE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginListIteratorReferenced)
{
    using It = std::list<int>::iterator;
    EXPECT_FALSE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginList)
{
    using It = std::list<int>;
    EXPECT_TRUE(xad::detail::has_begin<It>::value);
}

TEST(TypeTraits, HasBeginNonIterable)
{
    using It = size_t;
    EXPECT_FALSE(xad::detail::has_begin<It>::value);
}
TEST(Traits, test_ARealtypes )
{
    static_assert(std::is_same<xad::DerivativesTraits<xad::AReal<double, 2>, 1>::type,
                               xad::AReal<double, 2>>::value,
                  "expected trades");
    static_assert(std::is_same<xad::Tape<xad::AReal<double, 2>, 1>::derivative_type,
                               xad::AReal<double, 2>>::value,
                  "expected type mismatch");
}

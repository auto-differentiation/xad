/*******************************************************************************

   Tests for the operations container.

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

#include <XAD/OperationsContainer.hpp>
#include <XAD/OperationsContainerPaired.hpp>
#include <gmock/gmock.h>

using namespace testing;

template <typename C>
class OperationsContainerTest : public Test
{
};

using test_containers1 = ::testing::Types<xad::OperationsContainer<double, int, 4>,
                                          xad::OperationsContainerPaired<double, int, 4>>;

TYPED_TEST_SUITE(OperationsContainerTest, test_containers1);

TYPED_TEST(OperationsContainerTest, isEmptyAtStart)
{
    auto c = TypeParam();

    EXPECT_THAT(c.empty(), IsTrue());
    EXPECT_THAT(c.size(), Eq(0));
}

TYPED_TEST(OperationsContainerTest, canReserveCapacity)
{
    auto c = TypeParam();

    c.reserve(50);
    EXPECT_THAT(c.capacity(), Ge(50));
    EXPECT_THAT(c.chunks(), Eq((50 + TypeParam::chunk_size - 1) / TypeParam::chunk_size));
}

TYPED_TEST(OperationsContainerTest, canAppendElementsAndAccess)
{
    auto c = TypeParam();

    auto m = {1.0, 2.0, 3.0};
    auto s = {3, 4, 5};
    c.append_n(m.begin(), s.begin(), 3);

    EXPECT_THAT(c.size(), Eq(3));
    EXPECT_THAT(c.empty(), IsFalse());
    EXPECT_THAT(c[0], Pair(1.0, 3));
    EXPECT_THAT(c[1], Pair(2.0, 4));
    EXPECT_THAT(c[2], Pair(3.0, 5));
}

TYPED_TEST(OperationsContainerTest, canAppendElementsMultiChunk)
{
    auto c = TypeParam();

    auto m = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    auto s = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    c.append_n(m.begin(), s.begin(), 9);
    c.append_n(m.begin(), s.begin(), 9);

    EXPECT_THAT(c.size(), Eq(18));
    EXPECT_THAT(c.empty(), IsFalse());
    for (unsigned i = 0; i < 18; ++i)
    {
        EXPECT_THAT(c[i], Pair(static_cast<double>(i % 9 + 1), i % 9 + 1));
    }
}

TYPED_TEST(OperationsContainerTest, canPushBack)
{
    auto c = TypeParam();
    for (int i = 0; i < 10; ++i)
    {
        c.push_back(static_cast<double>(i), i);
    }

    for (unsigned i = 0; i < 10; ++i)
    {
        EXPECT_THAT(c[i], Pair(static_cast<double>(i), i));
    }
}

TYPED_TEST(OperationsContainerTest, canPushBackUnsafe)
{
    auto c = TypeParam();
    c.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        c.push_back_unsafe(static_cast<double>(i), i);
    }

    for (unsigned i = 0; i < 10; ++i)
    {
        EXPECT_THAT(c[i], Pair(static_cast<double>(i), i));
    }
}

TYPED_TEST(OperationsContainerTest, canResizeExtendingSize)
{
    auto c = TypeParam();
    c.push_back(42.0, 123);
    c.resize(8);

    EXPECT_THAT(c.size(), Eq(8));
    EXPECT_THAT(c[0], Pair(42.0, 123));
    for (unsigned i = 1; i < 8; ++i)
    {
        EXPECT_THAT(c[i], Pair(0.0, 0)) << "for i=" << i;
    }
}

TYPED_TEST(OperationsContainerTest, canResizeShrinkingSize)
{
    auto c = TypeParam();
    for (int i = 0; i < 10; ++i)
    {
        c.push_back(static_cast<double>(i), i);
    }

    c.resize(5);

    EXPECT_THAT(c.size(), Eq(5));
    for (unsigned i = 0; i < 5; ++i)
    {
        EXPECT_THAT(c[i], Pair(static_cast<double>(i), i));
    }
}

TYPED_TEST(OperationsContainerTest, canClear)
{
    auto c = TypeParam();
    c.push_back(42.0, 123);
    c.push_back(42.0, 123);
    c.clear();

    EXPECT_THAT(c.size(), Eq(0));
    EXPECT_THAT(c.empty(), IsTrue());
}

namespace
{
struct TestStruct
{
    TestStruct() { ++items; }
    TestStruct(const TestStruct&) { ++items; }
    TestStruct(TestStruct&&) { ++items; }
    TestStruct& operator=(TestStruct&&)
    {
        ++items;
        return *this;
    }
    TestStruct& operator=(const TestStruct&)
    {
        ++items;
        return *this;
    }
    ~TestStruct() { --items; }

    static int items;
};

int TestStruct::items = 0;

}  // namespace

template <typename C>
class OperationsContainerTest2 : public Test
{
};

using test_containers2 = ::testing::Types<xad::OperationsContainer<TestStruct, int, 4>,
                                          xad::OperationsContainerPaired<TestStruct, int, 4>>;
TYPED_TEST_SUITE(OperationsContainerTest2, test_containers2);

TYPED_TEST(OperationsContainerTest2, callsDestructOnDisposal)
{
    TestStruct::items = 0;
    {
        auto c = TypeParam();
        c.push_back(TestStruct(), 1);
        c.push_back(TestStruct(), 2);
        EXPECT_THAT(c[1], Pair(_, 2));
        EXPECT_THAT(TestStruct::items, Eq(2));
    }
    EXPECT_THAT(TestStruct::items, Eq(0));
}

TYPED_TEST(OperationsContainerTest2, callsDestructOnResize)
{
    TestStruct::items = 0;
    auto c = TypeParam();
    c.push_back(TestStruct(), 1);
    c.push_back(TestStruct(), 1);
    c.resize(1);
    EXPECT_THAT(TestStruct::items, Eq(1));
}

TYPED_TEST(OperationsContainerTest2, callsDestructOnClear)
{
    TestStruct::items = 0;
    auto c = TypeParam();
    c.push_back(TestStruct(), 1);
    c.push_back(TestStruct(), 1);
    c.clear();
    EXPECT_THAT(TestStruct::items, Eq(0));
}

TYPED_TEST(OperationsContainerTest2, callsConstructOnResize)
{
    TestStruct::items = 0;
    auto c = TypeParam();
    c.resize(3);

    EXPECT_THAT(TestStruct::items, Eq(3));
}

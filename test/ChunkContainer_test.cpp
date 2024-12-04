/*******************************************************************************

   Tests for the ChunkContainer, used by the tape.

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

#include <XAD/ChunkContainer.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;
using xad::ChunkContainer;

TEST(ChunkContainer, alloc_less_than_alignment)
{
    void* p1 = ::xad::detail::aligned_alloc(128, 32);

    EXPECT_THAT(p1, NotNull());
}

TEST(ChunkContainer, too_large_throws_bad_alloc)
{
    auto construct_huge = [&]
    {
        auto c = ChunkContainer<int, static_cast<std::size_t>(-1)>();
        XAD_UNUSED_VARIABLE(c);
    };
    EXPECT_THAT(construct_huge, Throws<std::bad_alloc>());
}

TEST(ChunkContainer, iterator)
{
    ChunkContainer<int> chk;
    for (int i = 0; i < 10; ++i) chk.push_back(i);

    auto it = chk.iterator_at(5);
    auto itend = chk.iterator_at(10);
    int i = 5;
    while (it != itend)
    {
        EXPECT_EQ(i, *it);
        ++it;
        ++i;
    }
}

TEST(ChunkContainer, iterator_over_end)
{
    using container = ChunkContainer<int, 16>;
    container chk;
    for (int i = 0; i < int(container::chunk_size + 5); ++i) chk.push_back(i);

    auto it = chk.iterator_at(container::chunk_size - 4);
    auto itend = chk.iterator_at(container::chunk_size + 5);
    int i = int(container::chunk_size - 4);
    while (it != itend)
    {
        EXPECT_EQ(i, *it);
        ++it;
        ++i;
    }
}

TEST(ChunkContainer, uninitialized_extend)
{
    using container = ChunkContainer<int, 16>;
    container chk;
    std::size_t i = 0;
    for (; i < container::chunk_size - 4; ++i) chk.push_back(int(i));
    chk.uninitialized_extend(10);
    EXPECT_EQ(container::chunk_size - 4 + 10, chk.size());
    auto it = chk.iterator_at(i);
    for (std::size_t j = i + 10; i < j; ++i)
    {
        ::new (&*it++) int(static_cast<int>(i));
    }

    for (std::size_t j = 0; j < container::chunk_size - 4 + 10; ++j) EXPECT_EQ(int(j), chk[j]);
}

#if defined(__GNUC__) && !defined(__clang__)
// we're only comparing pointer addresses in the tests below to verify move
// behaviour, but GCC 12 sees this as use-after-free and flags warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuse-after-free"
#endif

TEST(ChunkContainer, move_construct)
{
    ChunkContainer<int> chk;
    chk.push_back(123);

    auto addr = &chk[0];

    ChunkContainer<int> chk2(std::move(chk));

    EXPECT_THAT(chk2[0], Eq(123));
    EXPECT_THAT(addr, Eq(&chk2[0]));
}

TEST(ChunkContainer, move_assign)
{
    ChunkContainer<int> chk;
    chk.push_back(123);

    auto addr = &chk[0];

    ChunkContainer<int> chk2;
    chk2.push_back(42);
    auto addr2 = &chk2[0];
    chk2 = std::move(chk);

    EXPECT_THAT(chk2[0], Eq(123));
    EXPECT_THAT(addr, Eq(&chk2[0]));
    EXPECT_THAT(addr, Ne(addr2));
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

TEST(ChunkContainer, multichunk)
{
    ChunkContainer<int, 8> chk;
    for (int i = 0; i < 20; ++i) chk.push_back(i);

    for (int i = 0; i < 20; ++i) EXPECT_THAT(chk[size_t(i)], Eq(i)) << "at " << i;
}

namespace
{
struct NonPodTester
{
    NonPodTester() { ++constructions; }
    NonPodTester(const NonPodTester&) { ++copies; }
    NonPodTester(NonPodTester&&) = delete;

    ~NonPodTester() { ++destructions; }

    static void reset()
    {
        constructions = 0;
        destructions = 0;
        copies = 0;
    }

    static int constructions;
    static int copies;
    static int destructions;
};

int NonPodTester::constructions = 0;
int NonPodTester::destructions = 0;
int NonPodTester::copies = 0;
}  // namespace

TEST(ChunkContainer, non_pod_type)
{
    {
        NonPodTester::reset();
        ChunkContainer<NonPodTester, 8> chk;
        EXPECT_THAT(NonPodTester::constructions, Eq(0));
        for (int i = 0; i < 20; ++i) chk.push_back({});
        EXPECT_THAT(NonPodTester::constructions, Eq(20));
        EXPECT_THAT(NonPodTester::copies, Eq(20));
    }
    EXPECT_THAT(NonPodTester::copies + NonPodTester::constructions, Eq(NonPodTester::destructions));
}

TEST(ChunkContainer, non_pod_type_single_chunk_destruct)
{
    {
        NonPodTester::reset();
        ChunkContainer<NonPodTester, 8> chk;
        for (int i = 0; i < 4; ++i) chk.push_back({});
    }
    EXPECT_THAT(NonPodTester::copies + NonPodTester::constructions, Eq(NonPodTester::destructions));
}

TEST(ChunkContainer, non_pod_type_full_chunk_destruct)
{
    {
        NonPodTester::reset();
        ChunkContainer<NonPodTester, 8> chk;
        for (int i = 0; i < 8; ++i) chk.push_back({});
    }
    EXPECT_THAT(NonPodTester::copies + NonPodTester::constructions, Eq(NonPodTester::destructions));
}

TEST(ChunkContainer, resize_with_same_size_has_no_effect)
{
    ChunkContainer<int, 8> chk;
    chk.push_back(0);
    chk.push_back(1);

    auto cap_before = chk.capacity();
    chk.resize(2);
    EXPECT_THAT(chk.size(), Eq(2u));
    EXPECT_THAT(chk[0], Eq(0));
    EXPECT_THAT(chk[1], Eq(1));
    EXPECT_THAT(chk.capacity(), Eq(cap_before));
}

TEST(ChunkContainer, resize)
{
    ChunkContainer<int, 8> chk;
    for (int i = 0; i < 10; ++i) chk.push_back(i);

    EXPECT_THAT(chk.size(), Eq(10u));

    chk.resize(15);
    EXPECT_THAT(chk.size(), Eq(15u));
    chk[12] = 12;

    for (int i = 0; i < 10; ++i) EXPECT_THAT(chk[size_t(i)], Eq(i));
    for (int i = 10; i < 15; ++i)
    {
        if (i == 12)
            EXPECT_THAT(chk[size_t(i)], Eq(12));
        else
            EXPECT_THAT(chk[size_t(i)], Eq(0));
    }
}

TEST(ChunkContainer, clear_method)
{
    ChunkContainer<int, 8> chk;
    
    // populate the ct
    for (int i = 0; i < 20; ++i)
    {   
       chk.push_back(i);
    }
    // verify init state
    EXPECT_THAT(chk.size(), Eq(20u));
    
    // clear the ct
    chk.clear();
    
    // check size is reseted 
    EXPECT_THAT(chk.size(), Eq(0u));
    
    // check if we can still add elements after clear
    for (int i = 0; i < 10; ++i) 
    {
       chk.push_back(i);
    }
    EXPECT_THAT(chk.size(), Eq(10u));
    for (int i = 0; i < 10; ++i) 
    {
        EXPECT_THAT(chk[size_t(i)], Eq(i)) << "at " << i;
    }
}

TEST(ChunkContainer, resize_fills_with_values)
{
    ChunkContainer<int, 8> chk;
    chk.push_back(42);
    chk.resize(12, 10);

    EXPECT_THAT(chk.size(), Eq(12u));
    for (unsigned i = 1; i < 12; ++i) EXPECT_THAT(chk[i], Eq(10));
}

TEST(ChunkContainer, resize_fills_with_values_full_chunk)
{
    ChunkContainer<int, 8> chk;
    chk.push_back(42);
    chk.resize(8, 10);

    EXPECT_THAT(chk.size(), Eq(8u));
    for (unsigned i = 1; i < 8; ++i) EXPECT_THAT(chk[i], Eq(10));
}

TEST(ChunkContainer, append)
{
    ChunkContainer<int, 8> chk;
    for (int i = 0; i < 14; ++i) chk.push_back(i);

    std::vector<int> newvals = {14, 15, 16, 17};
    // note: we can only append sizes less than chunk size (8)
    chk.append(newvals.begin(), newvals.end());

    EXPECT_THAT(chk.size(), Eq(18u));

    for (int i = 0; i < 18; ++i) EXPECT_THAT(chk[size_t(i)], Eq(i)) << "at " << i;
}

TEST(ChunkContainer, append_without_new_chunk)
{
    ChunkContainer<int, 8> chk;
    for (int i = 0; i < 2; ++i) chk.push_back(i);

    std::vector<int> newvals = {2, 3, 4, 5};
    // note: we can only append sizes less than chunk size (8)
    chk.append(newvals.begin(), newvals.end());

    EXPECT_THAT(chk.size(), Eq(6U));

    for (int i = 0; i < 6; ++i) EXPECT_THAT(chk[size_t(i)], Eq(i)) << "at " << i;
}

TEST(ChunkContainer, push_back_no_check)
{
    ChunkContainer<int, 8> chk;
    // note: push_back_no_check expects space to be reserved beforehand
    chk.reserve(17);
    for (int i = 0; i < 17; ++i) chk.push_back_no_check(i);

    EXPECT_THAT(chk.size(), Eq(17u));

    for (int i = 0; i < 17; ++i) EXPECT_THAT(chk[size_t(i)], Eq(i)) << "at " << i;
}

TEST(ChunkContainer, emplace_back)
{
    ChunkContainer<std::pair<int, int>, 8> chk;
    for (int i = 0; i < 17; ++i) chk.emplace_back(i, i);

    EXPECT_THAT(chk.size(), Eq(17u));

    for (int i = 0; i < 17; ++i)
    {
        EXPECT_THAT(chk[size_t(i)], Pair(i, i)) << "at " << i;
    }
}

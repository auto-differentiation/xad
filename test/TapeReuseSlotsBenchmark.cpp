/*******************************************************************************

   Benchmark for reusing tape slots freed out of order (XAD_TAPE_REUSE_SLOTS)

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, version 3 of the License only.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include <XAD/Tape.hpp>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
// set this to true to run the benchmark
constexpr bool RUN_BENCHMARKS = false;
}  // namespace

TEST(TapeReuseSlotsBenchmark, outOfOrderFrees)
{
    if (!RUN_BENCHMARKS)
        GTEST_SKIP() << "Skipping benchmark by default";

    // a fixed pseudo-random order to free the variables in
    const std::size_t n = 20000;
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::uint64_t state = 1;
    for (std::size_t i = n - 1; i > 0; --i)
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        std::swap(order[i], order[std::size_t((state >> 33) % (i + 1))]);
    }

    xad::Tape<double> tape;
    std::vector<xad::Tape<double>::slot_type> slots(n);
    for (int round = 0; round < 20; ++round)
    {
        for (auto& s : slots) s = tape.registerVariable();
        for (auto i : order) tape.unregisterVariable(slots[i]);
    }
    EXPECT_EQ(0U, tape.getNumVariables());
}

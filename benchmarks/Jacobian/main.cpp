/*******************************************************************************

   Jacobian benchmarks.

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

#include <benchmark/benchmark.h>

#include "jacobian.hpp"


static void JacobianAdj(benchmark::State &state)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    std::function<std::vector<AD>(std::vector<AD>&)> foo = [](std::vector<AD>& x) -> std::vector<AD>
    { return {sin(x[0] + x[1]), sin(x[1] + x[2]), cos(x[2] + x[3]), cos(x[3] + x[0])}; };

    for (auto _ : state)
    {
        auto jacobian = computeJacobian(x_ad, foo);
    }
}

static void JacobianFwd(benchmark::State &state)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    std::function<std::vector<AD>(std::vector<AD>&)> foo = [](std::vector<AD>& x) -> std::vector<AD>
    { return {sin(x[0] + x[1]), sin(x[1] + x[2]), cos(x[2] + x[3]), cos(x[3] + x[0])}; };

    for (auto _ : state)
    {
        auto jacobian = computeJacobian(x_ad, foo);
    }
}

BENCHMARK(JacobianAdj);
BENCHMARK(JacobianFwd);
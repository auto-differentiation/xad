/*******************************************************************************

   Math benchmark for XAD. Tests the performance of unary, binary and
   ternary functions in both forward and adjoint modes.

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

#include <XAD/XAD.hpp>

#include "math.hpp"
#include "../util.hpp"

static void MathUnaryAdj(benchmark::State &state)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    AD x = 0.1112;

    for (auto _ : state)
    {
        for (auto &func : make_unary_functions<AD>())
        {
            tape.registerInput(x);
            tape.newRecording();
            AD y = func(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
        }
    }
} BENCHMARK(MathUnaryAdj);

static void MathUnaryFwd(benchmark::State &state)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    AD x = 0.1112;

    for (auto _ : state)
    {
        for (auto &func : make_unary_functions<AD>())
        {
            derivative(x) = 1.0;
            AD y = func(x);
            XAD_UNUSED_VARIABLE(derivative(y));
            derivative(x) = 0.0;
        }
    }
} BENCHMARK(MathUnaryFwd);
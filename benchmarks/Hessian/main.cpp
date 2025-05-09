/*******************************************************************************

   Hessian benchmarks.

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

#include "hessian.hpp"
#include "../util.hpp"

static void HessianFwdAdj(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_foo<AD>(), & tape);
    }
} BENCHMARK(HessianFwdAdj);


static void HessianFwdFwd(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});


    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_foo<AD>());
    }
} BENCHMARK(HessianFwdFwd);

static void HessianFwdAdjAckley(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_ackley<AD>(), & tape);
    }
} BENCHMARK(HessianFwdAdjAckley);

static void HessianFwdFwdAckley(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});


    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_ackley<AD>());
    }
} BENCHMARK(HessianFwdFwdAckley);

static void HessianFwdAdjNeuralLoss(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_neuralLoss<AD>(), & tape);
    }
}

static void HessianFwdFwdNeuralLoss(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});


    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_neuralLoss<AD>());
    }
} BENCHMARK(HessianFwdFwdNeuralLoss);

static void HessianFwdAdjSparse(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2, 91.13, 9.92, 1.3, 1.2, 0.14, 125.0, 1.5, 1.3, 1.2, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_sparse<AD>(), & tape);
    }
} BENCHMARK(HessianFwdAdjSparse);

static void HessianFwdFwdSparse(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2, 91.13, 9.92, 1.3, 1.2, 0.14, 125.0, 1.5, 1.3, 1.2, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_sparse<AD>());
    }
} BENCHMARK(HessianFwdFwdSparse);

static void HessianFwdAdjDense(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2, 91.13, 9.92, 1.3, 1.2, 0.14, 125.0, 1.5, 1.3, 1.2, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_dense<AD>(), & tape);
    }
} BENCHMARK(HessianFwdAdjDense);

static void HessianFwdFwdDense(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2, 91.13, 9.92, 1.3, 1.2, 0.14, 125.0, 1.5, 1.3, 1.2, 1.5, 1.3, 1.2});

    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, make_dense<AD>());
    }
} BENCHMARK(HessianFwdFwdDense);
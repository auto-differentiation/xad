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
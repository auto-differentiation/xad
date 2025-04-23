#include <benchmark/benchmark.h>

#include <XAD/XAD.hpp>
#include <XAD/Hessian.hpp>

static void HessianFwdAdj(benchmark::State &state)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    std::function<AD(std::vector<AD> &)> foo = [](std::vector<AD> &x) -> AD
    { return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0]); };


    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, foo);
    }
}

static void HessianFwdFwd(benchmark::State &state)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});

    std::function<AD(std::vector<AD> &)> foo = [](std::vector<AD> &x) -> AD
    { return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0]); };


    for (auto _ : state)
    {
        auto hessian = computeHessian(x_ad, foo);
    }
}

BENCHMARK(HessianFwdAdj);
BENCHMARK(HessianFwdFwd);
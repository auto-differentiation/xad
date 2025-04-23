#include <benchmark/benchmark.h>

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>


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
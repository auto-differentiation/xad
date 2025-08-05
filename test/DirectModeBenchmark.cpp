#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "COSHestonEngineExpr.hpp"

namespace
{
// set this to false to run the benchmarks for direct mode vs expression template mode
constexpr bool RUN_BENCHMARKS = false;
}  // namespace

TEST(RealDirectBenchmark, ARealBenchmark)
{
    if (!RUN_BENCHMARKS)
        GTEST_SKIP() << "Skipping benchmark by default";
    xad::Tape<double> tp;
    for (int i = 0; i < 100000; i++)
    {
        TestHeston<xad::AD> tester;
        xad::AD x = 0.8;
        tp.registerInput(x);
        tp.newRecording();
        xad::AD y = tester.c4(x);
        tp.registerOutput(y);
        derivative(y) = 1.0;
        tp.computeAdjoints();
        derivative(x);
        tp.clearAll();
    }
}

TEST(RealDirectBenchmark, ARealDirectBenchmark)
{
    if (!RUN_BENCHMARKS)
        GTEST_SKIP() << "Skipping benchmark by default";
    xad::Tape<double> tp;
    TestHeston<xad::ARealDirect<double>> tester;
    xad::ARealDirect<double> x = 0.8;
    xad::ARealDirect<double> y;
    for (int i = 0; i < 100000; i++)
    {
        tp.registerInput(x.base());
        tp.newRecording();
        y = tester.c4(x);
        tp.registerOutput(y.base());
        derivative(y) = 1.0;
        tp.computeAdjoints();
        derivative(x);
        tp.clearAll();
    }
}

TEST(RealDirectBenchmark, FRealBenchmark)
{
    if (!RUN_BENCHMARKS)
        GTEST_SKIP() << "Skipping benchmark by default";
    TestHeston<xad::FReal<double>> tester;
    xad::FReal<double> x = 0.8;
    xad::FReal<double> y;
    for (int i = 0; i < 100000; i++)
    {
        y = tester.c4(x);
    }
}

TEST(RealDirectBenchmark, FRealDirectBenchmark)
{
    if (!RUN_BENCHMARKS)
        GTEST_SKIP() << "Skipping benchmark by default";
    TestHeston<xad::ARealDirect<double>> tester;
    xad::ARealDirect<double> x = 0.8;
    xad::ARealDirect<double> y;
    for (int i = 0; i < 100000; i++)
    {
        y = tester.c4(x);
    }
}

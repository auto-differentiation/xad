/*******************************************************************************

   LIBOR Swaption Portfolio Pricer - AAD Benchmark

   This benchmark compares different approaches for computing sensitivities
   in Monte Carlo pricing of a LIBOR swaption portfolio.

   APPROACHES TESTED:
     FD      - Finite Differences (bump-and-revalue)
     XAD     - XAD tape-based reverse-mode AAD
     JIT     - Forge JIT-compiled native code (scalar)
     JIT-AVX - Forge JIT + AVX2 SIMD (4 paths/instruction)

   BUILD CONFIGURATIONS:
     Baseline (XAD_FORGE_ENABLED=OFF): FD, XAD
     Full     (XAD_FORGE_ENABLED=ON):  FD, XAD, JIT, JIT-AVX

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   The code is an adapted version of Prof. Mike Giles, available here:
   https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

#include "LiborData.hpp"
#include "LiborPricers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

SwaptionPortfolio setupTestPortfolio()
{
    SwaptionPortfolio p;
    // maturities of the swaptions in years
    p.maturities = {4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40};
    p.swaprates = {.045, .05,  .055, .045, .05,  .055, .045, .05,
                   .055, .045, .05,  .055, .045, .05,  .055};
    return p;
}

MarketParameters setupTestMarket()
{
    MarketParameters market;
    market.delta = 0.05;
    market.L0.assign(80, 0.05);
    market.lambda.assign(80, 0.2);
    return market;
}

bool checkError(double ad_value, double fd_value, const std::string& what)
{
    if (std::abs(ad_value - fd_value) / (std::abs(fd_value) + 1e-6) > 1e-4)
    {
        std::cerr << std::fixed << std::setprecision(10);
        std::cerr << what << ": AD " << ad_value << " does not match FD " << fd_value << "\n";
        return true;
    }
    return false;
}

/// Compute mean of a vector
double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/// Compute standard deviation of a vector
double stddev(const std::vector<double>& v)
{
    if (v.size() <= 1) return 0.0;
    double m = mean(v);
    double sq_sum = 0.0;
    for (double x : v)
    {
        double diff = x - m;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
}

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h     Show this help message\n";
    std::cout << "  --validate     Run validation against finite differences\n";
    std::cout << "  --quick        Run quick benchmark (fewer iterations)\n";
    std::cout << "\nThis benchmark compares AD approaches for LIBOR swaption pricing.\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "Build: Full benchmark (Forge JIT enabled)\n";
#else
    std::cout << "Build: Baseline (Forge JIT disabled)\n";
#endif
}

int main(int argc, char** argv)
{
    // Parse options
    bool doValidation = false;
    bool quickMode = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--validate")
            doValidation = true;
        else if (arg == "--quick")
            quickMode = true;
    }

    constexpr unsigned long long SEED = 91672912;

    SwaptionPortfolio portfolio = setupTestPortfolio();
    MarketParameters market = setupTestMarket();

    // Benchmark configuration
    std::vector<int> pathCounts = {10, 100, 1000, 10000};
    std::vector<std::string> pathLabels = {"10", "100", "1K", "10K"};

    size_t warmupIterations = quickMode ? 1 : 3;
    size_t benchmarkIterations = quickMode ? 3 : 10;

    // Results storage
    struct TimingResult
    {
        double fd_mean = 0, fd_std = 0;
        double xad_mean = 0, xad_std = 0;
#ifdef XAD_FORGE_ENABLED
        double jit_mean = 0, jit_std = 0;
        double jit_avx_mean = 0, jit_avx_std = 0;
#endif
    };
    std::vector<TimingResult> results(pathCounts.size());

    // Print header
    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "  LIBOR Swaption Portfolio Pricer - AAD Benchmark\n";
    std::cout << "=============================================================================\n";
    std::cout << "\n";
    std::cout << "  INSTRUMENT:\n";
    std::cout << "    Portfolio of " << portfolio.maturities.size() << " European swaptions\n";
    std::cout << "    Maturities: 4, 8, 20, 28, 40 years (3 each)\n";
    std::cout << "    Model: LIBOR Market Model (lognormal forwards)\n";
    std::cout << "\n";
    std::cout << "  MARKET INPUTS:\n";
    std::cout << "    delta:  1 parameter\n";
    std::cout << "    lambda: " << market.lambda.size() << " volatility parameters\n";
    std::cout << "    L0:     " << market.L0.size() << " initial forward rates\n";
    std::cout << "    Total:  " << (1 + market.lambda.size() + market.L0.size()) << " sensitivities\n";
    std::cout << "\n";
    std::cout << "  APPROACHES TESTED:\n";
    std::cout << "    FD      - Finite Differences (bump-and-revalue)\n";
    std::cout << "    XAD     - XAD tape-based reverse-mode AAD\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "    JIT     - Forge JIT-compiled native code\n";
    std::cout << "    JIT-AVX - Forge JIT + AVX2 SIMD (4 paths/instruction)\n";
    std::cout << "\n";
    std::cout << "  BUILD: Full benchmark (Forge JIT enabled)\n";
#else
    std::cout << "\n";
    std::cout << "  BUILD: Baseline (Forge JIT disabled)\n";
#endif
    std::cout << "\n";
    std::cout << "  CONFIGURATION:\n";
    std::cout << "    Path counts:     10, 100, 1K, 10K\n";
    std::cout << "    Warmup/Bench:    " << warmupIterations << "/" << benchmarkIterations
              << " iterations\n";
    std::cout << "\n";

    // Run benchmarks for each path count
    for (size_t tc = 0; tc < pathCounts.size(); ++tc)
    {
        int numPaths = pathCounts[tc];
        std::cout << "  Running " << pathLabels[tc] << " paths..." << std::flush;

        std::vector<double> fd_times, xad_times;
#ifdef XAD_FORGE_ENABLED
        std::vector<double> jit_times, jit_avx_times;
#endif

        for (size_t iter = 0; iter < warmupIterations + benchmarkIterations; ++iter)
        {
            bool recordTiming = (iter >= warmupIterations);

            // ---------------------------------------------------------------------
            // Finite Differences
            // ---------------------------------------------------------------------
            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioFD(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming)
                    fd_times.push_back(elapsed.count());
            }

            // ---------------------------------------------------------------------
            // XAD Tape
            // ---------------------------------------------------------------------
            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioAD(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming)
                    xad_times.push_back(elapsed.count());
            }

#ifdef XAD_FORGE_ENABLED
            // ---------------------------------------------------------------------
            // JIT (Scalar)
            // ---------------------------------------------------------------------
            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioJIT(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming)
                    jit_times.push_back(elapsed.count());
            }

            // ---------------------------------------------------------------------
            // JIT-AVX
            // ---------------------------------------------------------------------
            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioJIT_AVX(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming)
                    jit_avx_times.push_back(elapsed.count());
            }
#endif
        }

        // Store results
        results[tc].fd_mean = mean(fd_times);
        results[tc].fd_std = stddev(fd_times);
        results[tc].xad_mean = mean(xad_times);
        results[tc].xad_std = stddev(xad_times);
#ifdef XAD_FORGE_ENABLED
        results[tc].jit_mean = mean(jit_times);
        results[tc].jit_std = stddev(jit_times);
        results[tc].jit_avx_mean = mean(jit_avx_times);
        results[tc].jit_avx_std = stddev(jit_avx_times);
#endif

        std::cout << " Done." << std::endl;
    }

    // =========================================================================
    // Print results table
    // =========================================================================
    std::cout << "\n";
    std::cout << "  " << std::string(79, '=') << "\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "  RESULTS: LIBOR Swaption Benchmark (times in ms)\n";
#else
    std::cout << "  RESULTS: LIBOR Swaption Benchmark - BASELINE (times in ms)\n";
#endif
    std::cout << "  " << std::string(79, '=') << "\n";
    std::cout << "\n";

#ifdef XAD_FORGE_ENABLED
    // Full benchmark table
    std::cout << "   Paths |       FD |      XAD |      JIT |  JIT-AVX | XAD/JIT | XAD/AVX\n";
    std::cout << "  -------+----------+----------+----------+----------+---------+---------\n";

    for (size_t tc = 0; tc < pathCounts.size(); ++tc)
    {
        double speedup_jit = results[tc].xad_mean / results[tc].jit_mean;
        double speedup_avx = results[tc].xad_mean / results[tc].jit_avx_mean;

        std::cout << "  " << std::setw(6) << pathCounts[tc] << " |"
                  << std::fixed << std::setprecision(1) << std::setw(9) << results[tc].fd_mean
                  << " |" << std::setw(9) << results[tc].xad_mean << " |" << std::setw(9)
                  << results[tc].jit_mean << " |" << std::setw(9) << results[tc].jit_avx_mean
                  << " |" << std::setprecision(2) << std::setw(7) << speedup_jit << "x |"
                  << std::setw(7) << speedup_avx << "x\n";
    }
#else
    // Baseline table (FD and XAD only)
    std::cout << "   Paths |       FD |      XAD | FD/XAD\n";
    std::cout << "  -------+----------+----------+--------\n";

    for (size_t tc = 0; tc < pathCounts.size(); ++tc)
    {
        double speedup = results[tc].fd_mean / results[tc].xad_mean;

        std::cout << "  " << std::setw(6) << pathCounts[tc] << " |"
                  << std::fixed << std::setprecision(1) << std::setw(9) << results[tc].fd_mean
                  << " |" << std::setw(9) << results[tc].xad_mean << " |" << std::setprecision(2)
                  << std::setw(6) << speedup << "x\n";
    }
#endif

    std::cout << "\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "  Speedup = XAD time / JIT time\n";
#else
    std::cout << "  FD/XAD = ratio of Finite Differences to XAD tape\n";
#endif
    std::cout << "\n";

    // =========================================================================
    // Optional validation
    // =========================================================================
    if (doValidation)
    {
        std::cout << "  " << std::string(79, '-') << "\n";
        std::cout << "  VALIDATION (comparing against Finite Differences)\n";
        std::cout << "  " << std::string(79, '-') << "\n";

        int numPaths = 10000;
        bool hasError = false;

        auto resFD = pricePortfolioFD(portfolio, market, numPaths, SEED);
        auto resAD = pricePortfolioAD(portfolio, market, numPaths, SEED);

        // Check XAD against FD
        hasError = checkError(resAD.price, resFD.price, "XAD price") || hasError;
        hasError = checkError(resAD.d_delta, resFD.d_delta, "XAD d_delta") || hasError;
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            hasError =
                checkError(resAD.d_lambda[i], resFD.d_lambda[i], "XAD lambda[" + std::to_string(i) + "]") ||
                hasError;
        }
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            hasError = checkError(resAD.d_L0[i], resFD.d_L0[i], "XAD L0[" + std::to_string(i) + "]") ||
                       hasError;
        }

        if (!hasError)
            std::cout << "  XAD validation: PASSED\n";

#ifdef XAD_FORGE_ENABLED
        hasError = false;
        auto resJIT = pricePortfolioJIT(portfolio, market, numPaths, SEED);

        hasError = checkError(resJIT.price, resAD.price, "JIT price") || hasError;
        hasError = checkError(resJIT.d_delta, resAD.d_delta, "JIT d_delta") || hasError;
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            hasError =
                checkError(resJIT.d_lambda[i], resAD.d_lambda[i], "JIT lambda[" + std::to_string(i) + "]") ||
                hasError;
        }
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            hasError = checkError(resJIT.d_L0[i], resAD.d_L0[i], "JIT L0[" + std::to_string(i) + "]") ||
                       hasError;
        }

        if (!hasError)
            std::cout << "  JIT validation: PASSED\n";

        hasError = false;
        auto resAVX = pricePortfolioJIT_AVX(portfolio, market, numPaths, SEED);

        hasError = checkError(resAVX.price, resAD.price, "JIT-AVX price") || hasError;
        hasError = checkError(resAVX.d_delta, resAD.d_delta, "JIT-AVX d_delta") || hasError;
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            hasError = checkError(resAVX.d_lambda[i], resAD.d_lambda[i],
                                  "JIT-AVX lambda[" + std::to_string(i) + "]") ||
                       hasError;
        }
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            hasError =
                checkError(resAVX.d_L0[i], resAD.d_L0[i], "JIT-AVX L0[" + std::to_string(i) + "]") ||
                hasError;
        }

        if (!hasError)
            std::cout << "  JIT-AVX validation: PASSED\n";
#endif

        std::cout << "\n";
    }

    std::cout << "  Benchmark complete.\n\n";
    return 0;
}

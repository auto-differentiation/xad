/*******************************************************************************

   1st order adjoint mode for a Monte-Carlo LIBOR Swaption
   portfolio pricer.

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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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
    if (std::abs(ad_value - fd_value) / (fd_value + 1e-6) > 1e-4)
    {
        std::cerr << std::fixed << std::setprecision(10);
        std::cerr << what << ": AD " << ad_value << " does not match FD " << fd_value << "\n";
        return true;
    }
    return false;
}

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [numPaths] [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  test    Run finite difference validation\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "  jit     Run JIT-compiled pricing (requires xad-forge)\n";
#endif
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " 10000           # 10K paths, AAD only\n";
    std::cout << "  " << progName << " 10000 test      # 10K paths with FD validation\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "  " << progName << " 10000 jit       # 10K paths, compare AAD vs JIT\n";
    std::cout << "  " << progName << " 10000 jit test  # 10K paths, JIT + FD validation\n";
#endif
}

/// Runs pricing given number of paths and optionally tests against finite differences
/// for correctness.
///
/// Usage:
///   LiborSwaptionPricer [numPaths] [options]
///
/// Options:
///   test - Run finite difference validation
///   jit  - Run JIT-compiled pricing (requires xad-forge)
///
/// By default, it uses 10,000 paths and does not run tests against finite differences.
///
/// For example, for running 100,000 paths and compare to finite differences:
///
/// LiborSwaptionPricer 100000 test
///
int main(int argc, char** argv)
{
    const int NUM_PATHS = argc < 2 ? 10000 : std::atol(argv[1]);

    // Parse options
    bool doTests = false;
    bool doJIT = false;
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "test")
            doTests = true;
        else if (arg == "jit")
            doJIT = true;
        else if (arg == "help" || arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
    }

#ifndef XAD_FORGE_ENABLED
    if (doJIT)
    {
        std::cerr << "Error: JIT mode requires XAD_FORGE_ENABLED.\n";
        std::cerr << "Rebuild with -DXAD_LIBOR_ENABLE_FORGE=ON and xad-forge available.\n";
        return 1;
    }
#endif

    constexpr unsigned long long SEED = 91672912;

    SwaptionPortfolio p = setupTestPortfolio();
    MarketParameters market = setupTestMarket();

    std::cout << "=============================================================================\n";
    std::cout << "  LIBOR Swaption Portfolio Pricer - AAD Benchmark\n";
    std::cout << "=============================================================================\n";
    std::cout << "\n";
    std::cout << "  Configuration:\n";
    std::cout << "    Paths:        " << NUM_PATHS << "\n";
    std::cout << "    Swaptions:    " << p.maturities.size() << "\n";
    std::cout << "    Inputs:       " << (1 + market.lambda.size() + market.L0.size())
              << " (1 delta + " << market.lambda.size() << " lambda + "
              << market.L0.size() << " L0)\n";
#ifdef XAD_FORGE_ENABLED
    std::cout << "    Forge JIT:    Available\n";
#else
    std::cout << "    Forge JIT:    Not available (XAD_FORGE_ENABLED not set)\n";
#endif
    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(8);

    // -------------------------------------------------------------------------
    // Pure pricing (no sensitivities)
    // -------------------------------------------------------------------------
    std::cout << "-------- Pure pricing ---------------------\n";
    auto start{std::chrono::steady_clock::now()};
    auto resPlain = pricePortfolio(p, market, NUM_PATHS, SEED);
    auto end{std::chrono::steady_clock::now()};
    std::cout << "Portfolio price = " << resPlain.price << "\n";
    std::chrono::duration<double> elapsed_plain{end - start};

    // -------------------------------------------------------------------------
    // AAD pricing (tape-based)
    // -------------------------------------------------------------------------
    std::cout << "-------- AAD pricing (tape) ---------------\n";
    start = std::chrono::steady_clock::now();
    auto resAD = pricePortfolioAD(p, market, NUM_PATHS, SEED);
    end = std::chrono::steady_clock::now();
    std::cout << "Portfolio price         = " << resAD.price << "\n";
    std::cout << "Derivative w.r.t. delta = " << resAD.d_delta << "\n";
    std::cout << "(Showing first 3 of " << market.lambda.size() << " lambda derivatives)\n";
    for (size_t i = 0; i < 3 && i < market.lambda.size(); ++i)
    {
        std::cout << "Derivative w.r.t. lambda[" << i << "] = " << resAD.d_lambda[i] << "\n";
    }
    std::cout << "(Showing first 3 of " << market.L0.size() << " L0 derivatives)\n";
    for (size_t i = 0; i < 3 && i < market.L0.size(); ++i)
    {
        std::cout << "Derivative w.r.t. L0[" << i << "] = " << resAD.d_L0[i] << "\n";
    }
    std::chrono::duration<double> elapsed_ad{end - start};

    bool hasError = checkError(resAD.price, resPlain.price, "price");
    if (hasError)
    {
        return 1;
    }

#ifdef XAD_FORGE_ENABLED
    // -------------------------------------------------------------------------
    // JIT pricing (Forge backend)
    // -------------------------------------------------------------------------
    std::chrono::duration<double> elapsed_jit{0};
    JITStats jitStats;
    Results resJIT;

    if (doJIT)
    {
        std::cout << "-------- JIT pricing (Forge) --------------\n";
        start = std::chrono::steady_clock::now();
        resJIT = pricePortfolioJIT(p, market, NUM_PATHS, SEED, &jitStats);
        end = std::chrono::steady_clock::now();
        std::cout << "Portfolio price         = " << resJIT.price << "\n";
        std::cout << "Derivative w.r.t. delta = " << resJIT.d_delta << "\n";
        std::cout << "(Showing first 3 of " << market.lambda.size() << " lambda derivatives)\n";
        for (size_t i = 0; i < 3 && i < market.lambda.size(); ++i)
        {
            std::cout << "Derivative w.r.t. lambda[" << i << "] = " << resJIT.d_lambda[i] << "\n";
        }
        elapsed_jit = end - start;

        // Validate JIT results against AAD
        hasError = checkError(resJIT.price, resAD.price, "JIT price") || hasError;
        hasError = checkError(resJIT.d_delta, resAD.d_delta, "JIT d_delta") || hasError;
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            hasError = checkError(resJIT.d_lambda[i], resAD.d_lambda[i],
                                  "JIT lambda[" + std::to_string(i) + "]") ||
                       hasError;
        }
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            hasError = checkError(resJIT.d_L0[i], resAD.d_L0[i],
                                  "JIT L0[" + std::to_string(i) + "]") ||
                       hasError;
        }
    }
#endif

    // -------------------------------------------------------------------------
    // Timing summary
    // -------------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n";
    std::cout << "=============================================================================\n";
    std::cout << "  TIMING SUMMARY\n";
    std::cout << "=============================================================================\n";
    std::cout << "----- Plain: " << std::setw(10) << elapsed_plain.count() << " seconds\n";
    std::cout << "----- AAD  : " << std::setw(10) << elapsed_ad.count() << " seconds";
    std::cout << "  (slowdown vs plain: " << std::setprecision(1)
              << (elapsed_ad.count() / elapsed_plain.count()) << "x)\n";

#ifdef XAD_FORGE_ENABLED
    if (doJIT)
    {
        std::cout << std::setprecision(3);
        std::cout << "----- JIT  : " << std::setw(10) << elapsed_jit.count() << " seconds";
        std::cout << "  (speedup vs AAD: " << std::setprecision(1)
                  << (elapsed_ad.count() / elapsed_jit.count()) << "x)\n";
        std::cout << std::setprecision(3);
        std::cout << "      Compile time: " << jitStats.compileTimeMs << " ms\n";
    }
#endif

    // -------------------------------------------------------------------------
    // Finite difference validation (optional)
    // -------------------------------------------------------------------------
    if (doTests)
    {
        std::cout << "\n-------- Finite Difference validation -----\n";
        start = std::chrono::steady_clock::now();
        auto resFD = pricePortfolioFD(p, market, NUM_PATHS, SEED);
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_fd{end - start};
        std::cout << "----- FD   : " << std::setw(10) << elapsed_fd.count() << " seconds\n";

        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            hasError = checkError(resAD.d_lambda[i], resFD.d_lambda[i],
                                  "lambda[" + std::to_string(i) + "]") ||
                       hasError;
        }
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            hasError = checkError(resAD.d_L0[i], resFD.d_L0[i], "L0[" + std::to_string(i) + "]") ||
                       hasError;
        }
    }

    if (hasError)
    {
        std::cerr << "\nThere were errors.\n";
        return 1;
    }

    std::cout << "\nAll validations passed.\n";
    return 0;
}

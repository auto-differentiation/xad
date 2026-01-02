/*******************************************************************************

   Sample main file for 1st order adjoint mode for a Monte-Carlo LIBOR
   swaption portfolio pricer.

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
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

SwaptionPortfolio setupTestPortfolio()
{
    SwaptionPortfolio p;
    p.maturities = {4, 4, 4, 8, 8, 8, 20, 20, 20, 28, 28, 28, 40, 40, 40};
    p.swaprates = {.045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055, .045, .05, .055};
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

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h     Show this help message\n";
    std::cout << "  --quick        Run quick test (fewer iterations)\n";
    std::cout << "  --validate     Validate AAD results against finite differences\n";
}

int main(int argc, char** argv)
{
    // Parse options
    bool quickMode = false;
    bool validateMode = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--quick")
            quickMode = true;
        else if (arg == "--validate")
            validateMode = true;
    }

    constexpr unsigned long long SEED = 91672912;

    SwaptionPortfolio portfolio = setupTestPortfolio();
    MarketParameters market = setupTestMarket();

    int numPaths = quickMode ? 1000 : 10000;

    std::cout << "\n";
    std::cout << "LIBOR Swaption Portfolio Pricer\n";
    std::cout << "================================\n";
    std::cout << "Portfolio:    " << portfolio.maturities.size() << " European swaptions\n";
    std::cout << "Paths:        " << numPaths << "\n";
    std::cout << "Inputs:       " << (1 + market.lambda.size() + market.L0.size()) << " sensitivities\n";
    std::cout << "\n";

    // Run pricing without sensitivities
    std::cout << "Pure pricing (no sensitivities)...\n";
    auto start = std::chrono::steady_clock::now();
    auto resPlain = pricePortfolio(portfolio, market, numPaths, SEED);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "  Price: " << std::fixed << std::setprecision(6) << resPlain.price << "\n";
    std::cout << "  Time:  " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "\n";

    // Run AAD pricing
    std::cout << "AAD pricing (with sensitivities)...\n";
    start = std::chrono::steady_clock::now();
    auto resAD = pricePortfolioAD(portfolio, market, numPaths, SEED);
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    std::cout << "  Price: " << std::fixed << std::setprecision(6) << resAD.price << "\n";
    std::cout << "  Time:  " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "  d/d(delta):   " << std::scientific << std::setprecision(6) << resAD.d_delta << "\n";
    std::cout << "  d/d(L0[0]):   " << resAD.d_L0[0] << "\n";
    std::cout << "  d/d(lambda[0]): " << resAD.d_lambda[0] << "\n";
    std::cout << "\n";

    if (validateMode)
    {
        std::cout << "Validating against finite differences...\n";
        auto resFD = pricePortfolioFD(portfolio, market, numPaths, SEED);

        double priceDiff = std::abs(resAD.price - resFD.price);
        double deltaDiff = std::abs(resAD.d_delta - resFD.d_delta);

        std::cout << "  Price diff:       " << std::scientific << priceDiff << "\n";
        std::cout << "  d_delta diff:     " << deltaDiff << "\n";

        // Check a sample of derivatives
        int matchCount = 0;
        int totalChecked = 0;
        double tol = 1e-4;

        // Check d_delta
        if (std::abs(resAD.d_delta - resFD.d_delta) / (std::abs(resFD.d_delta) + 1e-10) < tol)
            matchCount++;
        totalChecked++;

        // Check d_lambda
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            if (std::abs(resAD.d_lambda[i] - resFD.d_lambda[i]) /
                    (std::abs(resFD.d_lambda[i]) + 1e-10) <
                tol)
                matchCount++;
            totalChecked++;
        }

        // Check d_L0
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            if (std::abs(resAD.d_L0[i] - resFD.d_L0[i]) / (std::abs(resFD.d_L0[i]) + 1e-10) < tol)
                matchCount++;
            totalChecked++;
        }

        std::cout << "  Derivatives OK:   " << matchCount << "/" << totalChecked << "\n";
        std::cout << "\n";

        if (matchCount == totalChecked)
        {
            std::cout << "VALIDATION PASSED\n";
        }
        else
        {
            std::cout << "VALIDATION FAILED\n";
            return 1;
        }
    }

    std::cout << "Done.\n\n";
    return 0;
}

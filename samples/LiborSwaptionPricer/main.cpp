/*******************************************************************************

   1st order adjoint mode for a Monte-Carlo LIBOR Swaption
   portfolio pricer.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   The code is an adapted version of Prof. Mike Giles, available here:
   https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

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

/// Runs pricing given number of paths and optionally tests against finite differences
/// for correctness.
///
/// Usage:
///   LiborSwaptionPricer [numPaths] [test]
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
    const bool doTests = argc < 3 ? false : argv[2] == std::string("test");
    constexpr unsigned long long SEED = 91672912;

    SwaptionPortfolio p = setupTestPortfolio();
    MarketParameters market = setupTestMarket();

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "-------- Pure pricing ---------------------\n";
    auto start{std::chrono::steady_clock::now()};
    auto resPlain = pricePortfolio(p, market, NUM_PATHS, SEED);
    auto end{std::chrono::steady_clock::now()};
    std::cout << "Portfolio price = " << resPlain.price << "\n";
    std::chrono::duration<double> elapsed_plain{end - start};

    std::cout << "-------- AD pricing -----------------------\n";
    start = std::chrono::steady_clock::now();
    auto resAD = pricePortfolioAD(p, market, NUM_PATHS, SEED);
    end = std::chrono::steady_clock::now();
    std::cout << "Portfolio price         = " << resAD.price << "\n";
    std::cout << "Derivative w.r.t. delta = " << resAD.d_delta << "\n";
    for (size_t i = 0; i < market.lambda.size(); ++i)
    {
        std::cout << "Derivative w.r.t. lambda[" << i << "] = " << resAD.d_lambda[i] << "\n";
    }
    for (size_t i = 0; i < market.L0.size(); ++i)
    {
        std::cout << "Derivative w.r.t. L0[" << i << "] = " << resAD.d_L0[i] << "\n";
    }
    std::chrono::duration<double> elapsed_ad{end - start};

    bool hasError = checkError(resAD.price, resPlain.price, "price");
    if (hasError)
    {
        return 1;
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n";
    std::cout << "----- Plain: " << std::setw(8) << elapsed_plain.count() << " seconds\n";
    std::cout << "----- AAD  : " << std::setw(8) << elapsed_ad.count() << " seconds\n";

    if (doTests)
    {
        start = std::chrono::steady_clock::now();
        auto resFD = pricePortfolioFD(p, market, NUM_PATHS, SEED);
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_fd{end - start};
        std::cout << "----- FD   : " << std::setw(8) << elapsed_fd.count() << " seconds\n";

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

        if (hasError)
        {
            std::cerr << "\nThere were errors.\n";
            return 1;
        }
        std::cout << "\nAll tests passed.\n";
    }

    return 0;
}

/*******************************************************************************

   Pricing functions for a portfolio of LIBOR swaptions.

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

#include <XAD/XAD.hpp>
#include "LiborData.hpp"
#include "LiborFunctions.hpp"
#include "LiborPricers.hpp"
#include <random>
#include <chrono>

#ifdef XAD_FORGE_ENABLED
#include <xad-forge/ForgeBackends.hpp>
#include <memory>
#include "LiborFunctionsJIT.hpp"
#endif

Results pricePortfolio(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                       int numPaths, unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);
    std::vector<double> samples(market.lambda.size() / 2);
    std::vector<double> L, tmp1, tmp2;

    Results res;
    res.price = 0.;
    for (int path = 0; path < numPaths; ++path)
    {
        // generate random samples
        std::generate(begin(samples), end(samples), [&]() { return dist(gen); });
        // generate the path, calculating LIBOR rates
        L.assign(begin(market.L0), end(market.L0));
        path_gen(market.delta, L, market.lambda, samples);
        // price portfolio on that path
        double v =
            value_portfolio(market.delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
        res.price += v;
    }
    res.price /= numPaths;

    return res;
}

// tape and active data type for 1st order adjoint computation
typedef xad::adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;

tape_type tape;

Results pricePortfolioAD(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                         int numPaths, unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);
    std::vector<double> samples(market.lambda.size() / 2);

    // AD types setup / copy input data
    std::vector<AD> L, tmp1, tmp2;
    std::vector<AD> lambda, L0;

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());
    for (int path = 0; path < numPaths; ++path)
    {
        // pathwise approach - make sure variables are erased
        tmp1.clear();
        tmp2.clear();
        L.clear();
        lambda.clear();
        L0.clear();
        tape.clearAll();

        // assign and register inputs
        AD delta = market.delta;
        lambda.assign(begin(market.lambda), end(market.lambda));
        L0.assign(begin(market.L0), end(market.L0));
        tape.registerInput(delta);
        tape.registerInputs(lambda);
        tape.registerInputs(L0);
        tape.newRecording();

        // generate random samples
        std::generate(begin(samples), end(samples), [&]() { return dist(gen); });
        // generate the path, calculating LIBOR rates
        L.assign(begin(L0), end(L0));
        path_gen(delta, L, lambda, samples);
        // price portfolio on that path
        AD v = value_portfolio(delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
        tape.registerOutput(v);
        derivative(v) = 1.0;
        tape.computeAdjoints();

        // update value and derivatives
        res.price += value(v);
        res.d_delta += derivative(delta);
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            res.d_lambda[i] += derivative(lambda[i]);
            res.d_L0[i] += derivative(L0[i]);
        }
    }

    // averaging
    res.price /= numPaths;
    res.d_delta /= numPaths;
    for (size_t i = 0; i < market.lambda.size(); ++i)
    {
        res.d_lambda[i] /= numPaths;
        res.d_L0[i] /= numPaths;
    }

    return res;
}

Results pricePortfolioFD(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                         int numPaths, unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);
    std::vector<double> samples(market.lambda.size() / 2);
    std::vector<double> L, tmp1, tmp2;

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());
    for (int path = 0; path < numPaths; ++path)
    {
        // generate random samples
        std::generate(begin(samples), end(samples), [&]() { return dist(gen); });

        // first get the value of this path

        // generate the path, calculating LIBOR rates
        L.assign(begin(market.L0), end(market.L0));
        path_gen(market.delta, L, market.lambda, samples);
        // price portfolio on that path
        double v =
            value_portfolio(market.delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
        res.price += v;

        // now bump inputs one by one to get all derivatives with finite differences
        double eps = 1e-5;
        {
            auto d = market.delta + eps;
            L.assign(begin(market.L0), end(market.L0));
            path_gen(d, L, market.lambda, samples);
            tmp1.clear();
            tmp2.clear();
            // price portfolio on that path
            double v1 =
                value_portfolio(d, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
            res.d_delta += (v1 - v) / eps;
        }

        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            L.assign(begin(market.L0), end(market.L0));
            tmp1.clear();
            tmp2.clear();
            L[i] += eps;
            path_gen(market.delta, L, market.lambda, samples);
            // price portfolio on that path
            double v1 = value_portfolio(market.delta, portfolio.maturities, portfolio.swaprates, L,
                                        tmp1, tmp2);
            res.d_L0[i] += (v1 - v) / eps;
        }
        auto lambda = market.lambda;
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            L.assign(begin(market.L0), end(market.L0));
            tmp1.clear();
            tmp2.clear();
            lambda[i] += eps;
            path_gen(market.delta, L, lambda, samples);
            // price portfolio on that path
            double v1 = value_portfolio(market.delta, portfolio.maturities, portfolio.swaprates, L,
                                        tmp1, tmp2);
            lambda[i] -= eps;
            res.d_lambda[i] += (v1 - v) / eps;
        }
    }
    res.price /= numPaths;
    res.d_delta /= numPaths;
    for (size_t i = 0; i < market.lambda.size(); ++i)
    {
        res.d_lambda[i] /= numPaths;
        res.d_L0[i] /= numPaths;
    }

    return res;
}

#ifdef XAD_FORGE_ENABLED

Results pricePortfolioJIT(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                          int numPaths, unsigned long long seed, JITStats* stats)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);

    const size_t numSamples = market.lambda.size() / 2;

    // Pre-generate all random samples for all paths
    // This allows us to set them as JIT inputs each path
    std::vector<std::vector<double>> allSamples(numPaths);
    for (int path = 0; path < numPaths; ++path)
    {
        allSamples[path].resize(numSamples);
        std::generate(begin(allSamples[path]), end(allSamples[path]),
                      [&]() { return dist(gen); });
    }

    // AD types for JIT - these persist across paths
    std::vector<AD> L, tmp1, tmp2;
    std::vector<AD> lambda, L0;
    std::vector<AD> jit_samples(numSamples);  // Random samples as AD inputs

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());

    // Create JIT compiler with Forge backend
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    AD delta;
    AD v;

    auto compileStart = std::chrono::steady_clock::now();

    // =========================================================================
    // Phase 1: Record and compile the computation graph (first path only)
    // =========================================================================

    // Assign and register market inputs
    delta = market.delta;
    lambda.assign(begin(market.lambda), end(market.lambda));
    L0.assign(begin(market.L0), end(market.L0));

    jit.registerInput(delta);
    jit.registerInputs(lambda);
    jit.registerInputs(L0);

    // Register random samples as JIT inputs - this is crucial!
    // Each path will update these values before forward pass
    for (size_t i = 0; i < numSamples; ++i)
    {
        jit_samples[i] = AD(allSamples[0][i]);  // Initialize with first path's samples
        jit.registerInput(jit_samples[i]);
    }

    jit.newRecording();

    // Generate the path using JIT-compatible path_gen with AD samples
    L.assign(begin(L0), end(L0));
    path_gen_jit(delta, L, lambda, jit_samples);

    // Price portfolio using JIT-compatible value_portfolio with ABool::If
    v = value_portfolio_jit(delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
    jit.registerOutput(v);

    // Compile the graph to native code
    jit.compile();

    auto compileEnd = std::chrono::steady_clock::now();
    if (stats)
    {
        stats->compileTimeMs =
            std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    }

    // =========================================================================
    // Phase 2: Execute compiled graph for all paths
    // =========================================================================

    for (int path = 0; path < numPaths; ++path)
    {
        // Set market input values (constant across paths, but needed for JIT)
        value(delta) = market.delta;
        for (size_t i = 0; i < market.lambda.size(); ++i)
            value(lambda[i]) = market.lambda[i];
        for (size_t i = 0; i < market.L0.size(); ++i)
            value(L0[i]) = market.L0[i];

        // Set random sample values for this path - this is what changes each path!
        for (size_t i = 0; i < numSamples; ++i)
            value(jit_samples[i]) = allSamples[path][i];

        // Forward pass - executes compiled native code
        double output;
        jit.forward(&output, 1);

        // Adjoint pass - computes all sensitivities
        jit.clearDerivatives();
        derivative(v) = 1.0;
        jit.computeAdjoints();

        // Accumulate results
        res.price += output;
        res.d_delta += derivative(delta);
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            res.d_lambda[i] += derivative(lambda[i]);
            res.d_L0[i] += derivative(L0[i]);
        }
    }

    // Averaging
    res.price /= numPaths;
    res.d_delta /= numPaths;
    for (size_t i = 0; i < market.lambda.size(); ++i)
    {
        res.d_lambda[i] /= numPaths;
        res.d_L0[i] /= numPaths;
    }

    return res;
}

#endif // XAD_FORGE_ENABLED

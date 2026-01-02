/*******************************************************************************

   JIT-accelerated pricing functions for a portfolio of LIBOR swaptions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   These functions use Forge JIT compilation for accelerated AAD computation.

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

#include "LiborPricersJIT.hpp"
#include "LiborFunctionsJIT.hpp"

#include <XAD/XAD.hpp>
#include <xad-forge/ForgeBackends.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>

#include <random>
#include <chrono>
#include <memory>
#include <cstring>

Results pricePortfolioJIT(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                          int numPaths, unsigned long long seed, JITStats* stats)
{
    // Use xad::AD directly (not bound to global tape) for JIT operations
    using JitAD = xad::AD;

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
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);  // Random samples as AD inputs

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());

    // Create JIT compiler with Forge backend
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    JitAD delta;
    JitAD v;

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
        jit_samples[i] = JitAD(allSamples[0][i]);  // Initialize with first path's samples
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

Results pricePortfolioJIT_AVX(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                              int numPaths, unsigned long long seed, JITStats* stats)
{
    // Use xad::AD directly (not bound to global tape) for JIT operations
    using JitAD = xad::AD;

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);

    const size_t numSamples = market.lambda.size() / 2;

    // Pre-generate all random samples for all paths
    std::vector<std::vector<double>> allSamples(numPaths);
    for (int path = 0; path < numPaths; ++path)
    {
        allSamples[path].resize(numSamples);
        std::generate(begin(allSamples[path]), end(allSamples[path]),
                      [&]() { return dist(gen); });
    }

    // AD types for JIT graph recording
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());

    // Create JIT compiler with Forge scalar backend for graph recording
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    JitAD delta;
    JitAD v;

    auto compileStart = std::chrono::steady_clock::now();

    // =========================================================================
    // Phase 1: Record the computation graph (using ScalarBackend)
    // =========================================================================

    delta = market.delta;
    lambda.assign(begin(market.lambda), end(market.lambda));
    L0.assign(begin(market.L0), end(market.L0));

    jit.registerInput(delta);
    jit.registerInputs(lambda);
    jit.registerInputs(L0);

    for (size_t i = 0; i < numSamples; ++i)
    {
        jit_samples[i] = JitAD(allSamples[0][i]);
        jit.registerInput(jit_samples[i]);
    }

    jit.newRecording();

    L.assign(begin(L0), end(L0));
    path_gen_jit(delta, L, lambda, jit_samples);
    v = value_portfolio_jit(delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
    jit.registerOutput(v);

    // Get the JIT graph and compile with AVX backend
    const auto& jitGraph = jit.getGraph();
    jit.deactivate();

    xad::forge::ForgeBackendAVX avxBackend(false);
    avxBackend.compile(jitGraph);

    auto compileEnd = std::chrono::steady_clock::now();
    if (stats)
    {
        stats->compileTimeMs =
            std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    }

    // =========================================================================
    // Phase 2: Execute compiled AVX kernel for all paths (4 at a time)
    // =========================================================================

    constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX::VECTOR_WIDTH;
    const int numBatches = (numPaths + BATCH_SIZE - 1) / BATCH_SIZE;

    // Total inputs: 1 (delta) + lambda.size() + L0.size() + numSamples
    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

    std::vector<double> inputBatch(BATCH_SIZE);
    std::vector<double> outputBatch(BATCH_SIZE);
    std::vector<double> adjointBatch(BATCH_SIZE, 1.0);
    std::vector<std::array<double, BATCH_SIZE>> inputGradients(numInputs);

    for (int batch = 0; batch < numBatches; ++batch)
    {
        int batchStart = batch * BATCH_SIZE;
        int actualBatchSize = std::min(BATCH_SIZE, numPaths - batchStart);

        // Set delta (same for all paths)
        for (int lane = 0; lane < BATCH_SIZE; ++lane)
            inputBatch[lane] = market.delta;
        avxBackend.setInputLanes(0, inputBatch.data());

        // Set lambda (same for all paths)
        for (size_t k = 0; k < market.lambda.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.lambda[k];
            avxBackend.setInputLanes(1 + k, inputBatch.data());
        }

        // Set L0 (same for all paths)
        for (size_t k = 0; k < market.L0.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.L0[k];
            avxBackend.setInputLanes(1 + market.lambda.size() + k, inputBatch.data());
        }

        // Set random samples (different for each path in batch)
        size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
        for (size_t m = 0; m < numSamples; ++m)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
            {
                int pathIdx = batchStart + lane;
                inputBatch[lane] = (pathIdx < numPaths) ? allSamples[pathIdx][m] : 0.0;
            }
            avxBackend.setInputLanes(sampleOffset + m, inputBatch.data());
        }

        // Execute forward + backward
        avxBackend.forwardAndBackward(adjointBatch.data(), outputBatch.data(), inputGradients);

        // Accumulate results
        for (int lane = 0; lane < actualBatchSize; ++lane)
        {
            res.price += outputBatch[lane];
            res.d_delta += inputGradients[0][lane];

            for (size_t k = 0; k < market.lambda.size(); ++k)
                res.d_lambda[k] += inputGradients[1 + k][lane];

            for (size_t k = 0; k < market.L0.size(); ++k)
                res.d_L0[k] += inputGradients[1 + market.lambda.size() + k][lane];
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

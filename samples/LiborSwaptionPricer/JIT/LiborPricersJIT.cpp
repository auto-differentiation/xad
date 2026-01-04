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
#include <xad-forge/ForgeBackend.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>

#include <random>
#include <chrono>
#include <memory>
#include <cstring>

// ============================================================================
// Performance Decomposition Functions
// ============================================================================

TimingDecomposition runDecompositionJIT(const SwaptionPortfolio& portfolio,
                                         const MarketParameters& market,
                                         int numPaths, unsigned long long seed)
{
    using JitAD = xad::AD;

    TimingDecomposition timing;
    timing.numPaths = numPaths;

    auto totalStart = std::chrono::steady_clock::now();

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);

    const size_t numSamples = market.lambda.size() / 2;

    // Pre-generate all random samples
    std::vector<std::vector<double>> allSamples(numPaths);
    for (int path = 0; path < numPaths; ++path)
    {
        allSamples[path].resize(numSamples);
        std::generate(begin(allSamples[path]), end(allSamples[path]),
                      [&]() { return dist(gen); });
    }

    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());

    xad::JITCompiler<double, 1> jit;

    JitAD delta;
    JitAD v;

    // --- Compilation Phase ---
    auto compileStart = std::chrono::steady_clock::now();

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

    // Compile with ForgeBackend
    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    auto compileEnd = std::chrono::steady_clock::now();
    timing.compileMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();

    // --- Execution Phase (per path) ---
    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

    double output;
    std::vector<double> inputGradients(numInputs);

    double setInputsTotal = 0.0;
    double forwardBackwardTotal = 0.0;
    double getGradientsTotal = 0.0;
    double accumulateTotal = 0.0;

    for (int path = 0; path < numPaths; ++path)
    {
        // Set inputs
        auto setStart = std::chrono::steady_clock::now();

        // delta
        double deltaVal = market.delta;
        backend.setInput(0, &deltaVal);

        // lambda
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            double lambdaVal = market.lambda[i];
            backend.setInput(1 + i, &lambdaVal);
        }

        // L0
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            double L0Val = market.L0[i];
            backend.setInput(1 + market.lambda.size() + i, &L0Val);
        }

        // samples
        size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
        for (size_t i = 0; i < numSamples; ++i)
        {
            double sampleVal = allSamples[path][i];
            backend.setInput(sampleOffset + i, &sampleVal);
        }

        auto setEnd = std::chrono::steady_clock::now();
        setInputsTotal += std::chrono::duration<double, std::milli>(setEnd - setStart).count();

        // Forward + backward (combined)
        auto fwdBwdStart = std::chrono::steady_clock::now();
        backend.forwardAndBackward(&output, inputGradients.data());
        auto fwdBwdEnd = std::chrono::steady_clock::now();
        forwardBackwardTotal += std::chrono::duration<double, std::milli>(fwdBwdEnd - fwdBwdStart).count();

        // Get gradients (already retrieved)
        auto getStart = std::chrono::steady_clock::now();
        auto getEnd = std::chrono::steady_clock::now();
        getGradientsTotal += std::chrono::duration<double, std::milli>(getEnd - getStart).count();

        // Accumulate
        auto accStart = std::chrono::steady_clock::now();
        res.price += output;
        res.d_delta += inputGradients[0];
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            res.d_lambda[i] += inputGradients[1 + i];
            res.d_L0[i] += inputGradients[1 + market.lambda.size() + i];
        }
        auto accEnd = std::chrono::steady_clock::now();
        accumulateTotal += std::chrono::duration<double, std::milli>(accEnd - accStart).count();
    }

    timing.setInputsMs = setInputsTotal;
    timing.forwardMs = forwardBackwardTotal;  // Forward+backward combined
    timing.backwardMs = 0.0;  // Included in forwardMs
    timing.getGradientsMs = getGradientsTotal;
    timing.accumulateMs = accumulateTotal;

    auto totalEnd = std::chrono::steady_clock::now();
    timing.totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timing;
}

TimingDecomposition runDecompositionJIT_AVX(const SwaptionPortfolio& portfolio,
                                             const MarketParameters& market,
                                             int numPaths, unsigned long long seed)
{
    using JitAD = xad::AD;

    TimingDecomposition timing;
    timing.numPaths = numPaths;

    auto totalStart = std::chrono::steady_clock::now();

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);

    const size_t numSamples = market.lambda.size() / 2;

    // Pre-generate all random samples
    std::vector<std::vector<double>> allSamples(numPaths);
    for (int path = 0; path < numPaths; ++path)
    {
        allSamples[path].resize(numSamples);
        std::generate(begin(allSamples[path]), end(allSamples[path]),
                      [&]() { return dist(gen); });
    }

    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);

    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());

    xad::JITCompiler<double, 1> jit;

    JitAD delta;
    JitAD v;

    // --- Compilation Phase ---
    auto compileStart = std::chrono::steady_clock::now();

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

    xad::forge::ForgeBackendAVX avxBackend(false);
    avxBackend.compile(jit.getGraph());

    auto compileEnd = std::chrono::steady_clock::now();
    timing.compileMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();

    // --- Execution Phase (batched) ---
    constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX::VECTOR_WIDTH;
    const int numBatches = (numPaths + BATCH_SIZE - 1) / BATCH_SIZE;
    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

    std::vector<double> inputBatch(BATCH_SIZE);
    std::vector<double> outputBatch(BATCH_SIZE);
    std::vector<double> inputGradients(numInputs * BATCH_SIZE);

    double setInputsTotal = 0.0;
    double forwardBackwardTotal = 0.0;  // AVX does forward+backward together
    double getGradientsTotal = 0.0;
    double accumulateTotal = 0.0;

    for (int batch = 0; batch < numBatches; ++batch)
    {
        int batchStart = batch * BATCH_SIZE;
        int actualBatchSize = std::min(BATCH_SIZE, numPaths - batchStart);

        // Set inputs
        auto setStart = std::chrono::steady_clock::now();

        for (int lane = 0; lane < BATCH_SIZE; ++lane)
            inputBatch[lane] = market.delta;
        avxBackend.setInput(0, inputBatch.data());

        for (size_t k = 0; k < market.lambda.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.lambda[k];
            avxBackend.setInput(1 + k, inputBatch.data());
        }

        for (size_t k = 0; k < market.L0.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.L0[k];
            avxBackend.setInput(1 + market.lambda.size() + k, inputBatch.data());
        }

        size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
        for (size_t m = 0; m < numSamples; ++m)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
            {
                int pathIdx = batchStart + lane;
                inputBatch[lane] = (pathIdx < numPaths) ? allSamples[pathIdx][m] : 0.0;
            }
            avxBackend.setInput(sampleOffset + m, inputBatch.data());
        }

        auto setEnd = std::chrono::steady_clock::now();
        setInputsTotal += std::chrono::duration<double, std::milli>(setEnd - setStart).count();

        // Forward + Backward (combined in AVX backend)
        auto fwdBwdStart = std::chrono::steady_clock::now();
        avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());
        auto fwdBwdEnd = std::chrono::steady_clock::now();
        forwardBackwardTotal += std::chrono::duration<double, std::milli>(fwdBwdEnd - fwdBwdStart).count();

        // Get gradients (already in inputGradients from forwardAndBackward)
        auto getStart = std::chrono::steady_clock::now();
        // Gradients are already retrieved in forwardAndBackward call
        auto getEnd = std::chrono::steady_clock::now();
        getGradientsTotal += std::chrono::duration<double, std::milli>(getEnd - getStart).count();

        // Accumulate (inputGradients is now flat: [input0_lane0, input0_lane1, ..., input1_lane0, ...])
        auto accStart = std::chrono::steady_clock::now();
        for (int lane = 0; lane < actualBatchSize; ++lane)
        {
            res.price += outputBatch[lane];
            res.d_delta += inputGradients[0 * BATCH_SIZE + lane];

            for (size_t k = 0; k < market.lambda.size(); ++k)
                res.d_lambda[k] += inputGradients[(1 + k) * BATCH_SIZE + lane];

            for (size_t k = 0; k < market.L0.size(); ++k)
                res.d_L0[k] += inputGradients[(1 + market.lambda.size() + k) * BATCH_SIZE + lane];
        }
        auto accEnd = std::chrono::steady_clock::now();
        accumulateTotal += std::chrono::duration<double, std::milli>(accEnd - accStart).count();
    }

    timing.setInputsMs = setInputsTotal;
    timing.forwardMs = forwardBackwardTotal;  // Forward+backward combined for AVX
    timing.backwardMs = 0.0;  // Included in forwardMs for AVX
    timing.getGradientsMs = getGradientsTotal;
    timing.accumulateMs = accumulateTotal;

    auto totalEnd = std::chrono::steady_clock::now();
    timing.totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timing;
}

// ============================================================================
// Original Pricing Functions
// ============================================================================

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

    // Create JIT compiler for graph recording
    xad::JITCompiler<double, 1> jit;

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

    // Compile with ForgeBackend
    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    auto compileEnd = std::chrono::steady_clock::now();
    if (stats)
    {
        stats->compileTimeMs =
            std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    }

    // =========================================================================
    // Phase 2: Execute compiled graph for all paths
    // =========================================================================

    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

    double output;
    std::vector<double> inputGradients(numInputs);

    for (int path = 0; path < numPaths; ++path)
    {
        // Set delta
        double deltaVal = market.delta;
        backend.setInput(0, &deltaVal);

        // Set lambda
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            double lambdaVal = market.lambda[i];
            backend.setInput(1 + i, &lambdaVal);
        }

        // Set L0
        for (size_t i = 0; i < market.L0.size(); ++i)
        {
            double L0Val = market.L0[i];
            backend.setInput(1 + market.lambda.size() + i, &L0Val);
        }

        // Set random samples for this path
        size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
        for (size_t i = 0; i < numSamples; ++i)
        {
            double sampleVal = allSamples[path][i];
            backend.setInput(sampleOffset + i, &sampleVal);
        }

        // Forward + backward (combined)
        backend.forwardAndBackward(&output, inputGradients.data());

        // Accumulate results
        res.price += output;
        res.d_delta += inputGradients[0];
        for (size_t i = 0; i < market.lambda.size(); ++i)
        {
            res.d_lambda[i] += inputGradients[1 + i];
            res.d_L0[i] += inputGradients[1 + market.lambda.size() + i];
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

    // Create JIT compiler for graph recording
    xad::JITCompiler<double, 1> jit;

    JitAD delta;
    JitAD v;

    auto compileStart = std::chrono::steady_clock::now();

    // =========================================================================
    // Phase 1: Record the computation graph
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

    // Compile with AVX backend
    xad::forge::ForgeBackendAVX avxBackend(false);
    avxBackend.compile(jit.getGraph());

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
    std::vector<double> inputGradients(numInputs * BATCH_SIZE);

    for (int batch = 0; batch < numBatches; ++batch)
    {
        int batchStart = batch * BATCH_SIZE;
        int actualBatchSize = std::min(BATCH_SIZE, numPaths - batchStart);

        // Set delta (same for all paths)
        for (int lane = 0; lane < BATCH_SIZE; ++lane)
            inputBatch[lane] = market.delta;
        avxBackend.setInput(0, inputBatch.data());

        // Set lambda (same for all paths)
        for (size_t k = 0; k < market.lambda.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.lambda[k];
            avxBackend.setInput(1 + k, inputBatch.data());
        }

        // Set L0 (same for all paths)
        for (size_t k = 0; k < market.L0.size(); ++k)
        {
            for (int lane = 0; lane < BATCH_SIZE; ++lane)
                inputBatch[lane] = market.L0[k];
            avxBackend.setInput(1 + market.lambda.size() + k, inputBatch.data());
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
            avxBackend.setInput(sampleOffset + m, inputBatch.data());
        }

        // Execute forward + backward
        avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());

        // Accumulate results (inputGradients is flat: [input0_lane0, input0_lane1, ..., input1_lane0, ...])
        for (int lane = 0; lane < actualBatchSize; ++lane)
        {
            res.price += outputBatch[lane];
            res.d_delta += inputGradients[0 * BATCH_SIZE + lane];

            for (size_t k = 0; k < market.lambda.size(); ++k)
                res.d_lambda[k] += inputGradients[(1 + k) * BATCH_SIZE + lane];

            for (size_t k = 0; k < market.L0.size(); ++k)
                res.d_L0[k] += inputGradients[(1 + market.lambda.size() + k) * BATCH_SIZE + lane];
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

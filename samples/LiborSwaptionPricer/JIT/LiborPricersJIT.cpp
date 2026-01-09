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
// Helper Functions (reduce code duplication)
// ============================================================================

namespace
{

/// Generate random samples for all Monte Carlo paths
inline std::vector<std::vector<double>> generateSamples(int numPaths, size_t numSamples,
                                                        unsigned long long seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0., 1.);

    std::vector<std::vector<double>> allSamples(numPaths);
    for (int path = 0; path < numPaths; ++path)
    {
        allSamples[path].resize(numSamples);
        std::generate(begin(allSamples[path]), end(allSamples[path]), [&]() { return dist(gen); });
    }
    return allSamples;
}

/// Initialize results structure with proper sizes
inline Results initResults(const MarketParameters& market)
{
    Results res;
    res.price = 0.;
    res.d_lambda.resize(market.lambda.size());
    res.d_delta = 0.0;
    res.d_L0.resize(market.L0.size());
    return res;
}

/// Average results over number of paths
inline void averageResults(Results& res, int numPaths)
{
    res.price /= numPaths;
    res.d_delta /= numPaths;
    for (size_t i = 0; i < res.d_lambda.size(); ++i)
    {
        res.d_lambda[i] /= numPaths;
        res.d_L0[i] /= numPaths;
    }
}

/// Set scalar backend inputs for one path
inline void setScalarInputs(xad::JITBackend<double>& backend, const MarketParameters& market,
                            const std::vector<double>& pathSamples)
{
    double deltaVal = market.delta;
    backend.setInput(0, &deltaVal);

    for (size_t i = 0; i < market.lambda.size(); ++i)
    {
        double lambdaVal = market.lambda[i];
        backend.setInput(1 + i, &lambdaVal);
    }

    for (size_t i = 0; i < market.L0.size(); ++i)
    {
        double L0Val = market.L0[i];
        backend.setInput(1 + market.lambda.size() + i, &L0Val);
    }

    size_t sampleOffset = 1 + market.lambda.size() + market.L0.size();
    for (size_t i = 0; i < pathSamples.size(); ++i)
    {
        double sampleVal = pathSamples[i];
        backend.setInput(sampleOffset + i, &sampleVal);
    }
}

/// Accumulate scalar results from one path
inline void accumulateScalarResults(Results& res, double output,
                                    const std::vector<double>& inputGradients, size_t lambdaSize)
{
    res.price += output;
    res.d_delta += inputGradients[0];
    for (size_t i = 0; i < lambdaSize; ++i)
    {
        res.d_lambda[i] += inputGradients[1 + i];
        res.d_L0[i] += inputGradients[1 + lambdaSize + i];
    }
}

}  // namespace

// ============================================================================
// Performance Decomposition Functions
// ============================================================================

TimingDecomposition runDecompositionJIT(const SwaptionPortfolio& portfolio,
                                         const MarketParameters& market,
                                         int numPaths, unsigned long long seed)
{
    using JitAD = xad::AD;
    const size_t numSamples = market.lambda.size() / 2;

    TimingDecomposition timing;
    timing.numPaths = numPaths;

    auto totalStart = std::chrono::steady_clock::now();

    // Use helpers for common setup
    auto allSamples = generateSamples(numPaths, numSamples, seed);
    Results res = initResults(market);

    // JIT graph recording (kept inline - requires local vectors to remain in scope)
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);
    xad::JITCompiler<double, 1> jit;
    JitAD delta, v;

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
        // Set inputs (timed)
        auto setStart = std::chrono::steady_clock::now();
        setScalarInputs(backend, market, allSamples[path]);
        auto setEnd = std::chrono::steady_clock::now();
        setInputsTotal += std::chrono::duration<double, std::milli>(setEnd - setStart).count();

        // Forward + backward (timed)
        auto fwdBwdStart = std::chrono::steady_clock::now();
        backend.forwardAndBackward(&output, inputGradients.data());
        auto fwdBwdEnd = std::chrono::steady_clock::now();
        forwardBackwardTotal += std::chrono::duration<double, std::milli>(fwdBwdEnd - fwdBwdStart).count();

        // Get gradients (already retrieved in forwardAndBackward)
        auto getStart = std::chrono::steady_clock::now();
        auto getEnd = std::chrono::steady_clock::now();
        getGradientsTotal += std::chrono::duration<double, std::milli>(getEnd - getStart).count();

        // Accumulate (timed)
        auto accStart = std::chrono::steady_clock::now();
        accumulateScalarResults(res, output, inputGradients, market.lambda.size());
        auto accEnd = std::chrono::steady_clock::now();
        accumulateTotal += std::chrono::duration<double, std::milli>(accEnd - accStart).count();
    }

    timing.setInputsMs = setInputsTotal;
    timing.forwardMs = forwardBackwardTotal;
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
    const size_t numSamples = market.lambda.size() / 2;

    TimingDecomposition timing;
    timing.numPaths = numPaths;

    auto totalStart = std::chrono::steady_clock::now();

    // Use helpers for common setup
    auto allSamples = generateSamples(numPaths, numSamples, seed);
    Results res = initResults(market);

    // JIT graph recording (kept inline - requires local vectors to remain in scope)
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);
    xad::JITCompiler<double, 1> jit;
    JitAD delta, v;

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
    // AVX batched input setting kept inline - different data layout from scalar
    constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX::VECTOR_WIDTH;
    const int numBatches = (numPaths + BATCH_SIZE - 1) / BATCH_SIZE;
    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;

    std::vector<double> inputBatch(BATCH_SIZE);
    std::vector<double> outputBatch(BATCH_SIZE);
    std::vector<double> inputGradients(numInputs * BATCH_SIZE);

    double setInputsTotal = 0.0;
    double forwardBackwardTotal = 0.0;
    double getGradientsTotal = 0.0;
    double accumulateTotal = 0.0;

    for (int batch = 0; batch < numBatches; ++batch)
    {
        int batchStart = batch * BATCH_SIZE;
        int actualBatchSize = std::min(BATCH_SIZE, numPaths - batchStart);

        // Set inputs (timed)
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

        // Forward + Backward (timed)
        auto fwdBwdStart = std::chrono::steady_clock::now();
        avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());
        auto fwdBwdEnd = std::chrono::steady_clock::now();
        forwardBackwardTotal += std::chrono::duration<double, std::milli>(fwdBwdEnd - fwdBwdStart).count();

        // Get gradients (already retrieved in forwardAndBackward)
        auto getStart = std::chrono::steady_clock::now();
        auto getEnd = std::chrono::steady_clock::now();
        getGradientsTotal += std::chrono::duration<double, std::milli>(getEnd - getStart).count();

        // Accumulate (timed) - AVX layout: [input0_lane0..3, input1_lane0..3, ...]
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
    timing.forwardMs = forwardBackwardTotal;
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
    using JitAD = xad::AD;
    const size_t numSamples = market.lambda.size() / 2;

    // Use helpers for common setup
    auto allSamples = generateSamples(numPaths, numSamples, seed);
    Results res = initResults(market);

    // JIT graph recording (kept inline - requires local vectors to remain in scope)
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);
    xad::JITCompiler<double, 1> jit;
    JitAD delta, v;

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

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    auto compileEnd = std::chrono::steady_clock::now();
    if (stats)
    {
        stats->compileTimeMs =
            std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    }

    // Execute compiled graph for all paths using helpers
    const size_t numInputs = 1 + market.lambda.size() + market.L0.size() + numSamples;
    double output;
    std::vector<double> inputGradients(numInputs);

    for (int path = 0; path < numPaths; ++path)
    {
        setScalarInputs(backend, market, allSamples[path]);
        backend.forwardAndBackward(&output, inputGradients.data());
        accumulateScalarResults(res, output, inputGradients, market.lambda.size());
    }

    averageResults(res, numPaths);
    return res;
}

Results pricePortfolioJIT_AVX(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                              int numPaths, unsigned long long seed, JITStats* stats)
{
    using JitAD = xad::AD;
    const size_t numSamples = market.lambda.size() / 2;

    // Use helpers for common setup
    auto allSamples = generateSamples(numPaths, numSamples, seed);
    Results res = initResults(market);

    // JIT graph recording (kept inline - requires local vectors to remain in scope)
    std::vector<JitAD> L, tmp1, tmp2;
    std::vector<JitAD> lambda, L0;
    std::vector<JitAD> jit_samples(numSamples);
    xad::JITCompiler<double, 1> jit;
    JitAD delta, v;

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
    if (stats)
    {
        stats->compileTimeMs =
            std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
    }

    // Execute compiled AVX kernel for all paths (4 at a time)
    // AVX batched input setting kept inline - different data layout from scalar
    constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX::VECTOR_WIDTH;
    const int numBatches = (numPaths + BATCH_SIZE - 1) / BATCH_SIZE;
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

        avxBackend.forwardAndBackward(outputBatch.data(), inputGradients.data());

        // Accumulate results (inputGradients layout: [input0_lane0..3, input1_lane0..3, ...])
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

    averageResults(res, numPaths);
    return res;
}

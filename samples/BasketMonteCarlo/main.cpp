/*******************************************************************************

   Prices a European basket call on correlated assets by Monte Carlo and
   computes the pathwise delta and vega of every asset in adjoint mode, with the
   paths split across threads. Each thread has its own tape and records one path
   at a time, so memory does not grow with the number of paths. The Greeks are
   checked against bump-and-revalue with the same random numbers, and against a
   single-threaded run, which must give identical results.


   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

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

*******************************************************************************/

#include <XAD/XAD.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

// tape and active data type for 1st order adjoint computation
typedef xad::adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;

// the basket: equally weighted, at the money, one year
const int NUM_ASSETS = 12;
const double SPOTS[NUM_ASSETS] = {100.0, 95.0,  105.0, 98.0,  102.0, 110.0,
                                  90.0,  101.0, 97.0,  103.0, 99.0,  108.0};
const double VOLS[NUM_ASSETS] = {0.20, 0.25, 0.18, 0.30, 0.22, 0.35,
                                 0.15, 0.28, 0.24, 0.19, 0.32, 0.26};
const double WEIGHT = 1.0 / NUM_ASSETS;
const double STRIKE = 100.0;
const double RATE = 0.03;
const double MATURITY = 1.0;     // in years
const double CORRELATION = 0.5;  // between every pair of assets

const std::uint64_t SEED = 20261115;
const int DEFAULT_NUM_PATHS = 100000;
// paths are processed in chunks of this size, each with its own random number stream; the
// results do not depend on how the chunks are shared out between threads
const int PATHS_PER_CHUNK = 256;

// Small bumps: a path whose basket ends within a bump of the strike crosses the kink of the
// payoff, where the difference quotient is not the pathwise slope. Such paths make an error in
// the bumped Greeks roughly proportional to the bump size.
const double SPOT_BUMP = 1e-6;  // relative to the spot
const double VOL_BUMP = 1e-6;   // absolute

const double PRICE_TOLERANCE = 1e-12;  // relative, adjoint price against the plain price
const double BUMP_TOLERANCE = 1e-5;    // relative, adjoint Greeks against bump-and-revalue

// splitmix64 output function: scrambles the bits of x
std::uint64_t scrambleBits(std::uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// small splitmix64 generator with Box-Muller normals; hand-written rather than <random>, as
// std::normal_distribution gives different numbers with different standard libraries
class RandomGenerator
{
  public:
    // one independent stream per (seed, stream) pair
    RandomGenerator(std::uint64_t seed, std::uint64_t stream)
        : state_(scrambleBits(seed ^ scrambleBits(stream + 1))), hasSpare_(false), spare_(0.0)
    {
    }

    std::uint64_t drawBits()
    {
        state_ += 0x9E3779B97F4A7C15ULL;
        return scrambleBits(state_);
    }

    // uniform in (0, 1], so that its logarithm is finite
    double drawUniform() { return ((drawBits() >> 11) + 1.0) / 9007199254740992.0; }

    // standard normal, by the Box-Muller transform (which gives two at a time)
    double drawNormal()
    {
        if (hasSpare_)
        {
            hasSpare_ = false;
            return spare_;
        }
        const double TWO_PI = 6.283185307179586476925;
        double radius = std::sqrt(-2.0 * std::log(drawUniform()));
        double angle = TWO_PI * drawUniform();
        spare_ = radius * std::sin(angle);
        hasSpare_ = true;
        return radius * std::cos(angle);
    }

  private:
    std::uint64_t state_;
    bool hasSpare_;
    double spare_;
};

struct Market
{
    double spot[NUM_ASSETS];
    double vol[NUM_ASSETS];
    double cholesky[NUM_ASSETS][NUM_ASSETS];  // lower triangular factor of the correlation matrix
};

Market setupMarket()
{
    Market market;
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        market.spot[i] = SPOTS[i];
        market.vol[i] = VOLS[i];
    }

    // Cholesky factorisation of the correlation matrix (1 on the diagonal, CORRELATION
    // elsewhere); the correlation is not differentiated, so this stays in double
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        for (int j = 0; j < NUM_ASSETS; ++j) market.cholesky[i][j] = 0.0;
        for (int j = 0; j <= i; ++j)
        {
            double sum = (i == j) ? 1.0 : CORRELATION;
            for (int k = 0; k < j; ++k) sum -= market.cholesky[i][k] * market.cholesky[j][k];
            market.cholesky[i][j] = (i == j) ? std::sqrt(sum) : sum / market.cholesky[j][j];
        }
    }
    return market;
}

// correlated standard normals z = L * eps for one path
void generateCorrelatedNormals(RandomGenerator& gen, const Market& market, double* z)
{
    double eps[NUM_ASSETS];
    for (int i = 0; i < NUM_ASSETS; ++i) eps[i] = gen.drawNormal();
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        z[i] = 0.0;
        for (int j = 0; j <= i; ++j) z[i] += market.cholesky[i][j] * eps[j];
    }
}

// discounted basket call payoff on one path, written once for any (active or passive) type T
template <typename T>
T pricePath(const T* spot, const T* vol, const double* z)
{
    using std::exp;

    const double sqrtT = std::sqrt(MATURITY);
    T basket = 0.0;
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        T terminal =
            spot[i] * exp((RATE - 0.5 * vol[i] * vol[i]) * MATURITY + vol[i] * sqrtT * z[i]);
        basket += WEIGHT * terminal;
    }
    T intrinsic = basket - STRIKE;
    T payoff = intrinsic > 0.0 ? intrinsic : T(0.0);
    return std::exp(-RATE * MATURITY) * payoff;
}

// sums (per chunk) or averages (in total) over the paths
struct Results
{
    double price;
    double delta[NUM_ASSETS];
    double vega[NUM_ASSETS];
    std::size_t tapeBytes;  // largest tape memory of any thread; 0 without a tape
};

void clearResults(Results& r)
{
    r.price = 0.0;
    for (int i = 0; i < NUM_ASSETS; ++i) r.delta[i] = r.vega[i] = 0.0;
    r.tapeBytes = 0;
}

int countChunks(int numPaths) { return (numPaths + PATHS_PER_CHUNK - 1) / PATHS_PER_CHUNK; }

// Each worker processes chunks firstChunk, firstChunk + chunkStride, ... and writes the sums of
// chunk c to chunkSums[c]. Different threads never write to the same entry, so no locks are
// needed.
typedef void (*ChunkWorker)(const Market& market, int numPaths, int firstChunk, int chunkStride,
                            Results* chunkSums);

// plain pricing in double: the reference for the cost of the Greeks
void priceChunksPlain(const Market& market, int numPaths, int firstChunk, int chunkStride,
                      Results* chunkSums)
{
    double z[NUM_ASSETS];
    for (int c = firstChunk; c < countChunks(numPaths); c += chunkStride)
    {
        RandomGenerator gen(SEED, c);
        Results& sum = chunkSums[c];
        clearResults(sum);
        int endPath = std::min(numPaths, (c + 1) * PATHS_PER_CHUNK);
        for (int path = c * PATHS_PER_CHUNK; path < endPath; ++path)
        {
            generateCorrelatedNormals(gen, market, z);
            sum.price += pricePath(market.spot, market.vol, z);
        }
    }
}

// price and all deltas and vegas with one adjoint sweep per path
void priceChunksAdjoint(const Market& market, int numPaths, int firstChunk, int chunkStride,
                        Results* chunkSums)
{
    // Every thread needs its own tape. The active tape is thread-local, and the constructor makes
    // this tape the active one of the thread that runs this function.
    tape_type tape;

    double z[NUM_ASSETS];
    for (int c = firstChunk; c < countChunks(numPaths); c += chunkStride)
    {
        RandomGenerator gen(SEED, c);
        Results& sum = chunkSums[c];
        clearResults(sum);
        int endPath = std::min(numPaths, (c + 1) * PATHS_PER_CHUNK);
        for (int path = c * PATHS_PER_CHUNK; path < endPath; ++path)
        {
            generateCorrelatedNormals(gen, market, z);

            // record this path only: the tape is cleared and the inputs are registered afresh,
            // so it never holds more than one path, however many there are
            tape.clearAll();
            AD spot[NUM_ASSETS], vol[NUM_ASSETS];
            for (int i = 0; i < NUM_ASSETS; ++i)
            {
                spot[i] = market.spot[i];
                vol[i] = market.vol[i];
                tape.registerInput(spot[i]);
                tape.registerInput(vol[i]);
            }
            tape.newRecording();

            AD payoff = pricePath(spot, vol, z);

            tape.registerOutput(payoff);
            derivative(payoff) = 1.0;
            tape.computeAdjoints();

            // the pathwise Greeks are the derivatives of the payoff on each path
            sum.price += value(payoff);
            for (int i = 0; i < NUM_ASSETS; ++i)
            {
                sum.delta[i] += derivative(spot[i]);
                sum.vega[i] += derivative(vol[i]);
            }
        }
        sum.tapeBytes = tape.getMemory();
    }
}

// bump-and-revalue: central differences on every input, with the same random numbers
void priceChunksBumped(const Market& market, int numPaths, int firstChunk, int chunkStride,
                       Results* chunkSums)
{
    double spot[NUM_ASSETS], vol[NUM_ASSETS], z[NUM_ASSETS];
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        spot[i] = market.spot[i];
        vol[i] = market.vol[i];
    }

    for (int c = firstChunk; c < countChunks(numPaths); c += chunkStride)
    {
        RandomGenerator gen(SEED, c);
        Results& sum = chunkSums[c];
        clearResults(sum);
        int endPath = std::min(numPaths, (c + 1) * PATHS_PER_CHUNK);
        for (int path = c * PATHS_PER_CHUNK; path < endPath; ++path)
        {
            generateCorrelatedNormals(gen, market, z);
            sum.price += pricePath(spot, vol, z);

            for (int i = 0; i < NUM_ASSETS; ++i)
            {
                double h = SPOT_BUMP * market.spot[i];
                spot[i] = market.spot[i] + h;
                double up = pricePath(spot, vol, z);
                spot[i] = market.spot[i] - h;
                double down = pricePath(spot, vol, z);
                spot[i] = market.spot[i];
                sum.delta[i] += (up - down) / (2.0 * h);

                vol[i] = market.vol[i] + VOL_BUMP;
                up = pricePath(spot, vol, z);
                vol[i] = market.vol[i] - VOL_BUMP;
                down = pricePath(spot, vol, z);
                vol[i] = market.vol[i];
                sum.vega[i] += (up - down) / (2.0 * VOL_BUMP);
            }
        }
    }
}

// runs the worker on numThreads threads and averages the chunk sums, adding them up in chunk
// order so that the result is the same for any number of threads
Results runPaths(ChunkWorker worker, const Market& market, int numPaths, int numThreads,
                 double& seconds)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    std::vector<Results> chunkSums(countChunks(numPaths));
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t)
        threads.push_back(
            std::thread(worker, std::cref(market), numPaths, t, numThreads, chunkSums.data()));
    for (std::size_t t = 0; t < threads.size(); ++t) threads[t].join();

    Results total;
    clearResults(total);
    for (std::size_t c = 0; c < chunkSums.size(); ++c)
    {
        total.price += chunkSums[c].price;
        for (int i = 0; i < NUM_ASSETS; ++i)
        {
            total.delta[i] += chunkSums[c].delta[i];
            total.vega[i] += chunkSums[c].vega[i];
        }
        total.tapeBytes = std::max(total.tapeBytes, chunkSums[c].tapeBytes);
    }
    total.price /= numPaths;
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        total.delta[i] /= numPaths;
        total.vega[i] /= numPaths;
    }

    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return total;
}

double computeRelativeDifference(double a, double reference)
{
    return std::abs(a - reference) / std::max(std::abs(reference), 1e-300);
}

// true if both runs gave exactly the same price and Greeks
bool areIdentical(const Results& a, const Results& b)
{
    if (a.price != b.price)
        return false;
    for (int i = 0; i < NUM_ASSETS; ++i)
        if (a.delta[i] != b.delta[i] || a.vega[i] != b.vega[i])
            return false;
    return true;
}

/// Usage:
///   BasketMonteCarlo [numPaths] [numThreads]
///
/// By default it uses 100,000 paths and as many threads as the hardware supports.
int main(int argc, char* argv[])
{
    int numPaths = argc > 1 ? std::atoi(argv[1]) : DEFAULT_NUM_PATHS;
    int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (argc > 2)
        numThreads = std::atoi(argv[2]);
    else if (numThreads < 1)
        numThreads = 1;
    if (numPaths < 1 || numThreads < 1)
    {
        std::printf("usage: %s [numPaths] [numThreads], both positive\n", argv[0]);
        return 1;
    }
#ifdef XAD_NO_THREADLOCAL
    // without thread-local tapes, only one tape can be active in the whole process
    if (numThreads > 1)
    {
        std::printf("XAD was built with XAD_NO_THREADLOCAL: running on 1 thread\n\n");
        numThreads = 1;
    }
#endif

    Market market = setupMarket();
    std::printf("European basket call on %d assets, equally weighted, correlation %.2f\n",
                NUM_ASSETS, CORRELATION);
    std::printf("strike %.2f, rate %.2f%%, maturity %.2f years\n", STRIKE, RATE * 100.0, MATURITY);
    std::printf("%d paths on %d thread(s), in %d chunks of up to %d paths\n\n", numPaths,
                numThreads, countChunks(numPaths), PATHS_PER_CHUNK);

    double plainSeconds, adjointSeconds, bumpedSeconds;
    Results plain = runPaths(priceChunksPlain, market, numPaths, numThreads, plainSeconds);
    Results adjoint = runPaths(priceChunksAdjoint, market, numPaths, numThreads, adjointSeconds);
    Results bumped = runPaths(priceChunksBumped, market, numPaths, numThreads, bumpedSeconds);

    std::printf("price %.10f (plain), %.10f (XAD)\n\n", plain.price, adjoint.price);
    bool ok = true;
    // written so that a NaN fails the checks
    if (!(computeRelativeDifference(adjoint.price, plain.price) <= PRICE_TOLERANCE))
    {
        std::printf("the XAD price does not match the plain price\n");
        ok = false;
    }

    std::printf("%5s %8s %6s %13s %13s %10s %13s %13s %10s\n", "asset", "spot", "vol",
                "delta (XAD)", "delta (bump)", "rel. diff", "vega (XAD)", "vega (bump)",
                "rel. diff");
    for (int i = 0; i < NUM_ASSETS; ++i)
    {
        double relDelta = computeRelativeDifference(adjoint.delta[i], bumped.delta[i]);
        double relVega = computeRelativeDifference(adjoint.vega[i], bumped.vega[i]);
        std::printf("%5d %8.2f %6.2f %13.10f %13.10f %10.3e %13.10f %13.10f %10.3e\n", i,
                    market.spot[i], market.vol[i], adjoint.delta[i], bumped.delta[i], relDelta,
                    adjoint.vega[i], bumped.vega[i], relVega);
        if (!(relDelta <= BUMP_TOLERANCE) || !(relVega <= BUMP_TOLERANCE))
            ok = false;
    }
    if (!ok)
        std::printf("\nthe XAD Greeks do not match bump-and-revalue (relative tolerance %.0e)\n",
                    BUMP_TOLERANCE);

    // costs relative to one plain pricing, as absolute times depend on the machine
    std::printf("\ncost relative to one plain pricing (%d inputs)\n", 2 * NUM_ASSETS);
    std::printf("  price and all Greeks with XAD:  %6.1fx\n", adjointSeconds / plainSeconds);
    std::printf("  bump-and-revalue (central):     %6.1fx  (%d pricings per path)\n",
                bumpedSeconds / plainSeconds, 4 * NUM_ASSETS + 1);
    std::printf("largest tape: %.1f KiB per thread, whatever the number of paths\n",
                adjoint.tapeBytes / 1024.0);

    // the same random numbers and the same order of summation: a single thread must give
    // exactly the same results
    if (numThreads > 1)
    {
        double singleSeconds;
        Results single = runPaths(priceChunksAdjoint, market, numPaths, 1, singleSeconds);
        bool identical = areIdentical(adjoint, single);
        std::printf("\nXAD on 1 thread: %s results, %.1fx the time on %d threads\n",
                    identical ? "identical" : "DIFFERENT", singleSeconds / adjointSeconds,
                    numThreads);
        if (!identical)
            ok = false;
    }

    return ok ? 0 : 1;
}

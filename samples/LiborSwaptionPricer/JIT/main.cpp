/*******************************************************************************

   LIBOR Swaption Portfolio Pricer - JIT Benchmark

   This benchmark compares different approaches for computing sensitivities
   in Monte Carlo pricing of a LIBOR swaption portfolio.

   APPROACHES TESTED:
     FD      - Finite Differences (bump-and-revalue)
     XAD     - XAD tape-based reverse-mode AAD
     JIT     - Forge JIT-compiled native code (scalar)
     JIT-AVX - Forge JIT + AVX2 SIMD (4 paths/instruction)

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

#include "../LiborData.hpp"
#include "../LiborPricers.hpp"
#include "LiborPricersJIT.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

// ============================================================================
// Environment Detection
// ============================================================================

namespace
{

std::string getCpuInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    char brand[49] = {0};
    unsigned int regs[4];

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 0x80000000);
#else
    __get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[0] >= 0x80000004)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
#ifdef _WIN32
            __cpuid(reinterpret_cast<int*>(regs), 0x80000002 + i);
#else
            __get_cpuid(0x80000002 + i, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            std::memcpy(brand + i * 16, regs, 16);
        }
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos)
            result = result.substr(start);
        return result;
    }
#endif
    return "Unknown CPU";
}

std::string getPlatformInfo()
{
#ifdef _WIN32
    typedef LONG(WINAPI * RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    HMODULE hMod = GetModuleHandleW(L"ntdll.dll");
    if (hMod)
    {
        auto RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(hMod, "RtlGetVersion");
        if (RtlGetVersion)
        {
            RTL_OSVERSIONINFOW rovi = {0};
            rovi.dwOSVersionInfoSize = sizeof(rovi);
            if (RtlGetVersion(&rovi) == 0)
            {
                std::ostringstream oss;
                oss << "Windows " << rovi.dwMajorVersion << "." << rovi.dwMinorVersion << " (Build "
                    << rovi.dwBuildNumber << ")";
                return oss.str();
            }
        }
    }
    return "Windows";
#else
    struct utsname buf;
    if (uname(&buf) == 0)
    {
        std::ostringstream oss;
        oss << buf.sysname << " " << buf.release;
        return oss.str();
    }
    return "Unknown";
#endif
}

std::string getMemoryInfo()
{
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo))
    {
        double gb = static_cast<double>(memInfo.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
    {
        double gb = static_cast<double>(pages) * page_size / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#endif
    return "Unknown";
}

std::string getCompilerInfo()
{
#if defined(_MSC_VER)
    std::ostringstream oss;
    oss << "MSVC " << _MSC_VER / 100 << "." << _MSC_VER % 100;
#if defined(_DEBUG)
    oss << " (Debug)";
#else
    oss << " (Release)";
#endif
    return oss.str();
#elif defined(__clang__)
    std::ostringstream oss;
    oss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    return oss.str();
#elif defined(__GNUC__)
    std::ostringstream oss;
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return oss.str();
#else
    return "Unknown Compiler";
#endif
}

std::string getSimdInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int regs[4];
    std::vector<std::string> features;

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 1);
#else
    __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[2] & (1 << 0)) features.push_back("SSE3");
    if (regs[2] & (1 << 19)) features.push_back("SSE4.1");
    if (regs[2] & (1 << 20)) features.push_back("SSE4.2");
    if (regs[2] & (1 << 28)) features.push_back("AVX");

#ifdef _WIN32
    __cpuidex(reinterpret_cast<int*>(regs), 7, 0);
#else
    __get_cpuid_count(7, 0, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[1] & (1 << 5)) features.push_back("AVX2");
    if (regs[1] & (1 << 16)) features.push_back("AVX512F");

    if (features.empty())
        return "None detected";

    std::ostringstream oss;
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (i > 0) oss << ", ";
        oss << features[i];
    }
    return oss.str();
#else
    return "N/A (non-x86)";
#endif
}

}  // namespace

// ============================================================================
// Test Setup
// ============================================================================

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

// ============================================================================
// Statistics Helpers
// ============================================================================

double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

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

// ============================================================================
// Validation
// ============================================================================

struct ValidationResult
{
    bool priceMatch = false;
    int derivativesMatched = 0;
    int derivativesTotal = 0;
};

ValidationResult validateResults(const Results& test, const Results& reference, double tol = 1e-4)
{
    ValidationResult vr;
    vr.derivativesTotal = 1 + static_cast<int>(reference.d_lambda.size() + reference.d_L0.size());

    vr.priceMatch = std::abs(test.price - reference.price) / (std::abs(reference.price) + 1e-10) < tol;

    if (std::abs(test.d_delta - reference.d_delta) / (std::abs(reference.d_delta) + 1e-10) < tol)
        vr.derivativesMatched++;

    for (size_t i = 0; i < reference.d_lambda.size(); ++i)
    {
        if (std::abs(test.d_lambda[i] - reference.d_lambda[i]) /
                (std::abs(reference.d_lambda[i]) + 1e-10) < tol)
            vr.derivativesMatched++;
    }

    for (size_t i = 0; i < reference.d_L0.size(); ++i)
    {
        if (std::abs(test.d_L0[i] - reference.d_L0[i]) / (std::abs(reference.d_L0[i]) + 1e-10) < tol)
            vr.derivativesMatched++;
    }

    return vr;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h     Show this help message\n";
    std::cout << "  --quick        Run quick benchmark (fewer iterations, fewer path counts)\n";
    std::cout << "\nThis benchmark compares AD approaches for LIBOR swaption pricing.\n";
    std::cout << "Build: Full benchmark with Forge JIT\n";
}

int main(int argc, char** argv)
{
    bool quickMode = false;
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
    }

    constexpr unsigned long long SEED = 91672912;

    SwaptionPortfolio portfolio = setupTestPortfolio();
    MarketParameters market = setupTestMarket();

    std::vector<int> pathCounts = quickMode ? std::vector<int>{100, 1000, 10000}
                                            : std::vector<int>{10, 100, 1000, 10000, 50000, 100000, 400000};

    constexpr int FD_MAX_PATHS = 1000;

    size_t warmupIterations = quickMode ? 1 : 2;
    size_t benchmarkIterations = quickMode ? 2 : 3;

    size_t totalInputs = 1 + market.lambda.size() + market.L0.size();

    // =========================================================================
    // Print Header
    // =========================================================================
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  LIBOR Swaption Portfolio Pricer - JIT Benchmark\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << "\n  ENVIRONMENT\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Platform:     " << getPlatformInfo() << "\n";
    std::cout << "  CPU:          " << getCpuInfo() << "\n";
    std::cout << "  RAM:          " << getMemoryInfo() << "\n";
    std::cout << "  SIMD:         " << getSimdInfo() << "\n";
    std::cout << "  Compiler:     " << getCompilerInfo() << "\n";

    std::cout << "\n  INSTRUMENT\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Portfolio:    " << portfolio.maturities.size() << " European swaptions\n";
    std::cout << "  Maturities:   4, 8, 20, 28, 40 years (3 each)\n";
    std::cout << "  Model:        LIBOR Market Model (lognormal forwards)\n";

    std::cout << "\n  MARKET DATA\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  delta:        1 parameter\n";
    std::cout << "  lambda:       " << market.lambda.size() << " volatility parameters\n";
    std::cout << "  L0:           " << market.L0.size() << " initial forward rates\n";
    std::cout << "  Total inputs: " << totalInputs << " sensitivities\n";

    std::cout << "\n  BENCHMARK CONFIGURATION\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  Path counts:  ";
    for (size_t i = 0; i < pathCounts.size(); ++i)
    {
        if (i > 0) std::cout << ", ";
        if (pathCounts[i] >= 1000)
            std::cout << (pathCounts[i] / 1000) << "K";
        else
            std::cout << pathCounts[i];
    }
    std::cout << "\n";
    std::cout << "  Warmup:       " << warmupIterations << " iterations\n";
    std::cout << "  Measured:     " << benchmarkIterations << " iterations\n";

    std::cout << "\n  METHODS\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "  FD       Finite Differences (bump-and-revalue, paths <= " << FD_MAX_PATHS << " only)\n";
    std::cout << "  XAD      XAD tape-based reverse-mode AAD\n";
    std::cout << "  JIT      Forge JIT-compiled native code\n";
    std::cout << "  JIT-AVX  Forge JIT + AVX2 SIMD (4 paths/instruction)\n";

    // =========================================================================
    // Run Benchmarks
    // =========================================================================
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  RUNNING BENCHMARKS\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    struct TimingResult
    {
        bool fd_measured = false;
        double fd_mean = 0, fd_std = 0;
        double xad_mean = 0, xad_std = 0;
        double jit_mean = 0, jit_std = 0;
        double jit_avx_mean = 0, jit_avx_std = 0;
    };
    std::vector<TimingResult> results(pathCounts.size());

    for (size_t tc = 0; tc < pathCounts.size(); ++tc)
    {
        int numPaths = pathCounts[tc];

        std::cout << "  [" << (tc + 1) << "/" << pathCounts.size() << "] ";
        if (numPaths >= 1000)
            std::cout << (numPaths / 1000) << "K";
        else
            std::cout << numPaths;
        std::cout << " paths ";
        std::cout << std::string(10, '.') << " " << std::flush;

        std::vector<double> fd_times, xad_times, jit_times, jit_avx_times;

        for (size_t iter = 0; iter < warmupIterations + benchmarkIterations; ++iter)
        {
            bool recordTiming = (iter >= warmupIterations);

            if (numPaths <= FD_MAX_PATHS)
            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioFD(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming) fd_times.push_back(elapsed.count());
            }

            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioAD(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming) xad_times.push_back(elapsed.count());
            }

            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioJIT(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming) jit_times.push_back(elapsed.count());
            }

            {
                auto start = std::chrono::steady_clock::now();
                auto res = pricePortfolioJIT_AVX(portfolio, market, numPaths, SEED);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                if (recordTiming) jit_avx_times.push_back(elapsed.count());
            }
        }

        results[tc].fd_measured = (numPaths <= FD_MAX_PATHS);
        if (results[tc].fd_measured)
        {
            results[tc].fd_mean = mean(fd_times);
            results[tc].fd_std = stddev(fd_times);
        }
        results[tc].xad_mean = mean(xad_times);
        results[tc].xad_std = stddev(xad_times);
        results[tc].jit_mean = mean(jit_times);
        results[tc].jit_std = stddev(jit_times);
        results[tc].jit_avx_mean = mean(jit_avx_times);
        results[tc].jit_avx_std = stddev(jit_avx_times);

        std::cout << "done\n";
    }

    // =========================================================================
    // Validation
    // =========================================================================
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  VALIDATION\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";
    std::cout << "  Comparing all methods against Finite Differences (" << FD_MAX_PATHS << " paths):\n";
    std::cout << "\n";

    int validationPaths = FD_MAX_PATHS;
    auto resFD = pricePortfolioFD(portfolio, market, validationPaths, SEED);
    auto resXAD = pricePortfolioAD(portfolio, market, validationPaths, SEED);
    auto resJIT = pricePortfolioJIT(portfolio, market, validationPaths, SEED);
    auto resAVX = pricePortfolioJIT_AVX(portfolio, market, validationPaths, SEED);

    auto vrXAD = validateResults(resXAD, resFD);
    auto vrJIT = validateResults(resJIT, resFD);
    auto vrAVX = validateResults(resAVX, resFD);

    std::cout << "  Method   | Price | Derivatives | Status\n";
    std::cout << "  ---------+-------+-------------+--------\n";

    std::cout << "  XAD      |  " << (vrXAD.priceMatch ? "OK " : "ERR") << "  |   "
              << std::setw(3) << vrXAD.derivativesMatched << "/" << vrXAD.derivativesTotal
              << "   |  " << (vrXAD.priceMatch && vrXAD.derivativesMatched == vrXAD.derivativesTotal ? "PASS" : "FAIL") << "\n";

    std::cout << "  JIT      |  " << (vrJIT.priceMatch ? "OK " : "ERR") << "  |   "
              << std::setw(3) << vrJIT.derivativesMatched << "/" << vrJIT.derivativesTotal
              << "   |  " << (vrJIT.priceMatch && vrJIT.derivativesMatched == vrJIT.derivativesTotal ? "PASS" : "FAIL") << "\n";

    std::cout << "  JIT-AVX  |  " << (vrAVX.priceMatch ? "OK " : "ERR") << "  |   "
              << std::setw(3) << vrAVX.derivativesMatched << "/" << vrAVX.derivativesTotal
              << "   |  " << (vrAVX.priceMatch && vrAVX.derivativesMatched == vrAVX.derivativesTotal ? "PASS" : "FAIL") << "\n";

    // =========================================================================
    // Results Table
    // =========================================================================
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  RESULTS (mean +/- stddev, in ms)\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    std::cout << "  Paths  | Method   |        Mean |      StdDev\n";
    std::cout << "  -------+----------+-------------+-------------\n";

    for (size_t tc = 0; tc < pathCounts.size(); ++tc)
    {
        std::string pathLabel;
        if (pathCounts[tc] >= 1000)
            pathLabel = std::to_string(pathCounts[tc] / 1000) + "K";
        else
            pathLabel = std::to_string(pathCounts[tc]);

        std::cout << std::fixed << std::setprecision(2);

        if (results[tc].fd_measured)
        {
            std::cout << "  " << std::setw(6) << pathLabel << " | FD       |"
                      << std::setw(12) << results[tc].fd_mean << " |"
                      << std::setw(12) << results[tc].fd_std << "\n";
        }
        else
        {
            std::cout << "  " << std::setw(6) << pathLabel << " | FD       |"
                      << std::setw(12) << "-" << " |"
                      << std::setw(12) << "-" << "\n";
        }

        std::cout << "         | XAD      |"
                  << std::setw(12) << results[tc].xad_mean << " |"
                  << std::setw(12) << results[tc].xad_std << "\n";

        std::cout << "         | JIT      |"
                  << std::setw(12) << results[tc].jit_mean << " |"
                  << std::setw(12) << results[tc].jit_std << "\n";

        std::cout << "         | JIT-AVX  |"
                  << std::setw(12) << results[tc].jit_avx_mean << " |"
                  << std::setw(12) << results[tc].jit_avx_std << "\n";

        if (tc < pathCounts.size() - 1)
            std::cout << "  -------+----------+-------------+-------------\n";
    }

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  Benchmark complete.\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\n";

    return 0;
}

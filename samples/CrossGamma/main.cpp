/*******************************************************************************

   Computes the full cross-gamma matrix of second derivatives of a basket
   option with respect to asset spot prices in forward-over-adjoint mode,
   using closed-form lognormal moment matching. Checks matrix symmetry,
   validates against central finite differences of first derivatives,
   and reports the cost relative to single pricing and finite-difference bumping.

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

******************************************************************************/

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <XAD/Hessian.hpp>
#include <XAD/XAD.hpp>

// 2nd-order forward-over-adjoint mode
typedef xad::fwd_adj<double> mode_2nd;
typedef mode_2nd::tape_type tape_2nd;
typedef mode_2nd::active_type AD2;

// 1st-order adjoint mode for finite-difference verification
typedef xad::adj<double> mode_1st;
typedef mode_1st::tape_type tape_1st;
typedef mode_1st::active_type AD1;

// Standard normal cumulative distribution function N(x) = 0.5 * erfc(-x / sqrt(2))
template <typename T>
T normalCdf(const T& x)
{
    using std::erfc;
    using xad::erfc;
    const double inv_sqrt2 = 0.70710678118654752440;
    return 0.5 * erfc(-x * inv_sqrt2);
}

// Basket option price via Levy lognormal moment matching
template <typename T>
T basketPrice(const std::vector<T>& spots, const std::vector<double>& weights,
              const std::vector<double>& vols, const std::vector<std::vector<double>>& corr,
              double strike, double r, double T_mat)
{
    using std::exp;
    using std::log;
    using std::sqrt;
    using xad::exp;
    using xad::log;
    using xad::sqrt;

    std::size_t n = spots.size();

    // First moment M1 = E[B_T] = e^(r * T) * sum(w_i * S_i)
    T sum_ws = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        sum_ws += weights[i] * spots[i];
    }
    T M1 = sum_ws * exp(r * T_mat);

    // Second moment M2 = E[B_T^2] = sum_{i,j} w_i * w_j * S_i * S_j * e^((2r + rho_ij*s_i*s_j)*T)
    T M2 = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double exponent = (2.0 * r + corr[i][j] * vols[i] * vols[j]) * T_mat;
            M2 += weights[i] * weights[j] * spots[i] * spots[j] * exp(exponent);
        }
    }

    // Lognormal variance: nu^2 = ln(M2 / M1^2) = ln(M2) - 2 * ln(M1)
    T nu2 = log(M2) - 2.0 * log(M1);
    T nu = sqrt(nu2);

    // Black-Scholes call pricing on moment-matched lognormal basket
    T d1 = (log(M1 / strike) + 0.5 * nu2) / nu;
    T d2 = d1 - nu;

    return exp(-r * T_mat) * (M1 * normalCdf(d1) - strike * normalCdf(d2));
}

// Computes 1st-order deltas using adjoint mode for central finite-difference verification
std::vector<double> computeDeltas(const std::vector<double>& spots,
                                  const std::vector<double>& weights,
                                  const std::vector<double>& vols,
                                  const std::vector<std::vector<double>>& corr, double strike,
                                  double r, double T_mat, tape_1st& tape)
{
    std::size_t n = spots.size();
    std::vector<AD1> spots_ad(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        spots_ad[i] = spots[i];
    }

    tape.registerInputs(spots_ad);
    tape.newRecording();

    AD1 price = basketPrice(spots_ad, weights, vols, corr, strike, r, T_mat);
    tape.registerOutput(price);
    derivative(price) = 1.0;
    tape.computeAdjoints();

    std::vector<double> deltas(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        deltas[i] = derivative(spots_ad[i]);
    }
    return deltas;
}

int main()
{
    const std::size_t N = 15;
    std::vector<double> spots(N);
    std::vector<double> weights(N, 1.0 / static_cast<double>(N));
    std::vector<double> vols(N);
    std::vector<std::vector<double>> corr(N, std::vector<double>(N));

    for (std::size_t i = 0; i < N; ++i)
    {
        spots[i] = 90.0 + 20.0 * static_cast<double>(i) / static_cast<double>(N - 1);
        vols[i] = 0.15 + 0.15 * static_cast<double>(i) / static_cast<double>(N - 1);
        for (std::size_t j = 0; j < N; ++j)
        {
            corr[i][j] = std::pow(0.5, std::abs(static_cast<int>(i) - static_cast<int>(j)));
        }
    }

    const double strike = 100.0;
    const double r = 0.05;
    const double T_mat = 1.0;

    double price0 = basketPrice(spots, weights, vols, corr, strike, r, T_mat);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Basket Option Cross-Gamma Sample (Second-Order AD)\n";
    std::cout << "=================================================\n";
    std::cout << "Number of basket assets: " << N << "\n";
    std::cout << "Option strike:           " << strike << "\n";
    std::cout << "Risk-free rate:          " << r << "\n";
    std::cout << "Maturity:                " << T_mat << " year\n";
    std::cout << "Basket Option Price:     " << price0 << "\n\n";

    // 1. Compute full second-derivative (Hessian) matrix using 2nd-order AD (forward-over-adjoint)
    tape_2nd tape2;
    std::vector<AD2> spots_ad2(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        spots_ad2[i] = spots[i];
    }

    std::function<AD2(std::vector<AD2>&)> priceFunc = [&](std::vector<AD2>& x) -> AD2
    { return basketPrice(x, weights, vols, corr, strike, r, T_mat); };

    auto H_ad = xad::computeHessian(spots_ad2, priceFunc, &tape2);

    // 2. Check matrix symmetry: max |H_ij - H_ji|
    double max_asym = 0.0;
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = i + 1; j < N; ++j)
        {
            max_asym = std::max(max_asym, std::abs(H_ad[i][j] - H_ad[j][i]));
        }
    }

    std::cout << "Symmetry Check:\n";
    std::cout << "  Max |Gamma_ij - Gamma_ji|: " << std::scientific << std::setprecision(2)
              << max_asym << std::fixed << std::setprecision(6) << "\n\n";

    // 3. Check against central finite differences of first derivatives (deltas)
    tape_1st tape1;
    const double h = 1e-6;
    double max_rel_err = 0.0;

    for (std::size_t j = 0; j < N; ++j)
    {
        auto s_plus = spots;
        auto s_minus = spots;
        s_plus[j] += h;
        s_minus[j] -= h;

        auto deltas_plus = computeDeltas(s_plus, weights, vols, corr, strike, r, T_mat, tape1);
        auto deltas_minus = computeDeltas(s_minus, weights, vols, corr, strike, r, T_mat, tape1);

        for (std::size_t i = 0; i < N; ++i)
        {
            double gamma_fd = (deltas_plus[i] - deltas_minus[i]) / (2.0 * h);
            double rel_err = std::abs(H_ad[i][j] - gamma_fd) / (std::abs(H_ad[i][j]) + 1e-8);
            max_rel_err = std::max(max_rel_err, rel_err);
        }
    }

    std::cout << "Finite-Difference Check (vs central differences of 1st-order deltas):\n";
    std::cout << "  Max relative difference:   " << std::scientific << std::setprecision(2)
              << max_rel_err << std::fixed << std::setprecision(6) << "\n\n";

    // Print a 4x4 submatrix snippet of the cross-gammas
    std::cout << "Cross-Gamma Matrix Snippet (first 4x4 assets):\n";
    std::cout << std::scientific << std::setprecision(4);
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            std::cout << std::setw(13) << H_ad[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::fixed << std::setprecision(6) << "\n";

    // 4. Performance benchmarking: report cost relative to single pricing
    const int REPEAT = 500;

    // Single pricing base timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < REPEAT; ++it)
    {
        volatile double dummy = basketPrice(spots, weights, vols, corr, strike, r, T_mat);
        (void)dummy;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_base = std::chrono::duration<double>(t1 - t0).count() / REPEAT;

    // AD forward-over-adjoint Hessian timing
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < REPEAT; ++it)
    {
        auto H = xad::computeHessian(spots_ad2, priceFunc, &tape2);
        (void)H;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_ad = std::chrono::duration<double>(t1 - t0).count() / REPEAT;

    // Finite-difference bumping timing (evaluates upper triangle by symmetry)
    const double h_bump = 1e-4;
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < REPEAT; ++it)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = i; j < N; ++j)
            {
                if (i == j)
                {
                    auto s_up = spots;
                    auto s_dn = spots;
                    s_up[i] += h_bump;
                    s_dn[i] -= h_bump;
                    volatile double p_up = basketPrice(s_up, weights, vols, corr, strike, r, T_mat);
                    volatile double p_dn = basketPrice(s_dn, weights, vols, corr, strike, r, T_mat);
                    (void)p_up;
                    (void)p_dn;
                }
                else
                {
                    auto s_pp = spots;
                    auto s_pm = spots;
                    auto s_mp = spots;
                    auto s_mm = spots;
                    s_pp[i] += h_bump;
                    s_pp[j] += h_bump;
                    s_pm[i] += h_bump;
                    s_pm[j] -= h_bump;
                    s_mp[i] -= h_bump;
                    s_mp[j] += h_bump;
                    s_mm[i] -= h_bump;
                    s_mm[j] -= h_bump;
                    volatile double p_pp = basketPrice(s_pp, weights, vols, corr, strike, r, T_mat);
                    volatile double p_pm = basketPrice(s_pm, weights, vols, corr, strike, r, T_mat);
                    volatile double p_mp = basketPrice(s_mp, weights, vols, corr, strike, r, T_mat);
                    volatile double p_mm = basketPrice(s_mm, weights, vols, corr, strike, r, T_mat);
                    (void)p_pp;
                    (void)p_pm;
                    (void)p_mp;
                    (void)p_mm;
                }
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_bump = std::chrono::duration<double>(t1 - t0).count() / REPEAT;

    std::cout << "Performance Comparison (" << N << " assets, " << (N * (N + 1) / 2)
              << " unique Hessian elements):\n";
    std::cout << "  Single pricing baseline:   " << std::setw(8) << time_base * 1e6
              << " us  (1.0x)\n";
    std::cout << "  AD Forward-over-Adjoint:   " << std::setw(8) << time_ad * 1e6 << " us  ("
              << time_ad / time_base << "x baseline)\n";
    std::cout << "  Finite-Difference Bumping: " << std::setw(8) << time_bump * 1e6 << " us  ("
              << time_bump / time_base << "x baseline)\n";

    // 5. Verification checks
    const double SYMMETRY_TOLERANCE = 1e-12;
    const double FD_TOLERANCE = 1e-4;

    if (max_asym > SYMMETRY_TOLERANCE)
    {
        std::cerr << "FAIL: Matrix asymmetry " << max_asym << " exceeds tolerance "
                  << SYMMETRY_TOLERANCE << "\n";
        return 1;
    }

    if (max_rel_err > FD_TOLERANCE)
    {
        std::cerr << "FAIL: Finite difference relative error " << max_rel_err
                  << " exceeds tolerance " << FD_TOLERANCE << "\n";
        return 1;
    }

    std::cout << "\nAll checks passed successfully.\n";
    return 0;
}

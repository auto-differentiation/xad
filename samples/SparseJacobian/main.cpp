/*******************************************************************************

   Computes the Jacobian of a function with a tridiagonal sparsity pattern
     y_i = x_{i-1} - 2 x_i + x_{i+1} + x_{i-1} x_{i+1} + 0.1 exp(x_i)
   (with x_{-1} = x_n = 0) using only 3 forward-mode passes, instead of one
   pass per input. The columns are coloured so that the columns sharing a
   colour never touch the same row. The result is checked against the dense
   Jacobian from adjoint mode, and the run times of both are printed.


   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, version 3 of the License only.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include <XAD/Jacobian.hpp>
#include <XAD/XAD.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// number of colours: row i depends on columns i-1, i and i+1, which all have different
// residues modulo 3
const std::size_t NUM_COLOURS = 3;

// the three non-zero diagonals of the tridiagonal Jacobian
// lower[i] = dy_i/dx_{i-1}, diag[i] = dy_i/dx_i, upper[i] = dy_i/dx_{i+1}
struct Bands
{
    std::vector<double> lower, diag, upper;

    explicit Bands(std::size_t n) : lower(n, 0.0), diag(n, 0.0), upper(n, 0.0) {}
};

// the function, written once for any active type T. Each output only depends on its neighbours.
template <typename T>
std::vector<T> f(const std::vector<T>& x)
{
    using std::exp;
    const std::size_t n = x.size();
    std::vector<T> y(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        T left = (i > 0) ? x[i - 1] : T(0.0);
        T right = (i + 1 < n) ? x[i + 1] : T(0.0);
        y[i] = left - 2.0 * x[i] + right + left * right + 0.1 * exp(x[i]);
    }
    return y;
}

// sparse Jacobian using NUM_COLOURS forward passes. (xad::fwd<double, NUM_COLOURS> could do all
// colours in a single pass by seeding derivative(x[j])[j % NUM_COLOURS] = 1; scalar passes are
// used here to keep the idea simple.)
Bands sparseJacobian(const std::vector<double>& x0)
{
    typedef xad::fwd<double>::active_type AD;
    const std::size_t n = x0.size();
    Bands jac(n);

    for (std::size_t colour = 0; colour < NUM_COLOURS; ++colour)
    {
        // seed all the columns of this colour at once: column j has colour j % 3
        std::vector<AD> x(x0.begin(), x0.end());
        for (std::size_t j = 0; j < n; ++j)
            derivative(x[j]) = (j % NUM_COLOURS == colour) ? 1.0 : 0.0;

        std::vector<AD> y = f(x);

        // derivative(y[i]) = sum over seeded columns j of dy_i/dx_j. Only one of the columns
        // i-1, i, i+1 has this colour, so the sum has a single term, which we can assign to
        // the right band.
        for (std::size_t i = 0; i < n; ++i)
        {
            double d = derivative(y[i]);
            if (i > 0 && (i - 1) % NUM_COLOURS == colour)
                jac.lower[i] = d;
            else if (i % NUM_COLOURS == colour)
                jac.diag[i] = d;
            else if (i + 1 < n && (i + 1) % NUM_COLOURS == colour)
                jac.upper[i] = d;
        }
    }
    return jac;
}

// dense Jacobian with adjoint mode: one backward pass per output
std::vector<std::vector<double>> denseJacobian(const std::vector<double>& x0)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    mode::tape_type tape;
    std::vector<AD> x(x0.begin(), x0.end());
    std::function<std::vector<AD>(std::vector<AD>&)> residual = [](std::vector<AD>& in)
    { return f(in); };
    return computeJacobian(x, residual);
}

std::vector<double> makeInput(std::size_t n)
{
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = 0.5 + std::sin(0.37 * static_cast<double>(i));
    return x;
}

// compares sparse and dense results, prints the largest error and returns true if they match
bool check(std::size_t n)
{
    const double TOL = 1e-12;
    std::vector<double> x = makeInput(n);
    Bands sparse = sparseJacobian(x);
    std::vector<std::vector<double>> dense = denseJacobian(x);

    double maxErr = 0.0;
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            // entry of the sparse result; everything outside the three bands is zero
            double s = 0.0;
            if (j + 1 == i)
                s = sparse.lower[i];
            else if (j == i)
                s = sparse.diag[i];
            else if (j == i + 1)
                s = sparse.upper[i];

            double err = std::abs(s - dense[i][j]);
            maxErr = std::max(maxErr, err);
            if (!(err <= TOL * std::max(1.0, std::abs(dense[i][j]))))
            {
                std::cout << "MISMATCH at (" << i << ", " << j << "): sparse = " << s
                          << ", dense = " << dense[i][j] << "\n";
                ok = false;
            }
        }
    std::cout << "n = " << n << ": max |sparse - dense| = " << maxErr
              << (ok ? " (OK)" : " (FAILED)") << "\n";
    return ok;
}

// odd, so the median is the single middle value
const int REPETITIONS = 11;

// runs the function once untimed (warm-up), then REPETITIONS times timed; returns the median.
// run() returns one Jacobian entry, which is added to checksum so the work is really used.
template <typename F>
double medianTimeMs(F run, double& checksum)
{
    typedef std::chrono::steady_clock clock;
    checksum += run();
    std::vector<double> times;
    for (int r = 0; r < REPETITIONS; ++r)
    {
        auto start = clock::now();
        checksum += run();
        times.push_back(std::chrono::duration<double, std::milli>(clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main()
{
    // correctness: n = 10 is not a multiple of 3, so the boundary rows are covered too
    bool ok = check(10);
    ok = check(1000) && ok;

    // timing for n = 1000
    const std::size_t n = 1000;
    std::vector<double> x = makeInput(n);

    double checksum = 0.0;
    double sparseMs = medianTimeMs([&] { return sparseJacobian(x).diag[n / 2]; }, checksum);
    double denseMs = medianTimeMs([&] { return denseJacobian(x)[n / 2][n / 2]; }, checksum);

    std::cout << "median of " << REPETITIONS << " runs:\n"
              << "sparse (" << NUM_COLOURS << " forward passes): " << sparseMs << " ms\n"
              << "dense  (" << n << " adjoint passes): " << denseMs << " ms\n"
              << "checksum (ignore): " << checksum << "\n";

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

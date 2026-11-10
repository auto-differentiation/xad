/*******************************************************************************

   Fits a Nelson-Siegel zero curve (4 parameters) to 20 noisy zero rates with
   Gauss-Newton. The Jacobian of the residuals comes from XAD's vector forward
   mode in a single pass; finite differences are used only to check it once, at
   the starting point.


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

#include <XAD/XAD.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

// parameters: beta0 (long-term level), beta1 (slope), beta2 (curvature), tau (decay time, years)
const std::size_t NUM_PARAMS = 4;
const char* const PARAM_NAMES[NUM_PARAMS] = {"beta0", "beta1", "beta2", "tau"};

const std::size_t MAX_ITERATIONS = 50;
const double STEP_TOLERANCE = 1e-10;  // stop when the relative parameter step is below this
const double FD_TOLERANCE = 1e-6;     // relative tolerance for the finite-difference check
const double NOISE = 2e-4;            // noise half-width: +/- 2 basis points
const double TAU_START = 1.0;         // starting guess for tau, in years

typedef std::vector<double> Vector;
typedef std::vector<Vector> Matrix;

// Nelson-Siegel zero rate for maturity t, written once for any (active or passive) type T
template <typename T>
T zeroRate(const T* p, double t)
{
    using std::exp;
    T x = t / p[3];
    T decay = exp(-x);
    T loading = (1.0 - decay) / x;
    return p[0] + p[1] * loading + p[2] * (loading - decay);
}

// residuals r_i = model(t_i) - observed_i
template <typename T>
std::vector<T> residuals(const T* p, const Vector& maturities, const Vector& observed)
{
    std::vector<T> r(maturities.size());
    for (std::size_t i = 0; i < maturities.size(); ++i)
        r[i] = zeroRate(p, maturities[i]) - observed[i];
    return r;
}

// Jacobian of the residuals (m x 4) in one pass of vector forward mode.
//
// Forward mode costs one directional derivative per parameter, adjoint mode one backward
// sweep per residual. With 4 parameters and 20 residuals forward mode is the better
// choice, and xad::fwd<double, 4> carries all 4 directions at once, so a single evaluation
// of the model gives the full Jacobian - no tape is needed.
Matrix residualJacobian(const Vector& params, const Vector& maturities, const Vector& observed,
                        Vector& r)
{
    typedef xad::fwd<double, NUM_PARAMS> mode;
    typedef mode::active_type AD;

    AD p[NUM_PARAMS];
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        p[k] = params[k];
        derivative(p[k])[k] = 1.0;  // seed direction k: d/dp_k
    }

    std::vector<AD> res = residuals(p, maturities, observed);

    Matrix jac(res.size(), Vector(NUM_PARAMS));
    r.resize(res.size());
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        r[i] = value(res[i]);
        for (std::size_t k = 0; k < NUM_PARAMS; ++k) jac[i][k] = derivative(res[i])[k];
    }
    return jac;
}

// Jacobian by central differences, used only as a check
Matrix finiteDifferenceJacobian(const Vector& params, const Vector& maturities,
                                const Vector& observed)
{
    Matrix jac(maturities.size(), Vector(NUM_PARAMS));
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        double h = 1e-6 * std::max(1.0, std::abs(params[k]));
        Vector up = params, down = params;
        up[k] += h;
        down[k] -= h;
        Vector rUp = residuals(up.data(), maturities, observed);
        Vector rDown = residuals(down.data(), maturities, observed);
        for (std::size_t i = 0; i < maturities.size(); ++i)
            jac[i][k] = (rUp[i] - rDown[i]) / (2.0 * h);
    }
    return jac;
}

double norm(const Vector& v)
{
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

// solves the 4x4 system a * x = b by Gaussian elimination with partial pivoting
Vector solve(Matrix a, Vector b)
{
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col)
    {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < n; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
                pivot = row;
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        for (std::size_t row = col + 1; row < n; ++row)
        {
            double factor = a[row][col] / a[col][col];
            for (std::size_t k = col; k < n; ++k) a[row][k] -= factor * a[col][k];
            b[row] -= factor * b[col];
        }
    }
    Vector x(n);
    for (std::size_t i = n; i-- > 0;)
    {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= a[i][k] * x[k];
        x[i] = s / a[i][i];
    }
    return x;
}

// Gauss-Newton step for the first n parameters (the others held fixed): solves
// (J^T J) dp = -J^T r, using the first n columns of J
Vector gaussNewtonStep(const Matrix& jac, const Vector& r, std::size_t n = NUM_PARAMS)
{
    Matrix jtj(n, Vector(n, 0.0));
    Vector jtr(n, 0.0);
    for (std::size_t i = 0; i < r.size(); ++i)
        for (std::size_t k = 0; k < n; ++k)
        {
            jtr[k] -= jac[i][k] * r[i];
            for (std::size_t l = 0; l < n; ++l) jtj[k][l] += jac[i][k] * jac[i][l];
        }
    return solve(jtj, jtr);
}

// largest relative difference between the XAD and finite-difference Jacobians
double checkJacobian(const Vector& params, const Vector& maturities, const Vector& observed)
{
    Vector r;
    Matrix ad = residualJacobian(params, maturities, observed, r);
    Matrix fd = finiteDifferenceJacobian(params, maturities, observed);

    std::printf("Jacobian check at the starting point (max over the %zu residuals):\n",
                maturities.size());
    std::printf("%8s %15s %15s %12s\n", "param", "max |XAD|", "max |FD|", "rel. diff");
    double worst = 0.0;
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        double maxAd = 0.0, maxFd = 0.0, maxDiff = 0.0;
        for (std::size_t i = 0; i < maturities.size(); ++i)
        {
            maxAd = std::max(maxAd, std::abs(ad[i][k]));
            maxFd = std::max(maxFd, std::abs(fd[i][k]));
            maxDiff = std::max(maxDiff, std::abs(ad[i][k] - fd[i][k]));
        }
        double rel = maxDiff / std::max(maxAd, 1e-300);
        worst = std::max(worst, rel);
        std::printf("%8s %15.8f %15.8f %12.3e\n", PARAM_NAMES[k], maxAd, maxFd, rel);
    }
    std::printf("\n");
    return worst;
}

int main()
{
    // maturities in years
    const double t[] = {0.25, 0.5, 0.75, 1.0, 1.5,  2.0,  2.5,  3.0,  4.0,  5.0,
                        6.0,  7.0, 8.0,  9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0};
    const Vector maturities(t, t + sizeof(t) / sizeof(t[0]));

    // "observed" zero rates: a known curve plus a little uniform noise. std::mt19937 gives
    // the same numbers on every platform; the standard distributions do not, so the mapping
    // to [-NOISE, NOISE] is done by hand.
    const double trueParams[NUM_PARAMS] = {0.045, -0.020, 0.015, 2.0};
    std::mt19937 rng(42);
    Vector observed(maturities.size());
    for (std::size_t i = 0; i < maturities.size(); ++i)
    {
        double u = static_cast<double>(rng()) / 4294967296.0;  // in [0, 1)
        observed[i] = zeroRate(trueParams, maturities[i]) + NOISE * (2.0 * u - 1.0);
    }

    // starting guess: tau = 1 year, and the betas that fit best for that tau. The model is
    // linear in the betas, so one Gauss-Newton step on them alone (from zero) solves that
    // linear least-squares problem exactly. A poor start for the betas can send Gauss-Newton
    // to a local minimum with a much smaller tau.
    Vector params(NUM_PARAMS, 0.0);
    params[3] = TAU_START;
    {
        Vector r0;
        Matrix jac0 = residualJacobian(params, maturities, observed, r0);
        Vector betas = gaussNewtonStep(jac0, r0, 3);
        for (std::size_t k = 0; k < 3; ++k) params[k] = betas[k];
    }

    double fdError = checkJacobian(params, maturities, observed);
    if (fdError > FD_TOLERANCE)
    {
        std::printf("XAD and finite-difference Jacobians differ by %.3e (> %.0e)\n", fdError,
                    FD_TOLERANCE);
        return 1;
    }

    std::printf("%4s %11s %11s %11s %11s %14s\n", "iter", "beta0", "beta1", "beta2", "tau", "|r|");
    Vector r;
    bool converged = false;
    for (std::size_t iter = 0; iter <= MAX_ITERATIONS; ++iter)
    {
        Matrix jac = residualJacobian(params, maturities, observed, r);
        double rNorm = norm(r);
        std::printf("%4zu %11.6f %11.6f %11.6f %11.6f %14.6e\n", iter, params[0], params[1],
                    params[2], params[3], rNorm);
        if (converged)
            break;

        Vector step = gaussNewtonStep(jac, r);

        // a full Gauss-Newton step can overshoot far from the solution: halve it until the
        // residual norm goes down and tau stays positive
        double alpha = 1.0;
        bool accepted = false;
        Vector trial(NUM_PARAMS);
        for (int halvings = 0; halvings < 30 && !accepted; ++halvings, alpha *= 0.5)
        {
            for (std::size_t k = 0; k < NUM_PARAMS; ++k) trial[k] = params[k] + alpha * step[k];
            accepted =
                trial[3] > 0.0 && norm(residuals(trial.data(), maturities, observed)) <= rNorm;
        }
        if (!accepted)
        {
            std::printf("no step reduces the residual norm\n");
            return 1;
        }

        double relStep = 0.0;
        for (std::size_t k = 0; k < NUM_PARAMS; ++k)
            relStep = std::max(
                relStep, std::abs(trial[k] - params[k]) / std::max(1e-2, std::abs(params[k])));
        params = trial;
        converged = relStep < STEP_TOLERANCE;
    }

    std::printf("\n%8s %11s %11s\n", "param", "true", "fitted");
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
        std::printf("%8s %11.6f %11.6f\n", PARAM_NAMES[k], trueParams[k], params[k]);

    // uniform noise of half-width NOISE has an RMS of NOISE / sqrt(3); a good fit leaves
    // residuals of about that size
    double rms = norm(r) / std::sqrt(static_cast<double>(r.size()));
    double noiseRms = NOISE / std::sqrt(3.0);
    std::printf("\nRMS residual = %.2f bp (noise RMS = %.2f bp)\n", rms * 1e4, noiseRms * 1e4);

    if (!converged)
    {
        std::printf("Gauss-Newton did not converge in %zu iterations\n", MAX_ITERATIONS);
        return 1;
    }
    if (!(rms < 1.5 * noiseRms))
    {
        std::printf("the fit is much worse than the noise level (a local minimum?)\n");
        return 1;
    }
    return 0;
}

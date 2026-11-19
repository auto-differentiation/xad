/*******************************************************************************

   Calibrates the five Heston parameters to a grid of 35 European call prices
   (5 maturities x 7 strikes) with Levenberg-Marquardt. The semi-analytic pricer
   integrates the characteristic function in complex arithmetic, so it runs on
   std::complex<xad::AReal<double>>; the Jacobian of the residuals comes from
   XAD's adjoint mode, and is checked against finite differences once.


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
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

// parameters: kappa (mean reversion), theta (long-run variance), sigma (vol of variance),
// rho (correlation), v0 (initial variance)
const std::size_t NUM_PARAMS = 5;
const char* const PARAM_NAMES[NUM_PARAMS] = {"kappa", "theta", "sigma", "rho", "v0"};

const double SPOT = 100.0;  // the interest rate and dividend yield are zero
const std::size_t NUM_NODES = 200;
const double INTEGRATION_LIMIT = 100.0;  // the integrand has decayed to nothing by then

const std::size_t MAX_ITERATIONS = 100;
const double COST_TOLERANCE = 1e-24;  // stop when the sum of squared residuals is below this
const double STEP_TOLERANCE = 1e-12;  // ... or the relative parameter step is below this
const double FD_TOLERANCE = 1e-5;     // relative tolerance for the finite-difference check
const double PARAM_TOLERANCE = 1e-6;  // recovered parameters must match to this (absolute)

typedef std::vector<double> Vector;
typedef std::vector<Vector> Matrix;

// Gauss-Legendre nodes and weights on [0, INTEGRATION_LIMIT], by Newton iteration on P_n
struct Quadrature
{
    Vector nodes, weights;

    Quadrature() : nodes(NUM_NODES), weights(NUM_NODES)
    {
        const double pi = std::acos(-1.0);
        const double n = static_cast<double>(NUM_NODES);
        for (std::size_t i = 0; i < NUM_NODES; ++i)
        {
            double x = std::cos(pi * (static_cast<double>(i) + 0.75) / (n + 0.5));
            double dp = 1.0;
            for (int it = 0; it < 100; ++it)
            {
                double p0 = 1.0, p1 = x;
                for (std::size_t k = 2; k <= NUM_NODES; ++k)
                {
                    double kk = static_cast<double>(k);
                    double p2 = ((2.0 * kk - 1.0) * x * p1 - (kk - 1.0) * p0) / kk;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n * (x * p1 - p0) / (x * x - 1.0);
                double dx = p1 / dp;
                x -= dx;
                if (std::abs(dx) < 1e-15)
                    break;
            }
            nodes[i] = 0.5 * INTEGRATION_LIMIT * (x + 1.0);
            weights[i] = INTEGRATION_LIMIT / ((1.0 - x * x) * dp * dp);
        }
    }
};

// Heston characteristic function E[exp(a * X)] of X = ln(S_T / S_0), in the "little Heston
// trap" form of Albrecher et al.: with exp(-d T) in the formula (not exp(+d T)), the
// complex logarithm does not jump across its branch cut, for any maturity.
template <typename Real>
std::complex<Real> hestonCf(const std::complex<Real>& a, const Real* p, double maturity)
{
    using std::exp;
    using std::log;
    using std::sqrt;
    typedef std::complex<Real> Cx;
    const Real &kappa = p[0], &theta = p[1], &sigma = p[2], &rho = p[3], &v0 = p[4];

    Cx b = Cx(kappa, Real(0.0)) - a * (rho * sigma);
    Cx d = sqrt(b * b + (a - a * a) * (sigma * sigma));
    Cx g = (b - d) / (b + d);
    Cx e = exp(-d * maturity);
    Cx one(Real(1.0), Real(0.0));
    Cx c = (b - d) * maturity - Real(2.0) * log((one - g * e) / (one - g));
    return exp(c * (kappa * theta / (sigma * sigma)) +
               (b - d) * (one - e) / (one - g * e) * (v0 / (sigma * sigma)));
}

// European call price by Lewis' formula,
//   C = S - sqrt(S K) / pi * int_0^inf Re[exp(i u k) phi(u - i/2)] / (u^2 + 1/4) du,  k = ln(S/K)
// with the integral evaluated by Gauss-Legendre quadrature. Written once for any real type
// Real (double, or an XAD active type).
template <typename Real>
Real hestonCall(const Real* p, double strike, double maturity, const Quadrature& quad)
{
    const double k = std::log(SPOT / strike);
    Real integral = 0.0;
    for (std::size_t i = 0; i < NUM_NODES; ++i)
    {
        const double u = quad.nodes[i];
        // a = i * (u - i/2) = 1/2 + i u
        std::complex<Real> a(Real(0.5), Real(u));
        std::complex<Real> f = hestonCf(a, p, maturity);
        // Re[exp(i u k) f] = cos(u k) Re f - sin(u k) Im f
        integral = integral + (f.real() * std::cos(u * k) - f.imag() * std::sin(u * k)) *
                                  (quad.weights[i] / (u * u + 0.25));
    }
    return SPOT - std::sqrt(SPOT * strike) / std::acos(-1.0) * integral;
}

struct Option
{
    double maturity, strike;
};

// residuals r_i = model_i - observed_i
template <typename Real>
std::vector<Real> residuals(const Real* p, const std::vector<Option>& options,
                            const Vector& observed, const Quadrature& quad)
{
    std::vector<Real> r(options.size());
    for (std::size_t i = 0; i < options.size(); ++i)
        r[i] = hestonCall(p, options[i].strike, options[i].maturity, quad) - observed[i];
    return r;
}

// Jacobian of the residuals (m x 5) in adjoint mode: the model is recorded once, then one
// backward sweep per residual gives one row of the Jacobian. Also returns the residuals.
Matrix residualJacobian(xad::adj<double>::tape_type& tape, const Vector& params,
                        const std::vector<Option>& options, const Vector& observed,
                        const Quadrature& quad, Vector& r)
{
    typedef xad::adj<double>::active_type AD;

    tape.clearAll();
    AD p[NUM_PARAMS];
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        p[k] = params[k];
        tape.registerInput(p[k]);
    }
    tape.newRecording();

    std::vector<AD> res = residuals(p, options, observed, quad);

    Matrix jac(res.size(), Vector(NUM_PARAMS));
    r.resize(res.size());
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        r[i] = value(res[i]);
        tape.registerOutput(res[i]);
    }
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        tape.clearDerivatives();
        derivative(res[i]) = 1.0;
        tape.computeAdjoints();
        for (std::size_t k = 0; k < NUM_PARAMS; ++k) jac[i][k] = derivative(p[k]);
    }
    return jac;
}

// Jacobian by central differences, used only as a check
Matrix finiteDifferenceJacobian(const Vector& params, const std::vector<Option>& options,
                                const Vector& observed, const Quadrature& quad)
{
    Matrix jac(options.size(), Vector(NUM_PARAMS));
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        double h = 1e-6 * std::max(1.0, std::abs(params[k]));
        Vector up = params, down = params;
        up[k] += h;
        down[k] -= h;
        Vector rUp = residuals(up.data(), options, observed, quad);
        Vector rDown = residuals(down.data(), options, observed, quad);
        for (std::size_t i = 0; i < options.size(); ++i)
            jac[i][k] = (rUp[i] - rDown[i]) / (2.0 * h);
    }
    return jac;
}

double sumSquares(const Vector& v)
{
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * v[i];
    return s;
}

// solves the n x n system a * x = b by Gaussian elimination with partial pivoting
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

// Levenberg-Marquardt step: solves (J^T J + lambda diag(J^T J)) dp = -J^T r
Vector levenbergMarquardtStep(const Matrix& jac, const Vector& r, double lambda)
{
    Matrix jtj(NUM_PARAMS, Vector(NUM_PARAMS, 0.0));
    Vector jtr(NUM_PARAMS, 0.0);
    for (std::size_t i = 0; i < r.size(); ++i)
        for (std::size_t k = 0; k < NUM_PARAMS; ++k)
        {
            jtr[k] -= jac[i][k] * r[i];
            for (std::size_t l = 0; l < NUM_PARAMS; ++l) jtj[k][l] += jac[i][k] * jac[i][l];
        }
    for (std::size_t k = 0; k < NUM_PARAMS; ++k) jtj[k][k] *= 1.0 + lambda;
    return solve(jtj, jtr);
}

// the parameters must stay in the domain of the model
bool isValid(const Vector& p)
{
    return p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0 && std::abs(p[3]) < 1.0 && p[4] > 0.0;
}

// largest relative difference between the XAD and finite-difference Jacobians
double checkJacobian(xad::adj<double>::tape_type& tape, const Vector& params,
                     const std::vector<Option>& options, const Vector& observed,
                     const Quadrature& quad)
{
    Vector r;
    Matrix ad = residualJacobian(tape, params, options, observed, quad, r);
    Matrix fd = finiteDifferenceJacobian(params, options, observed, quad);

    std::printf("Jacobian check at the starting point (max over the %zu residuals):\n",
                options.size());
    std::printf("%8s %15s %15s %12s\n", "param", "max |XAD|", "max |FD|", "rel. diff");
    double worst = 0.0;
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        double maxAd = 0.0, maxFd = 0.0, maxDiff = 0.0;
        for (std::size_t i = 0; i < options.size(); ++i)
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
    const Quadrature quad;

    // 5 maturities (years) x 7 strikes
    const double maturities[] = {0.25, 0.5, 1.0, 2.0, 3.0};
    const double strikes[] = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};
    std::vector<Option> options;
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 7; ++j)
        {
            Option o = {maturities[i], strikes[j]};
            options.push_back(o);
        }

    // "observed" prices: generated from known parameters, so the fit can be compared to them
    const double trueParams[NUM_PARAMS] = {2.0, 0.04, 0.35, -0.7, 0.05};
    Vector observed(options.size());
    for (std::size_t i = 0; i < options.size(); ++i)
        observed[i] = hestonCall(trueParams, options[i].strike, options[i].maturity, quad);

    // perturbed starting point
    const double start[NUM_PARAMS] = {1.0, 0.06, 0.5, -0.3, 0.08};
    Vector params(start, start + NUM_PARAMS);

    xad::adj<double>::tape_type tape;

    double fdError = checkJacobian(tape, params, options, observed, quad);
    if (fdError > FD_TOLERANCE)
    {
        std::printf("XAD and finite-difference Jacobians differ by %.3e (> %.0e)\n", fdError,
                    FD_TOLERANCE);
        return 1;
    }

    std::printf("%4s %9s %9s %9s %9s %9s %14s\n", "iter", "kappa", "theta", "sigma", "rho", "v0",
                "sum r^2");
    double lambda = 1e-3;
    bool converged = false;
    for (std::size_t iter = 0; iter <= MAX_ITERATIONS; ++iter)
    {
        Vector r;
        Matrix jac = residualJacobian(tape, params, options, observed, quad, r);
        double cost = sumSquares(r);
        std::printf("%4zu %9.5f %9.5f %9.5f %9.5f %9.5f %14.6e\n", iter, params[0], params[1],
                    params[2], params[3], params[4], cost);
        if (converged || cost < COST_TOLERANCE)
        {
            converged = true;
            break;
        }

        // raise lambda until the step is valid and reduces the cost, then relax it again
        bool accepted = false;
        Vector trial(NUM_PARAMS);
        double relStep = 0.0;
        for (int tries = 0; tries < 30 && !accepted; ++tries, lambda *= 10.0)
        {
            Vector step = levenbergMarquardtStep(jac, r, lambda);
            for (std::size_t k = 0; k < NUM_PARAMS; ++k) trial[k] = params[k] + step[k];
            accepted = isValid(trial) &&
                       sumSquares(residuals(trial.data(), options, observed, quad)) < cost;
            if (accepted)
            {
                relStep = 0.0;
                for (std::size_t k = 0; k < NUM_PARAMS; ++k)
                    relStep = std::max(relStep,
                                       std::abs(step[k]) / std::max(1e-2, std::abs(params[k])));
            }
        }
        if (!accepted)
        {
            std::printf("no step reduces the cost\n");
            return 1;
        }
        lambda = std::max(lambda / 100.0, 1e-12);
        params = trial;
        converged = relStep < STEP_TOLERANCE;
    }

    std::printf("\n%8s %11s %11s\n", "param", "true", "fitted");
    double worst = 0.0;
    for (std::size_t k = 0; k < NUM_PARAMS; ++k)
    {
        std::printf("%8s %11.6f %11.6f\n", PARAM_NAMES[k], trueParams[k], params[k]);
        worst = std::max(worst, std::abs(params[k] - trueParams[k]));
    }

    if (!converged)
    {
        std::printf("Levenberg-Marquardt did not converge in %zu iterations\n", MAX_ITERATIONS);
        return 1;
    }
    if (!(worst < PARAM_TOLERANCE))
    {
        std::printf("the fit does not recover the parameters (max error %.3e)\n", worst);
        return 1;
    }
    return 0;
}

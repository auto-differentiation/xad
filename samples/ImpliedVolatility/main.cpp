/*******************************************************************************

   Solves Black-Scholes for the implied volatility of a European call with
   Newton's method in plain double, then computes its sensitivities to the
   option price, spot and rate with the implicit function theorem: one adjoint
   sweep at the converged volatility, without differentiating the solver loop.
   The results are checked against closed-form values and against finite
   differences of the full solve.


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
#include <cmath>
#include <cstddef>
#include <cstdio>

// market data; the option price is generated from TRUE_VOL, which the solver should recover
const double SPOT = 100.0;
const double STRIKE = 105.0;
const double RATE = 0.05;
const double MATURITY = 1.0;  // in years
const double TRUE_VOL = 0.20;

const double INITIAL_VOL = 0.5;      // Newton's starting guess, deliberately far from the root
const int MAX_ITERATIONS = 50;       // Newton converges in a handful of iterations here
const double VOL_TOLERANCE = 1e-14;  // stop when the Newton step is below this

const double CLOSED_FORM_TOLERANCE = 1e-12;  // relative, against the closed-form sensitivities
const double FD_TOLERANCE = 1e-6;            // relative, against finite differences

// the inputs the implied volatility is differentiated with respect to
const std::size_t NUM_INPUTS = 3;
const char* const INPUT_NAMES[NUM_INPUTS] = {"price", "spot", "rate"};
enum
{
    PRICE_INDEX,
    SPOT_INDEX,
    RATE_INDEX
};

// standard normal cumulative distribution function, for double and XAD's active types
template <typename T>
T computeNormalCdf(const T& x)
{
    using std::erfc;
    return 0.5 * erfc(-x / std::sqrt(2.0));
}

// Black-Scholes price of a European call, written once for any (active or passive) type T
template <typename T>
T priceCall(const T& spot, const T& rate, const T& vol)
{
    using std::exp;
    using std::log;

    T volSqrtT = vol * std::sqrt(MATURITY);
    T d1 = (log(spot / STRIKE) + (rate + 0.5 * vol * vol) * MATURITY) / volSqrtT;
    T d2 = d1 - volSqrtT;
    return spot * computeNormalCdf(d1) - STRIKE * exp(-rate * MATURITY) * computeNormalCdf(d2);
}

struct Greeks
{
    double delta;  // dC/dspot
    double vega;   // dC/dvol
    double rho;    // dC/drate
};

// closed-form Black-Scholes Greeks of the call, used by Newton and as a reference
Greeks computeClosedFormGreeks(double spot, double rate, double vol)
{
    const double PI = 3.14159265358979323846;
    double volSqrtT = vol * std::sqrt(MATURITY);
    double d1 = (std::log(spot / STRIKE) + (rate + 0.5 * vol * vol) * MATURITY) / volSqrtT;
    double d2 = d1 - volSqrtT;
    double density = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * PI);

    Greeks g;
    g.delta = computeNormalCdf(d1);
    g.vega = spot * density * std::sqrt(MATURITY);
    g.rho = STRIKE * MATURITY * std::exp(-rate * MATURITY) * computeNormalCdf(d2);
    return g;
}

// Newton's method on F(vol) = priceCall(vol) - price in plain double, with the analytic vega
// as dF/dvol. Returns false if it does not converge.
bool solveImpliedVol(const double* inputs, double& vol, bool verbose)
{
    vol = INITIAL_VOL;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        double f = priceCall(inputs[SPOT_INDEX], inputs[RATE_INDEX], vol) - inputs[PRICE_INDEX];
        double vega = computeClosedFormGreeks(inputs[SPOT_INDEX], inputs[RATE_INDEX], vol).vega;
        if (verbose)
            std::printf("%4d %18.15f %14.6e\n", iter, vol, f);

        double step = f / vega;
        // never more than halve the volatility in one step, so it stays positive
        vol = std::max(vol - step, 0.5 * vol);
        if (std::abs(step) < VOL_TOLERANCE)
            return true;
    }
    return false;
}

// central finite differences of the full Newton solve with respect to every input
bool computeFiniteDifferences(const double* inputs, double* fd)
{
    for (std::size_t k = 0; k < NUM_INPUTS; ++k)
    {
        double h = 1e-5 * std::max(1.0, std::abs(inputs[k]));
        double up[NUM_INPUTS], down[NUM_INPUTS];
        for (std::size_t j = 0; j < NUM_INPUTS; ++j) up[j] = down[j] = inputs[j];
        up[k] += h;
        down[k] -= h;

        double volUp, volDown;
        if (!solveImpliedVol(up, volUp, false) || !solveImpliedVol(down, volDown, false))
            return false;
        fd[k] = (volUp - volDown) / (2.0 * h);
    }
    return true;
}

double computeRelativeDifference(double a, double reference)
{
    return std::abs(a - reference) / std::max(std::abs(reference), 1e-300);
}

int main()
{
    // tape and active data type for 1st order adjoint computation
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    double inputs[NUM_INPUTS];
    inputs[SPOT_INDEX] = SPOT;
    inputs[RATE_INDEX] = RATE;
    inputs[PRICE_INDEX] = priceCall(SPOT, RATE, TRUE_VOL);

    std::printf("European call: spot %.2f, strike %.2f, rate %.2f%%, maturity %.2f years\n", SPOT,
                STRIKE, RATE * 100.0, MATURITY);
    std::printf("market price %.10f (generated from vol = %.2f%%)\n\n", inputs[PRICE_INDEX],
                TRUE_VOL * 100.0);

    // ---- solve for the implied volatility in plain double ----

    std::printf("Newton's method\n%4s %18s %14s\n", "iter", "vol", "price error");
    double vol;
    if (!solveImpliedVol(inputs, vol, true))
    {
        std::printf("Newton's method did not converge in %d iterations\n", MAX_ITERATIONS);
        return 1;
    }
    std::printf("\nimplied vol = %.15f\n\n", vol);

    // ---- sensitivities of the implied volatility via the implicit function theorem ----
    //
    // Why not just run Newton on active types and differentiate through the iterations?
    // That gives the derivative of the *last iterate*, not of the root: it depends on the
    // starting guess and on when the loop stops, it can lag behind the values (and for
    // solvers such as bisection it is simply zero, as the iterates are piecewise constant in
    // the inputs), and every iteration ends up on the tape.
    //
    // Instead, at the root F(vol*, p) = priceCall(vol*; spot, rate) - price = 0 holds for all
    // inputs p near the current ones. Differentiating that identity gives
    //
    //     dvol*/dp = -(dF/dp) / (dF/dvol)
    //
    // so only the partial derivatives of F at the single point (vol*, p) are needed. One
    // adjoint sweep of F gives all of them at once, however many inputs there are.

    double sensitivity[NUM_INPUTS];
    tape_type tape;
    {
        // set independent variables and register them
        AD activeVol = vol;
        AD activeInputs[NUM_INPUTS];
        tape.registerInput(activeVol);
        for (std::size_t k = 0; k < NUM_INPUTS; ++k)
        {
            activeInputs[k] = inputs[k];
            tape.registerInput(activeInputs[k]);
        }

        // start recording derivatives
        tape.newRecording();

        AD f = priceCall(activeInputs[SPOT_INDEX], activeInputs[RATE_INDEX], activeVol) -
               activeInputs[PRICE_INDEX];

        // register output, seed adjoint, and compute all input adjoints
        tape.registerOutput(f);
        derivative(f) = 1.0;
        tape.computeAdjoints();

        // implicit function theorem: dvol*/dp = -(dF/dp) / (dF/dvol)
        double dFdVol = derivative(activeVol);
        for (std::size_t k = 0; k < NUM_INPUTS; ++k)
            sensitivity[k] = -derivative(activeInputs[k]) / dFdVol;

        std::printf("at the root: F = %.3e, dF/dvol = %.10f (vega)\n\n", value(f), dFdVol);
    }

    // closed form: F = C - price, so dvol/dprice = 1/vega, dvol/dspot = -delta/vega and
    // dvol/drate = -rho/vega
    Greeks g = computeClosedFormGreeks(SPOT, RATE, vol);
    double closedForm[NUM_INPUTS];
    closedForm[PRICE_INDEX] = 1.0 / g.vega;
    closedForm[SPOT_INDEX] = -g.delta / g.vega;
    closedForm[RATE_INDEX] = -g.rho / g.vega;

    double fd[NUM_INPUTS];
    if (!computeFiniteDifferences(inputs, fd))
    {
        std::printf("Newton's method did not converge for a perturbed input\n");
        return 1;
    }

    std::printf("sensitivities of the implied vol\n");
    std::printf("%10s %16s %16s %16s %11s %11s\n", "dvol/d", "implicit (XAD)", "closed form",
                "finite diff.", "rel. diff", "rel. diff");
    std::printf("%10s %16s %16s %16s %11s %11s\n", "", "", "", "", "closed", "FD");
    bool ok = true;
    for (std::size_t k = 0; k < NUM_INPUTS; ++k)
    {
        double relClosed = computeRelativeDifference(sensitivity[k], closedForm[k]);
        double relFd = computeRelativeDifference(sensitivity[k], fd[k]);
        std::printf("%10s %16.10f %16.10f %16.10f %11.3e %11.3e\n", INPUT_NAMES[k], sensitivity[k],
                    closedForm[k], fd[k], relClosed, relFd);
        // written so that a NaN fails the check
        if (!(relClosed <= CLOSED_FORM_TOLERANCE) || !(relFd <= FD_TOLERANCE))
            ok = false;
    }

    if (!ok)
    {
        std::printf(
            "\nthe implicit-function sensitivities do not match (tolerances %.0e closed "
            "form, %.0e finite differences)\n",
            CLOSED_FORM_TOLERANCE, FD_TOLERANCE);
        return 1;
    }
    return 0;
}

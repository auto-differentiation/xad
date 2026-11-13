/*******************************************************************************

   Prices a European call and put with the Black-Scholes formula and computes
   their Greeks (delta, vega, rho, theta) twice: with forward mode, one pass
   per input, and with adjoint mode, one recording for all inputs. Both are
   checked against the closed-form Greeks.


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
#include <iomanip>
#include <iostream>
#include <string>

const double PI = 3.14159265358979323846;

// Standard normal cumulative distribution function.
template <class T>
T normalCdf(const T& x)
{
    using std::erfc;
    using std::sqrt;
    return 0.5 * erfc(-x / sqrt(2.0));
}

// Black-Scholes price of a European option on a non-dividend paying stock,
// written once for any number type: double, xad::FReal<double> or xad::AReal<double>.
template <class T>
T blackScholes(bool isCall, const T& spot, double strike, const T& vol, const T& rate,
               const T& maturity)
{
    using std::exp;
    using std::log;
    using std::sqrt;

    T volSqrtT = vol * sqrt(maturity);
    T d1 = (log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / volSqrtT;
    T d2 = d1 - volSqrtT;
    T discount = exp(-rate * maturity);

    if (isCall)
        return spot * normalCdf(d1) - strike * discount * normalCdf(d2);
    return strike * discount * normalCdf<T>(-d2) - spot * normalCdf<T>(-d1);
}

struct Greeks
{
    double price;
    double delta;  // dV/dS
    double vega;   // dV/dsigma
    double rho;    // dV/dr
    double theta;  // dV/dt = -dV/dT, the change in value as calendar time passes
};

struct Market
{
    double spot;
    double strike;
    double vol;
    double rate;
    double maturity;
};

// Closed-form Greeks, used as the reference.
Greeks closedForm(bool isCall, const Market& m)
{
    double sqrtT = std::sqrt(m.maturity);
    double d1 = (std::log(m.spot / m.strike) + (m.rate + 0.5 * m.vol * m.vol) * m.maturity) /
                (m.vol * sqrtT);
    double d2 = d1 - m.vol * sqrtT;
    double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * PI);
    double discountedStrike = m.strike * std::exp(-m.rate * m.maturity);
    double sign = isCall ? 1.0 : -1.0;

    Greeks g;
    g.price = blackScholes(isCall, m.spot, m.strike, m.vol, m.rate, m.maturity);
    g.delta = isCall ? normalCdf(d1) : normalCdf(d1) - 1.0;
    g.vega = m.spot * pdf * sqrtT;
    g.rho = sign * m.maturity * discountedStrike * normalCdf(sign * d2);
    g.theta = -m.spot * pdf * m.vol / (2.0 * sqrtT) -
              sign * m.rate * discountedStrike * normalCdf(sign * d2);
    return g;
}

// Forward mode: one pass seeds the tangent of one input and gives the derivative of the price
// with respect to that input. Input 0 is the spot, 1 the volatility, 2 the rate, 3 the maturity.
double forwardPass(bool isCall, const Market& m, int input)
{
    typedef xad::FReal<double> AD;

    AD spot = m.spot, vol = m.vol, rate = m.rate, maturity = m.maturity;
    AD* inputs[4] = {&spot, &vol, &rate, &maturity};
    derivative(*inputs[input]) = 1.0;

    AD price = blackScholes(isCall, spot, m.strike, vol, rate, maturity);
    return derivative(price);
}

// Forward mode needs one pass per Greek.
Greeks forwardMode(bool isCall, const Market& m)
{
    Greeks g;
    g.price = blackScholes(isCall, m.spot, m.strike, m.vol, m.rate, m.maturity);
    g.delta = forwardPass(isCall, m, 0);
    g.vega = forwardPass(isCall, m, 1);
    g.rho = forwardPass(isCall, m, 2);
    g.theta = -forwardPass(isCall, m, 3);  // dV/dT -> dV/dt
    return g;
}

// Adjoint mode: one recording and one reverse pass give all Greeks at once.
Greeks adjointMode(bool isCall, const Market& m)
{
    typedef xad::AReal<double> AD;
    xad::Tape<double> tape;

    AD spot = m.spot, vol = m.vol, rate = m.rate, maturity = m.maturity;
    tape.registerInput(spot);
    tape.registerInput(vol);
    tape.registerInput(rate);
    tape.registerInput(maturity);
    tape.newRecording();

    AD price = blackScholes(isCall, spot, m.strike, vol, rate, maturity);

    tape.registerOutput(price);
    derivative(price) = 1.0;
    tape.computeAdjoints();

    Greeks g;
    g.price = value(price);
    g.delta = derivative(spot);
    g.vega = derivative(vol);
    g.rho = derivative(rate);
    g.theta = -derivative(maturity);  // dV/dT -> dV/dt
    return g;
}

// Prints one row of the table and returns the largest error against the reference.
double printRow(const std::string& label, const Greeks& g, const Greeks& ref)
{
    std::cout << std::left << std::setw(16) << label << std::right << std::fixed
              << std::setprecision(6) << std::setw(12) << g.price << std::setw(12) << g.delta
              << std::setw(12) << g.vega << std::setw(12) << g.rho << std::setw(12) << g.theta
              << "\n";

    // absolute error, relative for values larger than 1
    double values[5] = {g.price, g.delta, g.vega, g.rho, g.theta};
    double refs[5] = {ref.price, ref.delta, ref.vega, ref.rho, ref.theta};
    double maxError = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        double error = std::abs(values[i] - refs[i]) / std::max(1.0, std::abs(refs[i]));
        maxError = std::max(maxError, error);
    }
    return maxError;
}

int main()
{
    Market m = {100.0, 105.0, 0.2, 0.03, 1.5};
    const double TOLERANCE = 1e-12;

    std::cout << "Black-Scholes, spot = " << m.spot << ", strike = " << m.strike
              << ", vol = " << m.vol << ", rate = " << m.rate << ", maturity = " << m.maturity
              << "\n\n";
    std::cout << std::left << std::setw(16) << "" << std::right << std::setw(12) << "price"
              << std::setw(12) << "delta" << std::setw(12) << "vega" << std::setw(12) << "rho"
              << std::setw(12) << "theta"
              << "\n";

    double maxError = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        bool isCall = (i == 0);
        std::string name = isCall ? "call" : "put";
        Greeks ref = closedForm(isCall, m);

        printRow(name + " exact", ref, ref);
        maxError = std::max(maxError, printRow(name + " forward", forwardMode(isCall, m), ref));
        maxError = std::max(maxError, printRow(name + " adjoint", adjointMode(isCall, m), ref));
    }

    std::cout << "\nlargest difference to the closed form: " << std::scientific
              << std::setprecision(2) << maxError << "\n";
    if (!(maxError <= TOLERANCE))
    {
        std::cerr << "Error: difference exceeds " << TOLERANCE << "\n";
        return 1;
    }
    return 0;
}

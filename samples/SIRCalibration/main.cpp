/*******************************************************************************

   Fits the parameters of an SIR epidemic model to synthetic daily case
   counts. The model is plain C++ loops integrated with fixed-step RK4, and
   its gradient comes from XAD's adjoint mode. The gradient is checked against
   central finite differences.


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

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

const double POPULATION = 1000000.0;
const int DAYS = 150;
const int STEPS_PER_DAY = 4;
const int LOCKDOWN_DAY = 60;
const double LOCKDOWN_FACTOR = 0.5;   // beta is multiplied by this from LOCKDOWN_DAY on
const double NOISE_SD = 200.0;        // standard deviation of the observation noise

// Parameters are fitted as logarithms, so beta, gamma and I0 stay positive.
const int NUM_PARAMS = 3;

// SIR right-hand side: new infections per day and change in the infected count
template <typename T>
void sirDerivatives(const T& beta, const T& gamma, const T& s, const T& i, T& ds, T& di)
{
    T infections = beta * s * i / POPULATION;
    ds = -infections;
    di = infections - gamma * i;
}

// One classical RK4 step of size h on the (susceptible, infected) state
template <typename T>
void rk4Step(const T& beta, const T& gamma, T& s, T& i, double h)
{
    T ks1, ki1, ks2, ki2, ks3, ki3, ks4, ki4;
    T sTmp, iTmp;

    sirDerivatives(beta, gamma, s, i, ks1, ki1);

    sTmp = s + 0.5 * h * ks1;
    iTmp = i + 0.5 * h * ki1;
    sirDerivatives(beta, gamma, sTmp, iTmp, ks2, ki2);

    sTmp = s + 0.5 * h * ks2;
    iTmp = i + 0.5 * h * ki2;
    sirDerivatives(beta, gamma, sTmp, iTmp, ks3, ki3);

    sTmp = s + h * ks3;
    iTmp = i + h * ki3;
    sirDerivatives(beta, gamma, sTmp, iTmp, ks4, ki4);

    s = s + h / 6.0 * (ks1 + 2.0 * ks2 + 2.0 * ks3 + ks4);
    i = i + h / 6.0 * (ki1 + 2.0 * ki2 + 2.0 * ki3 + ki4);
}

// New cases on each day. Beta drops from LOCKDOWN_DAY on, which is a branch on a plain int.
template <typename T>
std::vector<T> simulateDailyCases(const T& beta, const T& gamma, const T& i0)
{
    std::vector<T> cases;
    cases.reserve(DAYS);

    T s = POPULATION - i0;
    T i = i0;
    for (int day = 0; day < DAYS; ++day)
    {
        T betaDay = beta;
        if (day >= LOCKDOWN_DAY)
            betaDay = beta * LOCKDOWN_FACTOR;

        T sStart = s;
        for (int step = 0; step < STEPS_PER_DAY; ++step)
            rk4Step(betaDay, gamma, s, i, 1.0 / STEPS_PER_DAY);

        T dayCases = sStart - s;
        cases.push_back(dayCases);
    }
    return cases;
}

// Mean squared residual against the observed cases, in units of NOISE_SD.
// u holds the logarithms of beta, gamma and I0.
template <typename T>
T modelLoss(const std::vector<T>& u, const std::vector<double>& observed)
{
    using std::exp;

    T beta = exp(u[0]);
    T gamma = exp(u[1]);
    T i0 = exp(u[2]);

    std::vector<T> cases = simulateDailyCases(beta, gamma, i0);

    T loss = 0.0;
    for (std::size_t day = 0; day < observed.size(); ++day)
    {
        T residual = (cases[day] - observed[day]) / NOISE_SD;
        loss = loss + residual * residual;
    }
    return loss / static_cast<double>(observed.size());
}

// Loss and its gradient with respect to u, from one adjoint sweep
double lossAndGradient(const std::vector<double>& u, const std::vector<double>& observed,
                       std::vector<double>& grad)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;
    std::vector<AD> uAd(NUM_PARAMS);
    for (int j = 0; j < NUM_PARAMS; ++j)
    {
        uAd[j] = u[j];
        tape.registerInput(uAd[j]);
    }

    tape.newRecording();

    AD loss = modelLoss(uAd, observed);

    tape.registerOutput(loss);
    derivative(loss) = 1.0;
    tape.computeAdjoints();

    for (int j = 0; j < NUM_PARAMS; ++j)
        grad[j] = derivative(uAd[j]);

    return xad::value(loss);
}

int main()
{
    const double TRUE_BETA = 0.35;
    const double TRUE_GAMMA = 0.1;
    const double TRUE_I0 = 20.0;

    // generate observed cases from the true parameters plus noise, with a fixed seed
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> trueCases = simulateDailyCases<double>(TRUE_BETA, TRUE_GAMMA, TRUE_I0);
    std::vector<double> observed(DAYS);
    for (int day = 0; day < DAYS; ++day)
        observed[day] = trueCases[day] + NOISE_SD * gauss(rng);

    std::cout << "Observed cases: " << DAYS << " days, population " << POPULATION
              << ", lockdown from day " << LOCKDOWN_DAY << "\n\n";

    // ---- adjoint gradient checked against central finite differences ----

    std::vector<double> u0 = {std::log(0.2), std::log(0.2), std::log(10.0)};
    std::vector<double> grad(NUM_PARAMS);
    double loss0 = lossAndGradient(u0, observed, grad);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "initial guess: loss = " << loss0 << "\n";
    std::cout << std::setw(6) << "param" << std::setw(16) << "adjoint grad"
              << std::setw(16) << "FD grad" << std::setw(16) << "rel. diff" << "\n";

    bool gradientsMatch = std::isfinite(loss0);
    const double FD_STEP = 1e-5;
    const double FD_TOLERANCE = 1e-5;
    for (int j = 0; j < NUM_PARAMS; ++j)
    {
        std::vector<double> uPlus = u0;
        std::vector<double> uMinus = u0;
        uPlus[j] += FD_STEP;
        uMinus[j] -= FD_STEP;

        double fdG = (modelLoss<double>(uPlus, observed) - modelLoss<double>(uMinus, observed))
                     / (2.0 * FD_STEP);
        double adjG = grad[j];
        double relDiff = std::abs(adjG - fdG) / (std::abs(adjG) + 1e-15);

        std::cout << std::setw(6) << j << std::setw(16) << adjG << std::setw(16) << fdG
                  << std::scientific << std::setprecision(3) << std::setw(16) << relDiff
                  << std::fixed << std::setprecision(6) << "\n";

        if (!std::isfinite(adjG) || !std::isfinite(fdG) || relDiff > FD_TOLERANCE)
            gradientsMatch = false;
    }
    std::cout << "\n";

    if (!gradientsMatch)
    {
        std::cerr << "finite-difference gradient check failed\n";
        return 1;
    }

    // ---- gradient descent with backtracking line search ----

    std::cout << "gradient descent on log-parameters\n";

    std::vector<double> u = u0;
    const int MAX_ITERATIONS = 20000;
    const double GRAD_TOLERANCE = 1e-10;
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        double loss = lossAndGradient(u, observed, grad);

        double gradNormSq = 0.0;
        for (int j = 0; j < NUM_PARAMS; ++j)
            gradNormSq += grad[j] * grad[j];
        if (gradNormSq < GRAD_TOLERANCE)
            break;

        // halve the step until the loss decreases enough
        double step = 1.0;
        std::vector<double> trial(NUM_PARAMS);
        while (step > 1e-12)
        {
            for (int j = 0; j < NUM_PARAMS; ++j)
                trial[j] = u[j] - step * grad[j];

            if (modelLoss<double>(trial, observed) <= loss - 1e-4 * step * gradNormSq)
                break;
            step *= 0.5;
        }
        u = trial;

        if (iter % 4000 == 0)
            std::cout << "iter " << std::setw(4) << iter << "  loss = " << loss << "\n";
    }

    std::cout << "\n" << std::setw(10) << "param" << std::setw(12) << "true"
              << std::setw(12) << "fitted" << "\n";
    const double trueParams[NUM_PARAMS] = {TRUE_BETA, TRUE_GAMMA, TRUE_I0};
    const char* names[NUM_PARAMS] = {"beta", "gamma", "I0"};
    for (int j = 0; j < NUM_PARAMS; ++j)
        std::cout << std::setw(10) << names[j] << std::setw(12) << trueParams[j]
                  << std::setw(12) << std::exp(u[j]) << "\n";

    std::cout << "\nfinal loss = " << modelLoss<double>(u, observed) << "\n";

    return 0;
}

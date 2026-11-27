/*******************************************************************************

   Maximum-likelihood estimation of an AR(1)-plus-noise linear state-space
   model using the Kalman filter and XAD's 1st-order adjoint mode.

   The mean negative log-likelihood and its gradient with respect to all
   unconstrained parameters are computed in a single reverse pass, handling
   missing observations by skipping the measurement update branch. The adjoint
   gradient is verified against central finite differences, and parameters are
   recovered using a stack-allocated BFGS optimizer with backtracking line
   search.


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

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

const int NUM_PARAMS = 3;
const int NUM_STEPS = 500;
const double LOG_TWO_PI = 1.83787706640934548356;

typedef std::array<double, NUM_PARAMS> ParamVec;

// Mean negative log-likelihood of an AR(1)-plus-noise state-space model:
//   x_t = phi * x_{t-1} + eta_t,   eta_t ~ N(0, sigmaEta^2)
//   y_t = x_t + eps_t,             eps_t ~ N(0, sigmaEps^2)
//
// Unconstrained parameters u:
//   u[0] -> phi = tanh(u[0]) in (-1, 1)    (guarantees stationarity)
//   u[1] -> sigmaEta^2 = exp(u[1]) > 0     (state noise variance)
//   u[2] -> sigmaEps^2 = exp(u[2]) > 0     (observation noise variance)
template <typename T>
T negativeLogLikelihood(const std::array<T, NUM_PARAMS>& u,
                        const std::vector<double>& y,
                        const std::vector<unsigned char>& isMissing,
                        int numObserved)
{
    using std::exp;
    using std::log;
    using std::tanh;

    // Hoist loop-invariant parameter transformations out of the time recursion
    const T phi = tanh(u[0]);
    const T phiSq = phi * phi;
    const T sigmaEtaSq = exp(u[1]);
    const T sigmaEpsSq = exp(u[2]);

    // Stationary prior distribution for the initial state x_0
    T xHat = 0.0;
    T p = sigmaEtaSq / (1.0 - phiSq);

    T nllSum = 0.0;
    const std::size_t n = y.size();
    for (std::size_t t = 0; t < n; ++t)
    {
        // Time update (prediction)
        const T xPred = phi * xHat;
        const T pPred = phiSq * p + sigmaEtaSq;

        // Branch on passive missingness mask: when missing, skip measurement update
        if (!isMissing[t])
        {
            const T v = y[t] - xPred;
            const T f = pPred + sigmaEpsSq;
            const T invF = 1.0 / f;

            // Measurement update: p = (pPred * sigmaEpsSq) / f preserves positivity
            // without catastrophic cancellation in (1 - K) * pPred
            xHat = xPred + (pPred * invF) * v;
            p = (pPred * sigmaEpsSq) * invF;

            // Accumulate log(f) + v^2 / f; 0.5 and log(2*pi) are folded after the loop
            nllSum = nllSum + log(f) + (v * v) * invF;
        }
        else
        {
            xHat = xPred;
            p = pPred;
        }
    }

    return 0.5 * (nllSum + static_cast<double>(numObserved) * LOG_TWO_PI) /
           static_cast<double>(numObserved);
}

// Compute mean negative log-likelihood and its gradient in one adjoint sweep.
// Reuses the caller's tape and uses stack-allocated arrays for zero heap allocations.
double nllAndGradient(xad::adj<double>::tape_type& tape,
                      const ParamVec& u,
                      const std::vector<double>& y,
                      const std::vector<unsigned char>& isMissing,
                      int numObserved,
                      ParamVec& grad)
{
    typedef xad::adj<double>::active_type AD;

    tape.clearAll();

    std::array<AD, NUM_PARAMS> uAd;
    for (int j = 0; j < NUM_PARAMS; ++j)
    {
        uAd[j] = u[j];
        tape.registerInput(uAd[j]);
    }

    tape.newRecording();

    AD loss = negativeLogLikelihood(uAd, y, isMissing, numObserved);

    tape.registerOutput(loss);
    derivative(loss) = 1.0;
    tape.computeAdjoints();

    for (int j = 0; j < NUM_PARAMS; ++j)
        grad[j] = derivative(uAd[j]);

    return xad::value(loss);
}

double dotProduct(const ParamVec& a, const ParamVec& b)
{
    double sum = 0.0;
    for (int i = 0; i < NUM_PARAMS; ++i)
        sum += a[i] * b[i];
    return sum;
}

}  // namespace

int main()
{
    const double TRUE_PHI = 0.85;
    const double TRUE_SIGMA_ETA_SQ = 1.00;
    const double TRUE_SIGMA_EPS_SQ = 0.25;

    // Simulate AR(1)-plus-noise observations with a fixed seed
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    const double sigmaEta = std::sqrt(TRUE_SIGMA_ETA_SQ);
    const double sigmaEps = std::sqrt(TRUE_SIGMA_EPS_SQ);
    const double initSd = std::sqrt(TRUE_SIGMA_ETA_SQ / (1.0 - TRUE_PHI * TRUE_PHI));

    std::vector<double> y(NUM_STEPS);
    std::vector<unsigned char> isMissing(NUM_STEPS, 0);

    double x = initSd * gauss(rng);
    int numMissing = 0;
    for (int t = 0; t < NUM_STEPS; ++t)
    {
        x = TRUE_PHI * x + sigmaEta * gauss(rng);
        y[t] = x + sigmaEps * gauss(rng);

        // Include both a contiguous missing gap and scattered missing points
        if ((t >= 100 && t < 105) || uniform(rng) < 0.04)
        {
            isMissing[t] = 1;
            ++numMissing;
        }
    }
    const int numObserved = NUM_STEPS - numMissing;

    std::cout << "Kalman filter MLE (AR(1) + noise)\n";
    std::cout << "Observations: " << NUM_STEPS << " steps (" << numObserved << " observed, "
              << numMissing << " missing)\n\n";

    // Starting guess away from the true parameters
    const ParamVec u0 = {{std::atanh(0.40), std::log(1.50), std::log(0.30)}};
    ParamVec grad = {{0.0, 0.0, 0.0}};

    // Allocate one tape and reuse it across all gradient evaluations
    xad::adj<double>::tape_type tape;
    const double loss0 = nllAndGradient(tape, u0, y, isMissing, numObserved, grad);

    // ---- Finite-difference gradient check at initial guess ----
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "initial guess: mean NLL = " << loss0 << "\n";
    std::cout << std::setw(14) << "param" << std::setw(16) << "adjoint grad"
              << std::setw(16) << "FD grad" << std::setw(16) << "rel. diff" << "\n";

    const char* const PARAM_NAMES[NUM_PARAMS] = {"atanh(phi)", "log(sigEta^2)", "log(sigEps^2)"};
    bool gradientsMatch = std::isfinite(loss0);
    const double FD_STEP = 1e-5;
    const double FD_TOLERANCE = 1e-5;

    for (int j = 0; j < NUM_PARAMS; ++j)
    {
        ParamVec uPlus = u0;
        ParamVec uMinus = u0;
        uPlus[j] += FD_STEP;
        uMinus[j] -= FD_STEP;

        const double fPlus = negativeLogLikelihood<double>(uPlus, y, isMissing, numObserved);
        const double fMinus = negativeLogLikelihood<double>(uMinus, y, isMissing, numObserved);
        const double fdG = (fPlus - fMinus) / (2.0 * FD_STEP);
        const double adjG = grad[j];
        const double relDiff = std::abs(adjG - fdG) / (std::abs(adjG) + 1e-15);

        std::cout << std::setw(14) << PARAM_NAMES[j] << std::setw(16) << adjG << std::setw(16)
                  << fdG << std::scientific << std::setprecision(3) << std::setw(16) << relDiff
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

    // ---- BFGS minimization with backtracking Armijo line search ----
    std::cout << "BFGS optimization (maximizing log-likelihood)\n";

    ParamVec u = u0;
    double loss = loss0;

    // Initialize inverse Hessian approximation H = I on the stack
    double invH[NUM_PARAMS][NUM_PARAMS] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    const int MAX_ITERATIONS = 100;
    const double GRAD_TOL = 1e-8;
    const double ARMIJO_C1 = 1e-4;

    int iter = 0;
    bool converged = false;
    for (; iter < MAX_ITERATIONS; ++iter)
    {
        const double gradNorm = std::sqrt(dotProduct(grad, grad));
        if (iter % 5 == 0 || gradNorm < GRAD_TOL)
            std::cout << "iter " << std::setw(3) << iter << "  mean NLL = " << loss
                      << "  |grad| = " << std::scientific << std::setprecision(3) << gradNorm
                      << std::fixed << std::setprecision(6) << "\n";

        if (gradNorm < GRAD_TOL)
        {
            converged = true;
            break;
        }

        // Search direction d = -H * grad
        ParamVec dir = {{0.0, 0.0, 0.0}};
        for (int i = 0; i < NUM_PARAMS; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < NUM_PARAMS; ++j)
                sum += invH[i][j] * grad[j];
            dir[i] = -sum;
        }

        // Guard against non-descent direction by resetting H = I
        double directionalDeriv = dotProduct(grad, dir);
        if (directionalDeriv >= 0.0)
        {
            for (int i = 0; i < NUM_PARAMS; ++i)
            {
                for (int j = 0; j < NUM_PARAMS; ++j)
                    invH[i][j] = (i == j) ? 1.0 : 0.0;
                dir[i] = -grad[i];
            }
            directionalDeriv = -dotProduct(grad, grad);
        }

        // Backtracking Armijo line search
        double alpha = 1.0;
        ParamVec uNext = u;
        double lossNext = loss;
        bool stepAccepted = false;
        while (alpha > 1e-12)
        {
            for (int i = 0; i < NUM_PARAMS; ++i)
                uNext[i] = u[i] + alpha * dir[i];

            lossNext = negativeLogLikelihood<double>(uNext, y, isMissing, numObserved);
            if (std::isfinite(lossNext) && lossNext <= loss + ARMIJO_C1 * alpha * directionalDeriv)
            {
                stepAccepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!stepAccepted)
            break;

        ParamVec gradNext = {{0.0, 0.0, 0.0}};
        lossNext = nllAndGradient(tape, uNext, y, isMissing, numObserved, gradNext);

        // BFGS inverse Hessian update:
        // H_{k+1} = (I - rho * s * y^T) * H_k * (I - rho * y * s^T) + rho * s * s^T
        ParamVec sVec;
        ParamVec yVec;
        for (int i = 0; i < NUM_PARAMS; ++i)
        {
            sVec[i] = uNext[i] - u[i];
            yVec[i] = gradNext[i] - grad[i];
        }

        const double sy = dotProduct(sVec, yVec);
        const double sNormSq = dotProduct(sVec, sVec);
        const double yNormSq = dotProduct(yVec, yVec);
        if (sy > 1e-12 * std::sqrt(sNormSq * yNormSq))
        {
            const double rho = 1.0 / sy;
            ParamVec hy = {{0.0, 0.0, 0.0}};
            for (int i = 0; i < NUM_PARAMS; ++i)
                for (int j = 0; j < NUM_PARAMS; ++j)
                    hy[i] += invH[i][j] * yVec[j];

            const double yHy = dotProduct(yVec, hy);
            const double scale = (1.0 + rho * yHy) * rho;

            for (int i = 0; i < NUM_PARAMS; ++i)
                for (int j = 0; j < NUM_PARAMS; ++j)
                    invH[i][j] += scale * sVec[i] * sVec[j] -
                                  rho * (sVec[i] * hy[j] + hy[i] * sVec[j]);
        }

        u = uNext;
        grad = gradNext;
        loss = lossNext;
    }

    if (!converged)
    {
        std::cerr << "BFGS did not converge within " << MAX_ITERATIONS << " iterations\n";
        return 1;
    }

    const double fittedPhi = std::tanh(u[0]);
    const double fittedSigmaEtaSq = std::exp(u[1]);
    const double fittedSigmaEpsSq = std::exp(u[2]);

    std::cout << "\nconverged in " << iter << " iterations\n";
    std::cout << std::setw(14) << "param" << std::setw(12) << "true" << std::setw(12) << "fitted"
              << "\n";
    std::cout << std::setw(14) << "phi" << std::setw(12) << TRUE_PHI << std::setw(12) << fittedPhi
              << "\n";
    std::cout << std::setw(14) << "sigmaEta^2" << std::setw(12) << TRUE_SIGMA_ETA_SQ
              << std::setw(12) << fittedSigmaEtaSq << "\n";
    std::cout << std::setw(14) << "sigmaEps^2" << std::setw(12) << TRUE_SIGMA_EPS_SQ
              << std::setw(12) << fittedSigmaEpsSq << "\n";
    std::cout << "\nfinal log-likelihood = " << (-loss * static_cast<double>(numObserved)) << "\n";

    return 0;
}

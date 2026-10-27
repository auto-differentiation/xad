/*******************************************************************************

   Demonstrates binary logistic regression with XAD's adjoint mode on a
   fixed-seed synthetic dataset. It compares two gradients with finite
   differences and shows the loss falling during gradient descent.


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

// Mean binary cross-entropy for both passive and active weights.
template <typename T>
T computeLoss(const std::vector<T>& w, const std::vector<std::vector<double>>& x,
              const std::vector<int>& y)
{
    using std::exp;
    using std::log1p;

    T total = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i)
    {
        T z = w.back();  // bias
        for (std::size_t j = 0; j < x[i].size(); ++j)
            z = z + w[j] * x[i][j];

        // Evaluate softplus(-z) without overflow when z is very negative.
        if (xad::value(z) >= 0.0)
            total = total + log1p(exp(-z)) + (1.0 - y[i]) * z;
        else
            total = total + log1p(exp(z)) - y[i] * z;
    }
    return total / static_cast<double>(y.size());
}

int main()
{
    const int N_POINTS   = 1000;
    const int N_FEATURES = 10;

    // tape and active data type for 1st order adjoint computation
    typedef xad::adj<double>  mode;
    typedef mode::tape_type   tape_type;
    typedef mode::active_type AD;

    // generate dataset with a fixed seed for reproducibility
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // random true weights used only to produce linearly-separable labels
    std::vector<double> wTrue(N_FEATURES + 1);
    for (int j = 0; j <= N_FEATURES; ++j)
        wTrue[j] = gauss(rng);

    std::vector<std::vector<double>> features(N_POINTS, std::vector<double>(N_FEATURES));
    std::vector<int> y(N_POINTS);

    for (int i = 0; i < N_POINTS; ++i)
    {
        double z = wTrue[N_FEATURES];  // bias
        for (int j = 0; j < N_FEATURES; ++j)
        {
            features[i][j] = gauss(rng);
            z += wTrue[j] * features[i][j];
        }
        y[i] = (z > 0.0) ? 1 : 0;
    }

    std::cout << "Dataset: " << N_POINTS << " points, " << N_FEATURES << " features\n\n";

    // ---- adjoint sweep at w = 0 ----
    // A single reverse pass gives the loss and all D+1 gradients at once.

    std::vector<double> w0(N_FEATURES + 1, 0.0);
    std::vector<double> grad(N_FEATURES + 1);
    tape_type tape;

    {
        // set independent variables and register them
        std::vector<AD> w(N_FEATURES + 1);
        for (int j = 0; j <= N_FEATURES; ++j)
        {
            w[j] = w0[j];
            tape.registerInput(w[j]);
        }

        // start recording derivatives
        tape.newRecording();

        AD loss = computeLoss(w, features, y);

        // register output, seed adjoint, and compute all input adjoints
        tape.registerOutput(loss);
        derivative(loss) = 1.0;
        tape.computeAdjoints();

        // collect gradients
        for (int j = 0; j <= N_FEATURES; ++j)
            grad[j] = derivative(w[j]);

        // output results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "adjoint sweep at w = 0\n";
        std::cout << "loss = " << value(loss) << "\n";
        std::cout << "gradients dL/dw:\n";
        for (int j = 0; j <= N_FEATURES; ++j)
            std::cout << "  w[" << std::setw(2) << j << "] = " << std::setw(12) << grad[j] << "\n";
        std::cout << "\n";
    }

    // ---- finite-difference check for w[0] and w[1] ----
    // Central differences: (f(w+FD_STEP) - f(w-FD_STEP)) / (2*FD_STEP)

    std::cout << "finite-difference check\n";
    std::cout << std::setw(6)  << "index"
              << std::setw(16) << "adjoint grad"
              << std::setw(16) << "FD grad"
              << std::setw(16) << "rel. diff"
              << "\n";

    bool gradientsMatch = true;
    const double FD_STEP = 1e-5;
    const double FD_TOLERANCE = 1e-5;
    for (int idx : {0, 1})
    {
        // evaluate loss with weight idx perturbed up and down
        double lossPlus  = 0.0;
        double lossMinus = 0.0;
        for (int pm = -1; pm <= 1; pm += 2)
        {
            std::vector<double> wPerturbed = w0;
            wPerturbed[idx] += pm * FD_STEP;

            double lossVal = computeLoss(wPerturbed, features, y);

            if (pm == 1)
                lossPlus = lossVal;
            else
                lossMinus = lossVal;
        }

        double fdG     = (lossPlus - lossMinus) / (2.0 * FD_STEP);
        double adjG    = grad[idx];
        double relDiff = std::abs(adjG - fdG) / (std::abs(adjG) + 1e-15);

        std::cout << std::setw(6)  << idx
                  << std::fixed << std::setprecision(6)
                  << std::setw(16) << adjG
                  << std::setw(16) << fdG
                  << std::scientific << std::setprecision(3)
                  << std::setw(16) << relDiff
                  << "\n";

        if (!std::isfinite(adjG) || !std::isfinite(fdG) || relDiff > FD_TOLERANCE)
            gradientsMatch = false;
    }
    std::cout << "\n";

    if (!gradientsMatch)
    {
        std::cerr << "finite-difference gradient check failed\n";
        return 1;
    }

    // ---- gradient descent (lr = 0.1, 20 iterations) ----
    // Loss should fall steadily at each step.

    std::cout << "gradient descent (lr = 0.1, 20 iterations)\n";
    std::cout << std::fixed << std::setprecision(6);

    std::vector<double> wGd(N_FEATURES + 1, 0.0);
    const double LEARNING_RATE = 0.1;

    for (int iter = 0; iter < 20; ++iter)
    {
        tape.clearAll();

        // set independent variables and register them
        std::vector<AD> w(N_FEATURES + 1);
        for (int j = 0; j <= N_FEATURES; ++j)
        {
            w[j] = wGd[j];
            tape.registerInput(w[j]);
        }

        // start recording derivatives
        tape.newRecording();

        AD loss = computeLoss(w, features, y);

        // seed adjoint and compute all input adjoints
        tape.registerOutput(loss);
        derivative(loss) = 1.0;
        tape.computeAdjoints();

        std::cout << "iter " << std::setw(2) << iter
                  << "  loss = " << std::setw(10) << value(loss) << "\n";

        // gradient-descent update
        for (int j = 0; j <= N_FEATURES; ++j)
            wGd[j] -= LEARNING_RATE * derivative(w[j]);
    }

    return 0;
}

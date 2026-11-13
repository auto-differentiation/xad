/*******************************************************************************

   Solves the 1D heat equation
     du/dt = alpha * d^2u/dx^2,   x in [0, 1],   u fixed at both ends,
   with an explicit finite-difference scheme over many time steps, and computes
   the gradient of the final temperature at the midpoint with respect to the
   whole initial temperature profile in one adjoint pass.

   The adjoint is computed with and without checkpointing, showing that
   checkpointing keeps the tape memory bounded by one stage of time steps.
   A few gradient entries are checked against finite differences, and the
   gradient profile is written to heat_gradient.csv.


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

#include "heat_checkpoint.hpp"

#include <XAD/XAD.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

typedef xad::adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;

const double PI = 3.14159265358979323846;

const int NX = 201;                    // grid points, dx = 1 / (NX - 1)
const double LAMBDA = 0.4;             // alpha * dt / dx^2, stable for <= 0.5
const int N_STEPS = 2000;              // time steps
const int CHECKPOINT_INTERVAL = 50;    // time steps per checkpointed stage
const std::size_t MID = (NX - 1) / 2;  // index of the midpoint

// Passive objective: temperature at the midpoint after N_STEPS time steps.
double finalMidpoint(std::vector<double> u)
{
    heatSteps(u, LAMBDA, N_STEPS);
    return u[MID];
}

// Records all time steps on the tape and computes the gradient in one adjoint pass.
double gradientFullTape(const std::vector<double>& u0, std::vector<double>& gradient,
                        std::size_t& tapeMemory)
{
    tape_type tape;

    std::vector<AD> inputs(u0.begin(), u0.end());
    for (std::size_t i = 0; i < inputs.size(); ++i) tape.registerInput(inputs[i]);
    tape.newRecording();

    std::vector<AD> u(inputs);
    heatSteps(u, LAMBDA, N_STEPS);
    AD y = u[MID];

    tape.registerOutput(y);
    derivative(y) = 1.0;
    tapeMemory = tape.getMemory();
    tape.computeAdjoints();

    for (std::size_t i = 0; i < inputs.size(); ++i) gradient[i] = derivative(inputs[i]);
    return value(y);
}

// Runs the time steps in stages with checkpoints, so that only one stage is on the tape
// at any time, and computes the same gradient.
double gradientCheckpointed(const std::vector<double>& u0, std::vector<double>& gradient,
                            std::size_t& tapeMemory, std::size_t& peakMemory,
                            std::size_t& checkpointMemory)
{
    tape_type tape;

    std::vector<AD> inputs(u0.begin(), u0.end());
    for (std::size_t i = 0; i < inputs.size(); ++i) tape.registerInput(inputs[i]);
    tape.newRecording();

    HeatCheckpointCallback<tape_type> checkpoint(LAMBDA);
    std::vector<AD> u(inputs);
    for (int step = 0; step < N_STEPS; step += CHECKPOINT_INTERVAL)
        checkpoint.computeStage(u, std::min(CHECKPOINT_INTERVAL, N_STEPS - step));
    AD y = u[MID];

    tape.registerOutput(y);
    derivative(y) = 1.0;
    tapeMemory = tape.getMemory();
    checkpointMemory = checkpoint.getCheckpointMemory();
    tape.computeAdjoints();
    peakMemory = std::max(tapeMemory, checkpoint.getPeakMemory());

    for (std::size_t i = 0; i < inputs.size(); ++i) gradient[i] = derivative(inputs[i]);
    return value(y);
}

void writeCsv(const char* filename, const std::vector<double>& u0, const std::vector<double>& uT,
              const std::vector<double>& gradient)
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Warning: could not open " << filename << " for writing\n";
        return;
    }
    out << "x,initial,final,gradient\n" << std::setprecision(12);
    for (std::size_t i = 0; i < u0.size(); ++i)
        out << double(i) / (NX - 1) << "," << u0[i] << "," << uT[i] << "," << gradient[i] << "\n";
    std::cout << "Gradient profile written to " << filename << "\n";
}

int main()
{
    std::vector<double> u0(NX);
    for (int i = 0; i < NX; ++i)
    {
        double x = double(i) / (NX - 1);
        u0[i] = std::sin(PI * x) + 0.5 * std::sin(4.0 * PI * x);
    }

    std::cout << "1D heat equation: " << NX << " grid points, " << N_STEPS
              << " explicit time steps, lambda = " << LAMBDA << "\n"
              << "Objective: temperature at x = 0.5 at the final time\n\n";

    std::vector<double> gradFull(NX), gradChk(NX);
    std::size_t memFull = 0, memChk = 0, peakChk = 0, memCheckpoints = 0;
    double yFull = gradientFullTape(u0, gradFull, memFull);
    double yChk = gradientCheckpointed(u0, gradChk, memChk, peakChk, memCheckpoints);

    std::cout << std::fixed << std::setprecision(10) << "u(0.5, T) = " << yFull << "\n\n";

    std::cout << std::setprecision(1) << "Tape memory without checkpointing: " << std::setw(9)
              << memFull / 1024.0 << " KiB\n"
              << "Tape memory with checkpointing (every " << CHECKPOINT_INTERVAL << " steps):\n"
              << "  after the forward pass:          " << std::setw(9) << memChk / 1024.0
              << " KiB\n"
              << "  peak during the adjoint pass:    " << std::setw(9) << peakChk / 1024.0
              << " KiB\n"
              << "  plus checkpoint storage:         " << std::setw(9) << memCheckpoints / 1024.0
              << " KiB\n\n";

    double maxDiff = std::abs(yFull - yChk);
    for (int i = 0; i < NX; ++i) maxDiff = std::max(maxDiff, std::abs(gradFull[i] - gradChk[i]));
    std::cout << std::scientific << std::setprecision(2)
              << "Largest difference with / without checkpointing: " << maxDiff << "\n\n";

    // central finite differences for a few entries
    const double FD_STEP = 1e-4;
    const double FD_TOLERANCE = 1e-8;
    const int checkIndices[] = {20, 70, int(MID), 120, 180};
    double maxFdError = 0.0;
    std::cout << std::setw(8) << "index" << std::setw(18) << "adjoint" << std::setw(18)
              << "finite diff." << std::setw(12) << "abs. diff."
              << "\n";
    for (int k = 0; k < 5; ++k)
    {
        int i = checkIndices[k];
        std::vector<double> up(u0), down(u0);
        up[i] += FD_STEP;
        down[i] -= FD_STEP;
        double fd = (finalMidpoint(up) - finalMidpoint(down)) / (2.0 * FD_STEP);
        double error = std::abs(fd - gradChk[i]);
        maxFdError = std::max(maxFdError, error);
        std::cout << std::setw(8) << i << std::setprecision(10) << std::setw(18) << gradChk[i]
                  << std::setw(18) << fd << std::setprecision(2) << std::setw(12) << error << "\n";
    }
    std::cout << "\n";

    std::vector<double> uT(u0);
    heatSteps(uT, LAMBDA, N_STEPS);
    writeCsv("heat_gradient.csv", u0, uT, gradChk);

    if (!(maxDiff <= 1e-12) || !(maxFdError <= FD_TOLERANCE) || !(peakChk < memFull))
    {
        std::cerr << "Error: gradient or memory check failed\n";
        return 1;
    }
    return 0;
}

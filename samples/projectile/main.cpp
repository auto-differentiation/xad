/*******************************************************************************

   Simulates 2D projectile motion with gravity and air drag using a fixed-step
   integrator and optimizes the launch angle and speed using XAD adjoint mode
   to hit a target point. Compares adjoint gradients with finite differences
   and exports trajectories across iterations to an SVG file.


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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Point
{
    double x;
    double y;
};

// Fixed-step simulation of a ball under gravity and quadratic air drag.
// Returns the squared distance (loss) to the target at the final step.
template <typename T>
T simulateProjectile(T v0, T theta, double targetX, double targetY, double dragCoeff,
                     double gravity, double dt, int nSteps,
                     std::vector<Point>* trajectory = nullptr)
{
    using std::cos;
    using std::sin;
    using std::sqrt;

    T x = 0.0;
    T y = 0.0;
    T vx = v0 * cos(theta);
    T vy = v0 * sin(theta);

    if (trajectory)
    {
        trajectory->clear();
        trajectory->push_back({xad::value(x), xad::value(y)});
    }

    for (int step = 0; step < nSteps; ++step)
    {
        T speed = sqrt(vx * vx + vy * vy + 1e-12);
        T ax = -dragCoeff * speed * vx;
        T ay = -gravity - dragCoeff * speed * vy;

        vx = vx + ax * dt;
        vy = vy + ay * dt;
        x = x + vx * dt;
        y = y + vy * dt;

        if (trajectory && (step % 10 == 0 || step == nSteps - 1))
        {
            trajectory->push_back({xad::value(x), xad::value(y)});
        }
    }

    T dx = x - targetX;
    T dy = y - targetY;
    return dx * dx + dy * dy;
}

// Write trajectories of selected iterations to an SVG file.
void exportSvg(const std::string& filename,
               const std::vector<std::pair<int, std::vector<Point>>>& iterations, double targetX,
               double targetY)
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Warning: Could not open " << filename << " for writing\n";
        return;
    }

    const double width = 800.0;
    const double height = 480.0;
    const double marginLeft = 70.0;
    const double marginRight = 40.0;
    const double marginTop = 40.0;
    const double marginBottom = 60.0;

    double minX = 0.0;
    double maxX = std::max(targetX * 1.15, 90.0);
    double minY = -5.0;
    double maxY = 25.0;

    for (const auto& item : iterations)
    {
        for (const auto& pt : item.second)
        {
            if (pt.x > maxX)
                maxX = pt.x * 1.05;
            if (pt.y > maxY)
                maxY = pt.y * 1.1;
            if (pt.y < minY)
                minY = pt.y * 1.1;
        }
    }

    auto toSvgX = [&](double x)
    { return marginLeft + (x - minX) / (maxX - minX) * (width - marginLeft - marginRight); };
    auto toSvgY = [&](double y)
    {
        return height - marginBottom -
               (y - minY) / (maxY - minY) * (height - marginTop - marginBottom);
    };

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "  <rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>\n";

    // Draw coordinate axes
    double groundY = toSvgY(0.0);
    out << "  <line x1=\"" << marginLeft << "\" y1=\"" << groundY << "\" x2=\""
        << (width - marginRight) << "\" y2=\"" << groundY
        << "\" stroke=\"#94a3b8\" stroke-width=\"1.5\" stroke-dasharray=\"4,4\"/>\n";
    out << "  <text x=\"" << (width - marginRight) << "\" y=\"" << (groundY + 16.0)
        << "\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#64748b\" text-anchor=\"end\">"
        << "Ground (y=0)</text>\n";

    // Target marker
    double targetSvgX = toSvgX(targetX);
    double targetSvgY = toSvgY(targetY);
    out << "  <circle cx=\"" << targetSvgX << "\" cy=\"" << targetSvgY
        << "\" r=\"7\" fill=\"none\" stroke=\"#dc2626\" stroke-width=\"2.5\"/>\n";
    out << "  <circle cx=\"" << targetSvgX << "\" cy=\"" << targetSvgY
        << "\" r=\"2\" fill=\"#dc2626\"/>\n";
    out << "  <text x=\"" << targetSvgX << "\" y=\"" << (targetSvgY - 12.0)
        << "\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"bold\" fill=\"#dc2626\" "
        << "text-anchor=\"middle\">Target (" << targetX << ", " << targetY << ")</text>\n";

    const std::vector<std::string> colors = {"#94a3b8", "#f97316", "#eab308",
                                             "#06b6d4", "#3b82f6", "#16a34a"};

    // Draw trajectories
    for (std::size_t i = 0; i < iterations.size(); ++i)
    {
        const auto& iterData = iterations[i];
        std::string color = colors[i % colors.size()];
        double strokeWidth = (i == iterations.size() - 1) ? 2.5 : 1.5;

        out << "  <polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"" << strokeWidth
            << "\" points=\"";
        for (const auto& pt : iterData.second)
        {
            out << toSvgX(pt.x) << "," << toSvgY(pt.y) << " ";
        }
        out << "\"/>\n";
    }

    // Legend
    double legendX = width - marginRight - 160.0;
    double legendY = marginTop + 10.0;
    out << "  <rect x=\"" << legendX << "\" y=\"" << legendY << "\" width=\"155\" height=\""
        << (iterations.size() * 20.0 + 20.0)
        << "\" fill=\"#ffffff\" stroke=\"#cbd5e1\" rx=\"6\" opacity=\"0.95\"/>\n";
    out << "  <text x=\"" << (legendX + 10.0) << "\" y=\"" << (legendY + 16.0)
        << "\" font-family=\"sans-serif\" font-size=\"11\" font-weight=\"bold\" fill=\"#334155\">"
        << "Optimization Progress</text>\n";

    for (std::size_t i = 0; i < iterations.size(); ++i)
    {
        double itemY = legendY + 36.0 + i * 20.0;
        std::string color = colors[i % colors.size()];
        out << "  <line x1=\"" << (legendX + 10.0) << "\" y1=\"" << (itemY - 3.0) << "\" x2=\""
            << (legendX + 30.0) << "\" y2=\"" << (itemY - 3.0) << "\" stroke=\"" << color
            << "\" stroke-width=\"2.5\"/>\n";
        out << "  <text x=\"" << (legendX + 36.0) << "\" y=\"" << itemY
            << "\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#475569\">Iter "
            << iterations[i].first << "</text>\n";
    }

    out << "</svg>\n";
    std::cout << "Exported trajectory visualization to " << filename << "\n";
}

int main()
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    const double GRAVITY = 9.81;
    const double DRAG_COEFF = 0.01;
    const double DT = 0.004;
    const int N_STEPS = 1000;  // total simulation duration = 4.0 seconds

    const double TARGET_X = 80.0;
    const double TARGET_Y = 0.0;

    // Initial guess for projectile launch parameters (will miss target)
    double v0Init = 45.0;
    double thetaInit = 40.0 * (M_PI / 180.0);  // 40 degrees

    std::cout << "=== XAD Projectile Aiming Optimization ===\n";
    std::cout << "Simulation: fixed-step integrator, " << N_STEPS << " steps, dt = " << DT << "s\n";
    std::cout << "Target: (" << TARGET_X << ", " << TARGET_Y << ")\n";
    std::cout << "Initial guess: v0 = " << v0Init << " m/s, theta = " << (thetaInit * 180.0 / M_PI)
              << " deg\n\n";

    tape_type tape;

    // ---- 1. Adjoint Gradient Computation ----
    AD v0Active = v0Init;
    AD thetaActive = thetaInit;

    tape.registerInput(v0Active);
    tape.registerInput(thetaActive);
    tape.newRecording();

    AD lossActive = simulateProjectile(v0Active, thetaActive, TARGET_X, TARGET_Y, DRAG_COEFF,
                                       GRAVITY, DT, N_STEPS);

    tape.registerOutput(lossActive);
    derivative(lossActive) = 1.0;
    tape.computeAdjoints();

    double adjGradV0 = derivative(v0Active);
    double adjGradTheta = derivative(thetaActive);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial loss = " << value(lossActive) << "\n";
    std::cout << "Adjoint gradients:\n";
    std::cout << "  dL/dv0     = " << std::setw(12) << adjGradV0 << "\n";
    std::cout << "  dL/dtheta  = " << std::setw(12) << adjGradTheta << "\n\n";

    // ---- 2. Finite-Difference Check ----
    // Central differences: (f(p + h) - f(p - h)) / (2 * h)
    const double FD_STEP = 1e-6;
    const double FD_TOLERANCE = 1e-4;

    double lossV0Plus = simulateProjectile(v0Init + FD_STEP, thetaInit, TARGET_X, TARGET_Y,
                                           DRAG_COEFF, GRAVITY, DT, N_STEPS);
    double lossV0Minus = simulateProjectile(v0Init - FD_STEP, thetaInit, TARGET_X, TARGET_Y,
                                            DRAG_COEFF, GRAVITY, DT, N_STEPS);
    double fdGradV0 = (lossV0Plus - lossV0Minus) / (2.0 * FD_STEP);

    double lossThPlus = simulateProjectile(v0Init, thetaInit + FD_STEP, TARGET_X, TARGET_Y,
                                           DRAG_COEFF, GRAVITY, DT, N_STEPS);
    double lossThMinus = simulateProjectile(v0Init, thetaInit - FD_STEP, TARGET_X, TARGET_Y,
                                            DRAG_COEFF, GRAVITY, DT, N_STEPS);
    double fdGradTheta = (lossThPlus - lossThMinus) / (2.0 * FD_STEP);

    double relDiffV0 = std::abs(adjGradV0 - fdGradV0) / (std::abs(adjGradV0) + 1e-15);
    double relDiffTh = std::abs(adjGradTheta - fdGradTheta) / (std::abs(adjGradTheta) + 1e-15);

    std::cout << "Finite-difference check:\n";
    std::cout << std::setw(10) << "param" << std::setw(16) << "adjoint grad" << std::setw(16)
              << "FD grad" << std::setw(16) << "rel. diff"
              << "\n";
    std::cout << std::setw(10) << "v0" << std::setw(16) << adjGradV0 << std::setw(16) << fdGradV0
              << std::scientific << std::setprecision(3) << std::setw(16) << relDiffV0 << "\n";
    std::cout << std::fixed << std::setprecision(6) << std::setw(10) << "theta" << std::setw(16)
              << adjGradTheta << std::setw(16) << fdGradTheta << std::scientific
              << std::setprecision(3) << std::setw(16) << relDiffTh << "\n\n";

    if (!std::isfinite(adjGradV0) || !std::isfinite(adjGradTheta) || relDiffV0 > FD_TOLERANCE ||
        relDiffTh > FD_TOLERANCE)
    {
        std::cerr << "Error: Finite-difference gradient validation failed!\n";
        return 1;
    }
    std::cout << "Finite-difference validation passed successfully.\n\n";

    // ---- 3. Gradient Descent Optimization ----
    std::cout << "Running gradient descent optimization (25 iterations):\n";
    std::cout << std::fixed << std::setprecision(4);

    double v0Current = v0Init;
    double thetaCurrent = thetaInit;
    const double LR_V = 0.002;
    const double LR_TH = 0.00005;

    std::vector<std::pair<int, std::vector<Point>>> trajectories;

    for (int iter = 0; iter < 25; ++iter)
    {
        tape.clearAll();

        AD v0 = v0Current;
        AD theta = thetaCurrent;
        tape.registerInput(v0);
        tape.registerInput(theta);

        tape.newRecording();

        std::vector<Point> traj;
        AD loss = simulateProjectile(v0, theta, TARGET_X, TARGET_Y, DRAG_COEFF, GRAVITY, DT,
                                     N_STEPS, &traj);

        tape.registerOutput(loss);
        derivative(loss) = 1.0;
        tape.computeAdjoints();

        if (iter % 5 == 0 || iter == 24)
        {
            trajectories.push_back({iter, traj});
            const Point& finalPt = traj.back();
            std::cout << "Iter " << std::setw(2) << iter << ": loss = " << std::setw(8)
                      << value(loss) << ", final pos = (" << std::setw(6) << finalPt.x << ", "
                      << std::setw(5) << finalPt.y << ")"
                      << ", v0 = " << std::setw(5) << v0Current << " m/s, theta = " << std::setw(5)
                      << (thetaCurrent * 180.0 / M_PI) << " deg\n";
        }

        // Gradient descent step
        v0Current -= LR_V * derivative(v0);
        thetaCurrent -= LR_TH * derivative(theta);
    }

    std::cout << "\n";
    exportSvg("trajectory.svg", trajectories, TARGET_X, TARGET_Y);

    std::cout << "\nSample completed successfully.\n";
    return 0;
}

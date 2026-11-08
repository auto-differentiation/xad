/*******************************************************************************

   Demonstrates XAD's adjoint mode with Eigen matrices. It computes the
   gradient of a least-squares residual solved with ColPivHouseholderQR and
   the gradient of log|det(A)| via PartialPivLU, and checks both against
   finite differences (and log|det(A)| against its analytical gradient).


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

// StdCompatibility puts XAD's math functions into namespace std, which Eigen relies on
#include <XAD/StdCompatibility.hpp>
#include <XAD/XAD.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Squared residual ||A x - b||^2 of the least-squares solution x, for passive and active types.
template <typename T>
T computeLeastSquaresLoss(const Mat<T>& A, const Vec<T>& b)
{
    Vec<T> x = A.colPivHouseholderQr().solve(b);
    Vec<T> r = A * x - b;
    return r.squaredNorm();
}

// log|det(A)| as the sum of log|U_ii| from an LU factorisation, which avoids the overflow and
// underflow of forming det(A) itself.
template <typename T>
T computeLogAbsDet(const Mat<T>& A)
{
    using std::abs;
    using std::log;

    Eigen::PartialPivLU<Mat<T>> lu(A);
    T result = 0.0;
    for (Eigen::Index i = 0; i < A.rows(); ++i) result += log(abs(lu.matrixLU()(i, i)));
    return result;
}

// Central finite differences of the scalar function f with respect to every entry of x.
template <typename F, typename MatrixType>
MatrixType computeFiniteDifferences(F f, MatrixType x)
{
    const double FD_STEP = 1e-6;
    MatrixType grad(x.rows(), x.cols());
    for (Eigen::Index j = 0; j < x.cols(); ++j)
    {
        for (Eigen::Index i = 0; i < x.rows(); ++i)
        {
            const double xij = x(i, j);
            x(i, j) = xij + FD_STEP;
            const double fPlus = f(x);
            x(i, j) = xij - FD_STEP;
            const double fMinus = f(x);
            x(i, j) = xij;
            grad(i, j) = (fPlus - fMinus) / (2.0 * FD_STEP);
        }
    }
    return grad;
}

// Prints the largest error |ad - ref| / (1 + |ref|) and returns whether it is within tolerance.
bool checkGradient(const std::string& label, const Mat<double>& ad, const Mat<double>& ref,
                   double tolerance)
{
    bool finite = true;
    double maxError = 0.0;
    for (Eigen::Index j = 0; j < ad.cols(); ++j)
    {
        for (Eigen::Index i = 0; i < ad.rows(); ++i)
        {
            if (!std::isfinite(ad(i, j)) || !std::isfinite(ref(i, j)))
                finite = false;
            else
                maxError = std::max(maxError,
                                    std::abs(ad(i, j) - ref(i, j)) / (1.0 + std::abs(ref(i, j))));
        }
    }

    const bool ok = finite && maxError <= tolerance;
    std::cout << "  " << std::left << std::setw(36) << label << std::right << "max error "
              << std::scientific << std::setprecision(2) << maxError << "  (tolerance " << tolerance
              << ")  " << (ok ? "OK" : "FAILED") << "\n";
    return ok;
}

int main()
{
    // tape and active data type for 1st order adjoint computation
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    const double FD_TOLERANCE = 1e-6;
    const double ANALYTIC_TOLERANCE = 1e-10;
    const Eigen::IOFormat MATRIX_FORMAT(6, 0, "  ", "\n", "    [", "]");

    bool gradientsMatch = true;
    tape_type tape;

    // ---- least squares: L(A, b) = ||A x - b||^2 with x = argmin ||A x - b|| ----
    // A is overdetermined and b is not in its column space, so the residual is non-zero.

    Mat<double> A0(5, 3);
    // clang-format off
    A0 <<  4.0,  1.0,  0.5,
           2.0,  3.0,  1.0,
           1.0, -1.0,  2.0,
           3.0,  0.5, -1.0,
          -2.0,  2.0,  1.5;
    // clang-format on
    Vec<double> b0(5);
    b0 << 1.0, 2.0, 3.0, 4.0, 5.0;

    Mat<double> gradA(A0.rows(), A0.cols());
    Vec<double> gradB(b0.size());
    {
        // set independent variables and register them
        Mat<AD> A = A0.cast<AD>();
        Vec<AD> b = b0.cast<AD>();
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            for (Eigen::Index i = 0; i < A.rows(); ++i) tape.registerInput(A(i, j));
        for (Eigen::Index i = 0; i < b.size(); ++i) tape.registerInput(b(i));

        // start recording derivatives
        tape.newRecording();

        AD loss = computeLeastSquaresLoss(A, b);

        // register output, seed adjoint, and compute all input adjoints
        tape.registerOutput(loss);
        derivative(loss) = 1.0;
        tape.computeAdjoints();

        for (Eigen::Index j = 0; j < A.cols(); ++j)
            for (Eigen::Index i = 0; i < A.rows(); ++i) gradA(i, j) = derivative(A(i, j));
        for (Eigen::Index i = 0; i < b.size(); ++i) gradB(i) = derivative(b(i));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "least squares, A is 5x3, solved with ColPivHouseholderQR\n";
        std::cout << "  x = " << A0.colPivHouseholderQr().solve(b0).transpose() << "\n";
        std::cout << "  ||A x - b||^2 = " << value(loss) << "\n";
        std::cout << "  dL/dA =\n" << gradA.format(MATRIX_FORMAT) << "\n";
        std::cout << "  dL/db =\n" << gradB.transpose().format(MATRIX_FORMAT) << "\n\n";
    }

    const Mat<double> fdA = computeFiniteDifferences(
        [&b0](const Mat<double>& A) { return computeLeastSquaresLoss(A, b0); }, A0);
    const Vec<double> fdB = computeFiniteDifferences(
        [&A0](const Vec<double>& b) { return computeLeastSquaresLoss(A0, b); }, b0);

    std::cout << "least-squares checks\n";
    gradientsMatch &= checkGradient("dL/dA vs finite differences", gradA, fdA, FD_TOLERANCE);
    gradientsMatch &= checkGradient("dL/db vs finite differences", gradB, fdB, FD_TOLERANCE);
    std::cout << "\n";

    // ---- log-determinant: f(M) = log|det(M)|, whose gradient is inverse(M)^T ----

    tape.clearAll();

    Mat<double> M0(4, 4);
    // clang-format off
    M0 << 4.0,  1.0,  0.5, 2.0,
          1.0,  5.0, -1.0, 0.5,
          0.3,  2.0,  6.0, 1.0,
          1.5, -0.7,  1.0, 3.0;
    // clang-format on

    Mat<double> gradM(M0.rows(), M0.cols());
    {
        // set independent variables and register them
        Mat<AD> M = M0.cast<AD>();
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            for (Eigen::Index i = 0; i < M.rows(); ++i) tape.registerInput(M(i, j));

        // start recording derivatives
        tape.newRecording();

        AD logDet = computeLogAbsDet(M);

        // register output, seed adjoint, and compute all input adjoints
        tape.registerOutput(logDet);
        derivative(logDet) = 1.0;
        tape.computeAdjoints();

        for (Eigen::Index j = 0; j < M.cols(); ++j)
            for (Eigen::Index i = 0; i < M.rows(); ++i) gradM(i, j) = derivative(M(i, j));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "log-determinant, M is 4x4, factorised with PartialPivLU\n";
        std::cout << "  log|det(M)| = " << value(logDet) << "\n";
        std::cout << "  d log|det(M)| / dM =\n" << gradM.format(MATRIX_FORMAT) << "\n\n";
    }

    const Mat<double> analyticM = M0.inverse().transpose();
    const Mat<double> fdM =
        computeFiniteDifferences([](const Mat<double>& M) { return computeLogAbsDet(M); }, M0);

    std::cout << "log-determinant checks\n";
    gradientsMatch &=
        checkGradient("gradient vs inverse(M)^T", gradM, analyticM, ANALYTIC_TOLERANCE);
    gradientsMatch &= checkGradient("gradient vs finite differences", gradM, fdM, FD_TOLERANCE);

    if (!gradientsMatch)
    {
        std::cerr << "gradient check failed\n";
        return 1;
    }
    return 0;
}

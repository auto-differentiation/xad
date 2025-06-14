/*******************************************************************************

   Tests for the compatibility and correctness of XAD with Eigen types.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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
#include <XAD/EigenCompatibility.hpp>

#include <gtest/gtest.h>

TEST(Eigen, MatrixInverseAdj)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 2, 2> A0;
    A0 << 2.0, 1.0,
          1.0, 3.0;

    Eigen::Matrix<double, 2, 2> numerical_grad;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = A_plus.inverse().sum();
            double f_minus = A_minus.inverse().sum();

            numerical_grad(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = A0(i, j);

    tape_type tape;
    tape.registerInputs(A.reshaped().begin(), A.reshaped().end());

    tape.newRecording();

    Eigen::Matrix<AD, 2, 2> B = A.inverse();

    for (AD &Bi : B.reshaped())
    {
        tape.registerOutput(Bi);
        derivative(Bi) = 1.0;
    }

    tape.computeAdjoints();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(A(i, j));
            double fd = numerical_grad(i, j);
            EXPECT_NEAR(ad, fd, 1e-5) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(Eigen, MatrixInverseFwd)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 2, 2> A0;
    A0 << 2.0, 1.0,
          1.0, 3.0;

    Eigen::Matrix<double, 2, 2> numerical_grad;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = A_plus.inverse().sum();
            double f_minus = A_minus.inverse().sum();

            numerical_grad(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = A0(i, j);

    for (AD &Ai : A.reshaped())
    {
        derivative(Ai) = 1.0;
    }

    Eigen::Matrix<AD, 2, 2> B = A.inverse();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(B(i, j));
            double fd = numerical_grad(i, j);
            EXPECT_NEAR(ad, fd, 1e-5) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(Eigen, MatrixMultiplicationAdj)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    constexpr double eps = 1e-6;

    Eigen::Matrix<double, 2, 2> A0;
    Eigen::Matrix<double, 2, 2> B0;

    A0 << 1.0, 2.0,
          3.1, 4.5;
    B0 << 2.09, 0.0,
          1.13, 2.0;

    Eigen::Matrix<double, 2, 2> numerical_grad_A;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = (A_plus * B0).sum();
            double f_minus = (A_minus * B0).sum();

            numerical_grad_A(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A, B, C;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            A(i, j) = A0(i, j);
            B(i, j) = B0(i, j);
        }

    tape_type tape;
    tape.registerInputs(A.reshaped().begin(), A.reshaped().end());
    tape.registerInputs(B.reshaped().begin(), B.reshaped().end());
    tape.newRecording();

    C = A * B;

    for (AD &Ci : C.reshaped()) {
        tape.registerOutput(Ci);
        derivative(Ci) = 1.0;
    }

    tape.computeAdjoints();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(A(i, j));
            double fd = numerical_grad_A(i, j);
            EXPECT_NEAR(ad, fd, 1e-5) << "Mismatch at A(" << i << ", " << j << ")";
        }
    }
}

TEST(Eigen, MatrixTraceAdj)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 2, 2> A0;
    A0 << 1.0, 2.2,
          3.0, 4.1;

    Eigen::Matrix<double, 2, 2> numerical_grad;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = A_plus.trace();
            double f_minus = A_minus.trace();

            numerical_grad(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = A0(i, j);

    tape_type tape;
    tape.registerInputs(A.reshaped().begin(), A.reshaped().end());
    tape.newRecording();

    AD trace = A.trace();

    tape.registerOutput(trace);
    derivative(trace) = 1.0;

    tape.computeAdjoints();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(A(i, j));
            double fd = numerical_grad(i, j);
            EXPECT_NEAR(ad, fd, 1e-7) << "Mismatch at A(" << i << ", " << j << ")";
        }
    }
}
TEST(Eigen, MatrixDeterminantAdj)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 2, 2> A0;
    A0 << 1.0, 5.6,
          3.1, 4.0;

    Eigen::Matrix<double, 2, 2> numerical_grad;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = A_plus.determinant();
            double f_minus = A_minus.determinant();

            numerical_grad(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = A0(i, j);
    tape_type tape;

    tape.registerInputs(A.reshaped().begin(), A.reshaped().end())
    ;
    tape.newRecording();

    AD det = A.determinant();

    tape.registerOutput(det);

    derivative(det) = 1.0;

    tape.computeAdjoints();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(A(i, j));
            double fd = numerical_grad(i, j);
            EXPECT_NEAR(ad, fd, 1e-7) << "Mismatch at A(" << i << ", " << j << ")";
        }
    }
}

TEST(Eigen, MatrixNormAdj)
{
    using mode = xad::adj<double>;
    using tape_type = mode::tape_type;
    using AD = mode::active_type;

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 2, 2> A0;
    A0 << 1.0, 5.6,
          3.1, 4.0;

    Eigen::Matrix<double, 2, 2> numerical_grad;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Eigen::Matrix<double, 2, 2> A_plus = A0;
            Eigen::Matrix<double, 2, 2> A_minus = A0;
            A_plus(i, j) += eps;
            A_minus(i, j) -= eps;

            double f_plus = A_plus.norm();
            double f_minus = A_minus.norm();

            numerical_grad(i, j) = (f_plus - f_minus) / (2 * eps);
        }
    }

    Eigen::Matrix<AD, 2, 2> A;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            A(i, j) = A0(i, j);

    tape_type tape;
    tape.registerInputs(A.reshaped().begin(), A.reshaped().end());
    tape.newRecording();

    AD norm = A.norm();

    tape.registerOutput(norm);
    derivative(norm) = 1.0;

    tape.computeAdjoints();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double ad = derivative(A(i, j));
            double fd = numerical_grad(i, j);
            EXPECT_NEAR(ad, fd, 1e-7) << "Mismatch at A(" << i << ", " << j << ")";
        }
    }
}
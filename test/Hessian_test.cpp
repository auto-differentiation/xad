/*******************************************************************************

   Tests for xad::Hessian class.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <complex>
#include <type_traits>
#include <typeinfo>
#include <vector>

using namespace ::testing;

template <class T>
T quad(std::vector<T> &x)
{
    T c = x[0] * x[0];
    T d = x[1] * x[1];
    return c + d;
}

template <class T>
T tquad(std::vector<T> &x)
{
    T c = x[0] * x[0];
    T d = x[1] * x[1];
    T e = x[2] * x[2];
    return c + d + e;
}

template <class T>
T foo(std::vector<T> &x)
{
    T c = exp(x[0]);
    T d = sin(x[1]);
    T e = cos(x[2]);
    return c + d + e;
}

template <class T>
T single(std::vector<T> &x)
{
    return x[0] * x[0] * x[0] + x[0];
}

TEST(HessianTest, QuadraticForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3, 2};
    xad::Hessian<AD> hes(quad<AD>, x, &tape);

    std::vector<std::vector<AD>> cross_hessian = {{2.0, 0.0}, {0.0, 2.0}};
    std::vector<std::vector<AD>> computed_hessian = hes.get();

    for (unsigned int i = 0; i < cross_hessian.size(); i++)
        for (unsigned int j = 0; j < cross_hessian.size(); j++)
            ASSERT_EQ(cross_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, SingleInputForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3};
    xad::Hessian<AD> hes(single<AD>, x, &tape);

    std::vector<std::vector<AD>> cross_hessian = {{18.0}};
    std::vector<std::vector<AD>> computed_hessian = hes.get();

    for (unsigned int i = 0; i < cross_hessian.size(); i++)
        for (unsigned int j = 0; j < cross_hessian.size(); j++)
            ASSERT_EQ(cross_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3, 2};
    xad::Hessian<AD> hes(quad<AD>, x);

    std::vector<std::vector<AD>> cross_hessian = {{2.0, 0.0}, {0.0, 2.0}};
    std::vector<std::vector<AD>> computed_hessian = hes.get();

    for (unsigned int i = 0; i < cross_hessian.size(); i++)
        for (unsigned int j = 0; j < cross_hessian.size(); j++)
            ASSERT_EQ(cross_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, SingleInputForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3};
    xad::Hessian<AD> hes(single<AD>, x);

    std::vector<std::vector<AD>> cross_hessian = {{18.0}};
    std::vector<std::vector<AD>> computed_hessian = hes.get();

    for (unsigned int i = 0; i < cross_hessian.size(); i++)
        for (unsigned int j = 0; j < cross_hessian.size(); j++)
            ASSERT_EQ(cross_hessian[i][j], computed_hessian[i][j]);
}

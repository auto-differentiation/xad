/*******************************************************************************

   Tests for hessian methods.

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

#include <XAD/Hessian.hpp>
#include <XAD/XAD.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <complex>
#include <functional>
#include <list>
#include <type_traits>
#include <typeinfo>
#include <vector>

using namespace ::testing;

TEST(HessianTest, QuadraticForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<AD>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticForwardAdjointWithIterator)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::list<std::list<AD>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    std::list<std::list<AD>> computed_hessian(x.size(), std::list<AD>(x.size(), 0.0));
    xad::computeHessian<decltype(begin(computed_hessian)), double>(x, foo, begin(computed_hessian),
                                                                   end(computed_hessian), &tape);

    auto row1 = computed_hessian.begin(), row2 = expected_hessian.begin();
    while (row1 != computed_hessian.end() && row2 != expected_hessian.end())
    {
        auto col1 = row1->begin(), col2 = row2->begin();
        while (col1 != row1->end() && col2 != row2->end())
        {
            ASSERT_EQ(*col1, *col2);
            col1++;
            col2++;
        }
        row1++;
        row2++;
    }
}

TEST(HessianTest, SingleInputForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3.0};

    // f(x) = x[0]^3 + x[0]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] * x[0] + x[0]; };

    std::vector<std::vector<AD>> expected_hessian = {{18.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<AD>> expected_hessian = {{2.0, 0.0},  //
                                                     {0.0, 2.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticForwardForwardWithIterator)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::list<std::list<AD>> expected_hessian = {{2.0, 0.0},  //
                                                 {0.0, 2.0}};

    std::list<std::list<AD>> computed_hessian(x.size(), std::list<AD>(x.size(), 0.0));
    xad::computeHessian<decltype(begin(computed_hessian)), double>(x, foo, begin(computed_hessian),
                                                                   end(computed_hessian));

    auto row1 = computed_hessian.begin(), row2 = expected_hessian.begin();
    while (row1 != computed_hessian.end() && row2 != expected_hessian.end())
    {
        auto col1 = row1->begin(), col2 = row2->begin();
        while (col1 != row1->end() && col2 != row2->end())
        {
            ASSERT_EQ(*col1, *col2);
            col1++;
            col2++;
        }
        row1++;
        row2++;
    }
}

TEST(HessianTest, SingleInputForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3.0};

    // f(x) = x[0]^3 + x[0]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] * x[0] + x[0]; };

    std::vector<std::vector<AD>> expected_hessian = {{18.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticThreeVariablesForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0};

    // f(x) = x[0]^2 + x[1]^2 + x[2]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; };

    std::vector<std::vector<AD>> expected_hessian = {
        {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, ComplexFunctionForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = x[0] * sin(x[1]) + x[2] * exp(x[3])
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * sin(x[1]) + x[2] * exp(x[3]); };

    std::vector<std::vector<AD>> expected_hessian = {{0.0, cos(x[1]), 0.0, 0.0},
                                                     {cos(x[1]), -x[0] * sin(x[1]), 0.0, 0.0},
                                                     {0.0, 0.0, 0.0, exp(x[3])},
                                                     {0.0, 0.0, exp(x[3]), x[2] * exp(x[3])}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, FourthOrderPolynomialForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0};

    // f(x) = x[0]^4 + x[1]^4 + x[2]^4
    auto foo = [](std::vector<AD> &x) -> AD { return pow(x[0], 4) + pow(x[1], 4) + pow(x[2], 4); };

    std::vector<std::vector<AD>> expected_hessian = {{12.0 * x[0] * x[0], 0.0, 0.0},
                                                     {0.0, 12.0 * x[1] * x[1], 0.0},
                                                     {0.0, 0.0, 12.0 * x[2] * x[2]}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, HigherOrderInteractionForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0};

    // f(x) = x[0] * x[1] * x[2]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[1] * x[2]; };

    std::vector<std::vector<AD>> expected_hessian = {{0.0, x[2], x[1]},  //
                                                     {x[2], 0.0, x[0]},
                                                     {x[1], x[0], 0.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticFourVariablesForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2
    auto foo = [](std::vector<AD> &x) -> AD
    { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]; };

    std::vector<std::vector<AD>> expected_hessian = {{2.0, 0.0, 0.0, 0.0},  //
                                                     {0.0, 2.0, 0.0, 0.0},
                                                     {0.0, 0.0, 2.0, 0.0},
                                                     {0.0, 0.0, 0.0, 2.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, LargeHessianForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x(16);
    for (size_t i = 0; i < 16; ++i) x[i] = static_cast<double>(i + 1);

    // f(x) = sum(x[i]^2) + sum(x[i] * x[j]), i < j
    auto foo = [](std::vector<AD> &x) -> AD
    {
        AD result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) result += x[i] * x[i];
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = i + 1; j < x.size(); ++j) result += x[i] * x[j];
        return result;
    };

    std::vector<std::vector<AD>> expected_hessian(16, std::vector<AD>(16, 1.0));
    for (size_t i = 0; i < 16; ++i) expected_hessian[i][i] = 2.0;

    auto computed_hessian = xad::computeHessian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, LargeHessianForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x(16);
    for (size_t i = 0; i < 16; ++i) x[i] = static_cast<double>(i + 1);

    // f(x) = sum(x[i]^2) + sum(x[i] * x[j]), i < j
    auto foo = [](std::vector<AD> &x) -> AD
    {
        AD result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) result += x[i] * x[i];
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = i + 1; j < x.size(); ++j) result += x[i] * x[j];
        return result;
    };

    std::vector<std::vector<AD>> expected_hessian(16, std::vector<AD>(16, 1.0));
    for (size_t i = 0; i < 16; ++i) expected_hessian[i][i] = 2.0;

    auto computed_hessian = xad::computeHessian<double>(x, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

TEST(HessianTest, QuadraticForwardAdjointAutoTape)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<AD>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    auto computed_hessian = xad::computeHessian<double>(x, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        for (unsigned int j = 0; j < expected_hessian[i].size(); j++)
            ASSERT_EQ(expected_hessian[i][j], computed_hessian[i][j]);
}

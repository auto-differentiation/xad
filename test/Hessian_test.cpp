/*******************************************************************************

   Tests for Hessian methods.

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

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<double>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticForwardAdjointAutoTape)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<double>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticForwardAdjointFetchTape)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<double>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticForwardAdjointWithIterator)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::list<std::list<double>> expected_hessian = {{2.0, 0.0}, {0.0, 2.0}};

    std::list<std::list<double>> computed_hessian(input.size(), std::list<double>(input.size(), 0.0));
    xad::computeHessian<decltype(begin(computed_hessian)), double>(input, foo, begin(computed_hessian),
                                                                   end(computed_hessian), &tape);

    auto expected_it = expected_hessian.begin();
    auto computed_it = computed_hessian.begin();
    for (; expected_it != expected_hessian.end() && computed_it != computed_hessian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(HessianTest, SingleInputForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {3.0};

    // f(x) = x[0]^3 + x[0]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] * x[0] + x[0]; };

    std::vector<std::vector<double>> expected_hessian = {{18.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::vector<std::vector<double>> expected_hessian = {{2.0, 0.0},  //
                                                         {0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticForwardForwardWithIterator)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input = {3.0, 2.0};

    // f(x) = x[0]^2 + x[1]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1]; };

    std::list<std::list<double>> expected_hessian = {{2.0, 0.0},  //
                                                     {0.0, 2.0}};

    std::list<std::list<double>> computed_hessian(input.size(), std::list<double>(input.size(), 0.0));
    xad::computeHessian<decltype(begin(computed_hessian)), double>(input, foo, begin(computed_hessian),
                                                                   end(computed_hessian));

    auto expected_it = expected_hessian.begin();
    auto computed_it = computed_hessian.begin();
    for (; expected_it != expected_hessian.end() && computed_it != computed_hessian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(HessianTest, SingleInputForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input = {3.0};

    // f(x) = x[0]^3 + x[0]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] * x[0] + x[0]; };

    std::vector<std::vector<double>> expected_hessian = {{18.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticThreeVariablesForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {1.0, 2.0, 3.0};

    // f(x) = x[0]^2 + x[1]^2 + x[2]^2
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; };

    std::vector<std::vector<double>> expected_hessian = {
        {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, ComplexFunctionForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {1.0, 2.0, 3.0, 4.0};

    // f(x) = x[0] * sin(x[1]) + x[2] * exp(x[3])
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * sin(x[1]) + x[2] * exp(x[3]); };

    std::vector<std::vector<double>> expected_hessian = {
        {0.0, cos(value(value(input[1]))), 0.0, 0.0},
        {cos(value(value(input[1]))), -value(value(input[0])) * sin(value(value(input[1]))), 0.0, 0.0},
        {0.0, 0.0, 0.0, exp(value(value(input[3])))},
        {0.0, 0.0, exp(value(value(input[3]))), value(value(input[2])) * exp(value(value(input[3])))}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, FourthOrderPolynomialForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {1.0, 2.0, 3.0};

    // f(x) = x[0]^4 + x[1]^4 + x[2]^4
    auto foo = [](std::vector<AD> &x) -> AD { return pow(x[0], 4) + pow(x[1], 4) + pow(x[2], 4); };

    std::vector<std::vector<double>> expected_hessian = {
        {12.0 * value(value(input[0])) * value(value(input[0])), 0.0, 0.0},
        {0.0, 12.0 * value(value(input[1])) * value(value(input[1])), 0.0},
        {0.0, 0.0, 12.0 * value(value(input[2])) * value(value(input[2]))}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, HigherOrderInteractionForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {1.0, 2.0, 3.0};

    // f(x) = x[0] * x[1] * x[2]
    auto foo = [](std::vector<AD> &x) -> AD { return x[0] * x[1] * x[2]; };

    std::vector<std::vector<double>> expected_hessian = {
        {0.0, value(value(input[2])), value(value(input[1]))},  //
        {value(value(input[2])), 0.0, value(value(input[0]))},
        {value(value(input[1])), value(value(input[0])), 0.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, QuadraticFourVariablesForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input = {1.0, 2.0, 3.0, 4.0};

    // f(x) = x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2
    auto foo = [](std::vector<AD> &x) -> AD
    { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]; };

    std::vector<std::vector<double>> expected_hessian = {{2.0, 0.0, 0.0, 0.0},  //
                                                         {0.0, 2.0, 0.0, 0.0},
                                                         {0.0, 0.0, 2.0, 0.0},
                                                         {0.0, 0.0, 0.0, 2.0}};

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, LargeHessianForwardAdjoint)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> input(16);
    for (size_t i = 0; i < 16; ++i) input[i] = static_cast<double>(i + 1);

    // f(x) = sum(x[i]^2) + sum(x[i] * x[j]), i < j
    auto foo = [](std::vector<AD> &x) -> AD
    {
        AD result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) result += x[i] * x[i];
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = i + 1; j < x.size(); ++j) result += x[i] * x[j];
        return result;
    };

    std::vector<std::vector<double>> expected_hessian(16, std::vector<double>(16, 1.0));
    for (size_t i = 0; i < 16; ++i) expected_hessian[i][i] = 2.0;

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo, &tape);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, LargeHessianForwardForward)
{
    typedef xad::fwd_fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input(16);
    for (size_t i = 0; i < 16; ++i) input[i] = static_cast<double>(i + 1);

    // f(x) = sum(x[i]^2) + sum(x[i] * x[j]), i < j
    auto foo = [](std::vector<AD> &x) -> AD
    {
        AD result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) result += x[i] * x[i];
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = i + 1; j < x.size(); ++j) result += x[i] * x[j];
        return result;
    };

    std::vector<std::vector<double>> expected_hessian(16, std::vector<double>(16, 1.0));
    for (size_t i = 0; i < 16; ++i) expected_hessian[i][i] = 2.0;

    std::vector<std::vector<double>> computed_hessian = xad::computeHessian<double>(input, foo);

    for (unsigned int i = 0; i < expected_hessian.size(); i++)
        EXPECT_THAT(computed_hessian[i], Pointwise(DoubleEq(), expected_hessian[i]));
}

TEST(HessianTest, OutOfBoundsDomainSizeMismatch)
{
    typedef xad::fwd_adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> input = {1.0, 2.0};

    auto func = [](std::vector<AD> &x) -> AD { return x[0]; };

    std::vector<std::vector<double>> jacobian(2, std::vector<double>(3));

    auto launch = [](std::vector<AD> x,
                     std::function<xad::AReal<xad::FReal<double>>(
                         std::vector<xad::AReal<xad::FReal<double>>> &)>
                         foo,
                     std::vector<std::vector<double>>::iterator first,
                     std::vector<std::vector<double>>::iterator last)
    {
        using RowIterator = decltype(first);
        xad::computeHessian<RowIterator, double>(x, foo, first, last);
    };

    EXPECT_THROW(launch(input, func, begin(jacobian), end(jacobian)), xad::OutOfRange);
}
/*******************************************************************************

   Tests for jacobian computation methods.

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

#define _USE_MATH_DEFINES
#include <XAD/Jacobian.hpp>
#include <XAD/XAD.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <functional>
#include <list>
#include <numeric>
#include <vector>

using namespace ::testing;

TEST(JacobianTest, SimpleAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {3.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + sin(v[1]), v[1] + sin(v[0])}; };

    std::vector<std::vector<double>> expected_jacobian = {{1.0, cos(value(x[1]))},  //
                                                          {cos(value(x[0])), 1.0}};

    std::vector<std::vector<double>> computed_jacobian =
        xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, SimpleAdjointIteratorAutoTape)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {3.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + sin(v[1]), v[1] + sin(v[0])}; };

    std::list<std::list<double>> expected_jacobian = {{1.0, cos(value(x[1]))},  //
                                                      {cos(value(x[0])), 1.0}};

    std::list<std::list<double>> computed_jacobian(2, std::list<double>(2, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto expected_it = expected_jacobian.begin();
    auto computed_it = computed_jacobian.begin();
    for (; expected_it != expected_jacobian.end() && computed_it != computed_jacobian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(JacobianTest, SimpleAdjointIteratorFetchTape)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;
    typedef mode::tape_type tape_type;

    tape_type tape;

    std::vector<AD> x = {3.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + sin(v[1]), v[1] + sin(v[0])}; };

    std::list<std::list<double>> expected_jacobian = {{1.0, cos(value(x[1]))},  //
                                                      {cos(value(x[0])), 1.0}};

    std::list<std::list<double>> computed_jacobian(2, std::list<double>(2, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto expected_it = expected_jacobian.begin();
    auto computed_it = computed_jacobian.begin();
    for (; expected_it != expected_jacobian.end() && computed_it != computed_jacobian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(JacobianTest, SimpleForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {-2.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + sin(v[1]), v[1] + sin(v[0])}; };

    std::vector<std::vector<double>> expected_jacobian = {{1.0, cos(value(x[1]))},  //
                                                          {cos(value(x[0])), 1.0}};

    std::vector<std::vector<double>> computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, SimpleForwardIterator)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {-2.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + sin(v[1]), v[1] + sin(v[0])}; };

    std::list<std::list<double>> expected_jacobian = {{1.0, cos(value(x[1]))},  //
                                                      {cos(value(x[0])), 1.0}};

    std::list<std::list<double>> computed_jacobian(2, std::list<double>(2, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto expected_it = expected_jacobian.begin();
    auto computed_it = computed_jacobian.begin();
    for (; expected_it != expected_jacobian.end() && computed_it != computed_jacobian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(JacobianTest, ComplexFunctionAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] * x[1], x[2] * exp(x[3]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] * v[1], v[2] * exp(v[3])}; };

    std::vector<std::vector<double>> expected_jacobian = {
        {value(x[1]), value(x[0]), 0.0, 0.0},
        {0.0, 0.0, exp(value(x[3])), value(x[2]) * exp(value(x[3]))}};

    std::vector<std::vector<double>> computed_jacobian =
        xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, DomainLargerThanCodomainForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] + x[1], x[2] * x[3] ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {v[0] + v[1], v[2] * v[3]}; };

    std::vector<std::vector<double>> expected_jacobian = {{1.0, 1.0, 0.0, 0.0},  //
                                                          {0.0, 0.0, value(x[3]), value(x[2])}};

    std::vector<std::vector<double>> computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, DomainLargerThanCodomainAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] + x[1], x[2] * x[3] ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {v[0] + v[1], v[2] * v[3]}; };

    std::vector<std::vector<double>> expected_jacobian = {{1.0, 1.0, 0.0, 0.0},  //
                                                          {0.0, 0.0, value(x[3]), value(x[2])}};

    std::vector<std::vector<double>> computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, DomainSmallerThanCodomainAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {2.0, 3.0};

    // f(x) = [ x[0] + x[1], x[0] - x[1], x[0] * x[1] ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + v[1], v[0] - v[1], v[0] * v[1]}; };

    std::vector<std::vector<double>> expected_jacobian = {{1.0, 1.0},  //
                                                          {1.0, -1.0},
                                                          {value(x[1]), value(x[0])}};

    std::vector<std::vector<double>> computed_jacobian =
        xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, ComplexDomainNotEqualCodomainForwardIterator)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0};

    // f(x) = [ x[0] + x[1], x[1] * x[2], exp(x[0]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD>
    { return {v[0] + v[1], v[1] * v[2], exp(v[0])}; };

    std::list<std::list<double>> expected_jacobian = {{1.0, 1.0, 0.0},  //
                                                      {0.0, value(x[2]), value(x[1])},
                                                      {exp(value(x[0])), 0.0, 0.0}};

    std::list<std::list<double>> computed_jacobian(3, std::list<double>(3, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto expected_it = expected_jacobian.begin();
    auto computed_it = computed_jacobian.begin();
    for (; expected_it != expected_jacobian.end() && computed_it != computed_jacobian.end();
         ++expected_it, ++computed_it)
        EXPECT_THAT(*computed_it, Pointwise(DoubleEq(), *expected_it));
}

TEST(JacobianTest, TrigonometricFunctionForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {M_PI / 4, M_PI / 3};

    // f(x) = [ sin(x[0]), cos(x[1]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {sin(v[0]), cos(v[1])}; };

    std::vector<std::vector<double>> expected_jacobian = {{cos(value(x[0])), 0.0},  //
                                                          {0.0, -sin(value(x[1]))}};

    std::vector<std::vector<double>> computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, TrigonometricFunctionAdjointAutoTape)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {M_PI / 4, M_PI / 3};

    // f(x) = [ sin(x[0]), cos(x[1]) ]
    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {sin(v[0]), cos(v[1])}; };

    std::vector<std::vector<double>> expected_jacobian = {{cos(value(x[0])), 0.0},  //
                                                          {0.0, -sin(value(x[1]))}};

    std::vector<std::vector<double>> computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        EXPECT_THAT(computed_jacobian[i], Pointwise(DoubleEq(), expected_jacobian[i]));
}

TEST(JacobianTest, OutOfBoundsDomainSizeMismatch)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0};

    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {v[0], v[1]}; };

    std::vector<std::vector<double>> jacobian(2, std::vector<double>(3));

    auto launch =
        [](std::vector<AD> x,
           std::function<std::vector<xad::AReal<double>>(std::vector<xad::AReal<double>> &)> f,
           std::vector<std::vector<double>>::iterator first,
           std::vector<std::vector<double>>::iterator last)
    {
        using RowIterator = decltype(first);
        xad::computeJacobian<RowIterator, double>(x, f, first, last);
    };

    EXPECT_THROW(launch(x, foo, begin(jacobian), end(jacobian)), xad::OutOfRange);
}

TEST(JacobianTest, OutOfBoundsCodomainSizeMismatch)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0};

    auto foo = [](std::vector<AD> &v) -> std::vector<AD> { return {v[0], v[0]}; };

    std::vector<std::vector<double>> jacobian(1, std::vector<double>(1));

    auto launch =
        [](std::vector<AD> x,
           std::function<std::vector<xad::AReal<double>>(std::vector<xad::AReal<double>> &)> f,
           std::vector<std::vector<double>>::iterator first,
           std::vector<std::vector<double>>::iterator last)
    {
        using RowIterator = decltype(first);
        xad::computeJacobian<RowIterator, double>(x, f, first, last);
    };

    EXPECT_THROW(launch(x, foo, begin(jacobian), end(jacobian)), xad::OutOfRange);
}

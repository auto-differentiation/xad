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

TEST(JacobianTest, SimpleAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {3.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + sin(x[1]), x[1] + sin(x[0])}; };

    std::vector<std::vector<AD>> expected_jacobian = {{1.0, cos(x[1])},  //
                                                      {cos(x[0]), 1.0}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, SimpleAdjointIteratorAutoTape)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {3.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + sin(x[1]), x[1] + sin(x[0])}; };

    std::list<std::list<AD>> expected_jacobian = {{1.0, cos(x[1])},  //
                                                  {cos(x[0]), 1.0}};

    std::list<std::list<AD>> computed_jacobian(2, std::list<AD>(2, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto row1 = computed_jacobian.begin(), row2 = expected_jacobian.begin();
    while (row1 != computed_jacobian.end() && row2 != expected_jacobian.end())
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

TEST(JacobianTest, SimpleForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {-2.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + sin(x[1]), x[1] + sin(x[0])}; };

    std::vector<std::vector<AD>> expected_jacobian = {{1.0, cos(x[1])},  //
                                                      {cos(x[0]), 1.0}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, SimpleForwardIterator)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {-2.0, 1.0};

    // f(x) = [ x[0] + sin(x[1]), x[1] + sin(x[0]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + sin(x[1]), x[1] + sin(x[0])}; };

    std::list<std::list<AD>> expected_jacobian = {{1.0, cos(x[1])},  //
                                                  {cos(x[0]), 1.0}};

    std::list<std::list<AD>> computed_jacobian(2, std::list<AD>(2, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto row1 = computed_jacobian.begin(), row2 = expected_jacobian.begin();
    while (row1 != computed_jacobian.end() && row2 != expected_jacobian.end())
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

TEST(JacobianTest, ComplexFunctionAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] * x[1], x[2] * exp(x[3]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] * x[1], x[2] * exp(x[3])}; };

    std::vector<std::vector<AD>> expected_jacobian = {{x[1], x[0], 0.0, 0.0},
                                                      {0.0, 0.0, exp(x[3]), x[2] * exp(x[3])}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, DomainLargerThanCodomainForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] + x[1], x[2] * x[3] ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {x[0] + x[1], x[2] * x[3]}; };

    std::vector<std::vector<AD>> expected_jacobian = {{1.0, 1.0, 0.0, 0.0},  //
                                                      {0.0, 0.0, x[3], x[2]}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, DomainLargerThanCodomainAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0, 4.0};

    // f(x) = [ x[0] + x[1], x[2] * x[3] ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {x[0] + x[1], x[2] * x[3]}; };

    std::vector<std::vector<AD>> expected_jacobian = {{1.0, 1.0, 0.0, 0.0},  //
                                                      {0.0, 0.0, x[3], x[2]}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, DomainSmallerThanCodomainAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<AD> x = {2.0, 3.0};

    // f(x) = [ x[0] + x[1], x[0] - x[1], x[0] * x[1] ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + x[1], x[0] - x[1], x[0] * x[1]}; };

    std::vector<std::vector<AD>> expected_jacobian = {{1.0, 1.0},  //
                                                      {1.0, -1.0},
                                                      {x[1], x[0]}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo, &tape);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, ComplexDomainNotEqualCodomainForwardIterator)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0, 3.0};

    // f(x) = [ x[0] + x[1], x[1] * x[2], exp(x[0]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {x[0] + x[1], x[1] * x[2], exp(x[0])}; };

    std::list<std::list<AD>> expected_jacobian = {{1.0, 1.0, 0.0},  //
                                                  {0.0, x[2], x[1]},
                                                  {exp(x[0]), 0.0, 0.0}};

    std::list<std::list<AD>> computed_jacobian(3, std::list<AD>(3, 0.0));
    xad::computeJacobian<decltype(begin(computed_jacobian)), double>(
        x, foo, begin(computed_jacobian), end(computed_jacobian));

    auto row1 = computed_jacobian.begin(), row2 = expected_jacobian.begin();
    while (row1 != computed_jacobian.end() && row2 != expected_jacobian.end())
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

TEST(JacobianTest, TrigonometricFunctionForward)
{
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {M_PI / 4, M_PI / 3};

    // f(x) = [ sin(x[0]), cos(x[1]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {sin(x[0]), cos(x[1])}; };

    std::vector<std::vector<AD>> expected_jacobian = {{cos(x[0]), 0.0},  //
                                                      {0.0, -sin(x[1])}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
}

TEST(JacobianTest, TrigonometricFunctionAdjointAutoTape)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {M_PI / 4, M_PI / 3};

    // f(x) = [ sin(x[0]), cos(x[1]) ]
    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {sin(x[0]), cos(x[1])}; };

    std::vector<std::vector<AD>> expected_jacobian = {{cos(x[0]), 0.0},  //
                                                      {0.0, -sin(x[1])}};

    auto computed_jacobian = xad::computeJacobian<double>(x, foo);

    for (unsigned int i = 0; i < expected_jacobian.size(); i++)
        for (unsigned int j = 0; j < expected_jacobian[i].size(); j++)
        {
            ASSERT_EQ(expected_jacobian[i][j], computed_jacobian[i][j]);
        }
}

TEST(JacobianTest, OutOfBoundsDomainSizeMismatch)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0, 2.0};

    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {x[0], x[1]}; };

    std::vector<std::vector<AD>> jacobian(2, std::vector<AD>(3));

    auto launch =
        [](std::vector<AD> x,
           std::function<std::vector<xad::AReal<double>>(std::vector<xad::AReal<double>> &)> foo,
           std::vector<std::vector<xad::AReal<double>>>::iterator first,
           std::vector<std::vector<xad::AReal<double>>>::iterator last)
    {
        using RowIterator = decltype(first);
        xad::computeJacobian<RowIterator, double>(x, foo, first, last);
    };

    EXPECT_THROW(launch(x, foo, begin(jacobian), end(jacobian)), xad::OutOfRange);
}

TEST(JacobianTest, OutOfBoundsCodomainSizeMismatch)
{
    typedef xad::adj<double> mode;
    typedef mode::active_type AD;

    std::vector<AD> x = {1.0};

    auto foo = [](std::vector<AD> &x) -> std::vector<AD> { return {x[0], x[0]}; };

    std::vector<std::vector<AD>> jacobian(1, std::vector<AD>(1));

    auto launch =
        [](std::vector<AD> x,
           std::function<std::vector<xad::AReal<double>>(std::vector<xad::AReal<double>> &)> foo,
           std::vector<std::vector<xad::AReal<double>>>::iterator first,
           std::vector<std::vector<xad::AReal<double>>>::iterator last)
    {
        using RowIterator = decltype(first);
        xad::computeJacobian<RowIterator, double>(x, foo, first, last);
    };

    EXPECT_THROW(launch(x, foo, begin(jacobian), end(jacobian)), xad::OutOfRange);
}

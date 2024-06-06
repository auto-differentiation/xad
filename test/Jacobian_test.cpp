/*******************************************************************************

   Tests for xad::Jacobian class.

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
#include <functional>
#include <vector>

template <typename T>
T foo1(std::vector<T> &x)
{
    return x[0] + sin(x[1]);
}

template <typename T>
T foo2(std::vector<T> &x)
{
    return x[1] + sin(x[0]);
}

TEST(JacobianTest, SimpleJacobianAdjoint)
{
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;

    std::vector<std::function<AD(std::vector<AD> &)>> funcs(2);
    funcs[0] = foo1<AD>;
    funcs[1] = foo2<AD>;

    std::vector<AD> x = {-2, 1};
    xad::Jacobian<AD> jac(funcs, x, &tape);

    std::vector<std::vector<AD>> cross_jacobian = {{1.0, cos(-2)}, {cos(1), 1.0}};
    std::vector<std::vector<AD>> computed_jacobian = jac.compute();

    for (unsigned int i = 0; i < cross_jacobian.size(); i++)
        for (unsigned int j = 0; j < cross_jacobian.size(); j++)
            ASSERT_EQ(cross_jacobian[i][j], computed_jacobian[i][j]);
}

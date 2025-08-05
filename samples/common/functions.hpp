/*******************************************************************************

   Test functions for computing derivatives.

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

#pragma once

#include <cmath>
#include <utility>
/// Applies the sin function a number of times.
template <class T>
void repeated_sin(int n, T& x)
{
    using std::sin;

    for (int i = 0; i < n; ++i) x = sin(x);
}

/// Arbitrary function with 4 inputs and one output
template <class T>
T f(const T& x0, const T& x1, const T& x2, const T& x3)
{
    T a = sin(x0) * cos(x1);
    T b = x2 * x3 - tan(x1 - x2);
    T c = a + 2 * b;
    return c * c;
}

/// Function with 2 outputs to demonstrate vector-mode adjoints
template <class T>
std::pair<T, T> f2(const T& x0, const T& x1, const T& x2, const T& x3)
{
    T a = sin(x0) * cos(x1);
    T b = x2 * x3 - tan(x1 - x2);
    T c = a + 2 * b;
    return {c * c, 4 * c + b};
}

/// Sum all elements in an array
inline double sum_elements(const double* x, int n)
{
    double ret = 0.0;
    for (int i = 0; i < n; ++i) ret += x[i];
    return ret;
}

/*******************************************************************************

   Hessian benchmark functions.

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


#pragma once

#include <functional>
#include <vector>
#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <XAD/XAD.hpp>
#include <XAD/Hessian.hpp>

// foo function
template <typename T>
std::function<T(std::vector<T> &)> make_foo() {
    return [](std::vector<T> &x) -> T {
        return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0]);
    };
}

// ackley function
template <typename T>
std::function<T(std::vector<T> &)> make_ackley() {
    return [](std::vector<T> &x) -> T {
        T sum = 0.0;
        T cossum = 0.0;
        T n = static_cast<T>(x.size());

        for (auto &xi : x) {
            sum += xi * xi;
            cossum += cos(2.0 * M_PI * xi);
        }

        return -20.0 * exp(-0.2 * sqrt(sum / n)) - exp(cossum / n) + 20.0 + exp(1.0);
    };
}

// neural loss function
template <typename T>
std::function<T(std::vector<T> &)> make_neuralLoss() {
    return [](std::vector<T> &x) -> T {
        T a = 0.0;
        T b = 1.11;

        for (std::size_t i = 0; i < x.size(); ++i) {
            a += x[i] * static_cast<T>(i);
        }

        return log(1 + exp(a + b));
    };
}

// sparse function
template <typename T>
std::function<T(std::vector<T> &)> make_sparse() {
    return [](std::vector<T> &x) -> T {
        T sum = 0.0;

        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T diff = x[i] - x[i + 1];
            sum += diff * diff;
        }

        return sum;
    };
}

// dense function
// the idea is that every variable has a dependency on every other variable
// that way every entry of the hessian is non-zero
template <typename T>
std::function<T(std::vector<T> &)> make_dense() {
    return [](std::vector<T> &x) -> T {
        T t = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            for (std::size_t j = 0; j < x.size(); ++j) {
                if (i == j) {
                    continue;
                }
                t += x[i] * x[j];
            }
        }

        return t;
    };
}
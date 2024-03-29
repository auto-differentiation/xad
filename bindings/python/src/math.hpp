/*******************************************************************************

   Exports all XAD math functions to Python.

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

#include <XAD/XAD.hpp>
#include <pybind11/pybind11.h>
#include <cmath>
#include <numeric>

namespace py = pybind11;

using AReal = xad::AReal<double>;
using FReal = xad::FReal<double>;

template <class T>
void add_math_functions(py::module_& m)
{
    m.def(
        "sqrt", [](const T& d) { return T(xad::sqrt(d)); }, "square root");
    m.def(
        "pow", [](const T& d, const double& b) { return T(xad::pow(d, b)); }, "power");
    m.def(
        "pow", [](const double& d, const T& b) { return T(xad::pow(d, b)); }, "power");
    m.def(
        "pow", [](const T& d, const T& b) { return T(xad::pow(d, b)); }, "power");
    m.def(
        "log10", [](const T& d) { return T(xad::log10(d)); }, "base 10 logarithm");
    m.def(
        "log", [](const T& d) { return T(xad::log(d)); }, "natural logarithm");
    m.def(
        "ldexp", [](const T& d, const int b) { return T(xad::ldexp(d, b)); },
        "mutiplies x by 2 to the power of exp");
    m.def(
        "exp", [](const T& d) { return T(xad::exp(d)); }, "exponential function");
    m.def(
        "exp2", [](const T& d) { return T(xad::exp2(d)); },
        "computes 2 to the power of the argument");
    m.def(
        "expm1", [](const T& d) { return T(xad::expm1(d)); }, "computes exp(x)-1");
    m.def(
        "log1p", [](const T& d) { return T(xad::log1p(d)); }, "computes log(1 + x)");
    m.def(
        "log2", [](const T& d) { return T(xad::log2(d)); }, "base 2 logarithm");

    m.def(
        "modf",
        [](const T& d)
        {
            double value = 1.0;
            T r = xad::modf(d, &value);
            return py::make_tuple(r, value);
        },
        "decomposes into integral and fractional parts");
    m.def(
        "ceil", [](const T& d) { return T(xad::ceil(d)); }, "rounding away from zero");
    m.def(
        "floor", [](const T& d) { return T(xad::floor(d)); }, "rounding towards zero");
    m.def(
        "frexp",
        [](const T& d)
        {
            int value = 1;
            T r = xad::frexp(d, &value);
            return py::make_tuple(r, value);
        },
        "decomposes into normalised fraction and an integral power of 2");
    m.def(
        "fmod", [](const T& d, const T& b) { return T(xad::fmod(d, b)); },
        "floating point remainer after integer division");

    m.def(
        "min", [](const T& d, const T& b) { return T(xad::min(d, b)); }, "minimum of 2 values");
    m.def(
        "min", [](const T& d, const double& b) { return T(xad::min(d, b)); },
        "minimum of 2 values");
    m.def(
        "min", [](const double& d, const T& b) { return T(xad::min(d, b)); },
        "minimum of 2 values");
    m.def(
        "max", [](const T& d, const T& b) { return T(xad::max(d, b)); }, "maximum of 2 values");
    m.def(
        "max", [](const T& d, const double& b) { return T(xad::max(d, b)); },
        "maximum of 2 values");
    m.def(
        "max", [](const double& d, const T& b) { return T(xad::max(d, b)); },
        "maximum of 2 values");
    m.def(
        "fmax", [](const T& d, const T& b) { return T(xad::fmax(d, b)); }, "maximum of 2 values");
    m.def(
        "fmax", [](const T& d, const double& b) { return T(xad::fmax(d, b)); },
        "maximum of 2 values");
    m.def(
        "fmax", [](const double& d, const T& b) { return T(xad::fmax(d, b)); },
        "maximum of 2 values");
    m.def(
        "fmin", [](const T& d, const T& b) { return T(xad::fmin(d, b)); }, "minimum of 2 values");
    m.def(
        "fmin", [](const T& d, const double& b) { return T(xad::fmin(d, b)); },
        "minimum of 2 values");
    m.def(
        "fmin", [](const double& d, const T& b) { return T(xad::fmin(d, b)); },
        "minimum of 2 values");
    m.def(
        "abs", [](const T& d) { return T(xad::abs((d))); }, "absolute value");
    m.def(
        "fabs", [](const T& d) { return T(xad::fabs(d)); }, "absolute value");

    m.def(
        "smooth_abs", [](const T& d) { return T(xad::smooth_abs(d)); },
        "smoothed abs function for well-defined derivatives");
    m.def(
        "smooth_max", [](const T& d, const T& b) { return T(xad::smooth_max(d, b)); },
        "smoothed max function for well-defined derivatives");
    m.def(
        "smooth_max", [](const T& d, const double& b) { return T(xad::smooth_max(d, b)); },
        "smoothed max function for well-defined derivatives");
    m.def(
        "smooth_max", [](const double& d, const T& b) { return T(xad::smooth_max(d, b)); },
        "smoothed max function for well-defined derivatives");
    m.def(
        "smooth_min", [](const T& d, const T& b) { return T(xad::smooth_min(d, b)); },
        "smoothed min function for well-defined derivatives");
    m.def(
        "smooth_min", [](const T& d, const double& b) { return T(xad::smooth_min(d, b)); },
        "smoothed min function for well-defined derivatives");
    m.def(
        "smooth_min", [](const double& d, const T& b) { return T(xad::smooth_min(d, b)); },
        "smoothed min function for well-defined derivatives");

    m.def(
        "tan", [](const T& d) { return T(xad::tan(d)); }, "tangent");
    m.def(
        "atan", [](const T& d) { return T(xad::atan(d)); }, "inverse tangent");
    m.def(
        "tanh", [](const T& d) { return T(xad::tanh(d)); }, "tangent hyperbolicus");
    m.def(
        "atan2", [](const T& d, const T& b) { return T(xad::atan2(d, b)); },
        "4-quadrant inverse tangent");
    m.def(
        "atan2", [](const T& d, const double& b) { return T(xad::atan2(d, b)); },
        "4 quadrant inverse tangent");
    m.def(
        "atan2", [](const double& d, const T& b) { return T(xad::atan2(d, b)); },
        "4 quadrant inverse tangent");
    m.def(
        "atanh", [](const T& d) { return T(xad::atanh(d)); }, "inverse tangent hyperbolicus");
    m.def(
        "cos", [](const T& d) { return T(xad::cos(d)); }, "cosine");
    m.def(
        "acos", [](const T& d) { return T(xad::acos(d)); }, "inverse cosine");
    m.def(
        "cosh", [](const T& d) { return T(xad::cosh(d)); }, "cosine hyperbolicus");
    m.def(
        "acosh", [](const T& d) { return T(xad::acosh(d)); }, "inverse cosine hyperbolicus");
    m.def(
        "sin", [](const T& d) { return T(xad::sin(d)); }, "sine");
    m.def(
        "asin", [](const T& d) { return T(xad::asin(d)); }, "inverse sine");
    m.def(
        "sinh", [](const T& d) { return T(xad::sinh(d)); }, "sine hyperbolicus");
    m.def(
        "asinh", [](const T& d) { return T(xad::asinh(d)); }, "inverse sine hyperbolicus");

    m.def(
        "cbrt", [](const T& d) { return T(xad::cbrt(d)); }, "cubic root");
    m.def(
        "erf", [](const T& d) { return T(xad::erf(d)); }, "error function");
    m.def(
        "erfc", [](const T& d) { return T(xad::erfc(d)); }, "complementary error function");
    m.def(
        "nextafter", [](const T& d, const T& b) { return T(xad::nextafter(d, b)); },
        "next representable value in the given direction");
    m.def(
        "nextafter", [](const T& d, const double& b) { return T(xad::nextafter(d, b)); },
        "next representable value in the given direction");
    m.def(
        "nextafter", [](const double& d, const T& b) { return T(xad::nextafter(d, b)); },
        "next representable value in the given direction");
    m.def(
        "remainder", [](const T& d, const T& b) { return T(xad::remainder(d, b)); },
        "signed remainder after integer division");
    m.def(
        "remainder", [](const T& d, const double& b) { return T(xad::remainder(d, b)); },
        "signed remainder after integer division");
    m.def(
        "remainder", [](const double& d, const T& b) { return T(xad::remainder(d, b)); },
        "signed remainder after integer division");
    m.def(
        "degrees", [](const T& d) { return T((d * 180) / M_PI); }, "convert radians to degrees");
    m.def(
        "radians", [](const T& d) { return T((d * M_PI) / 180); }, "convert degrees to radians");
    m.def(
        "copysign", [](const double& d, const T& b) { return T(xad::abs(d) * (b / xad::abs(b))); },
        "copy sign of one value to another");
    m.def(
        "copysign", [](const T& d, const double& b) { return T(xad::abs(d) * (b / xad::abs(b))); },
        "copy sign of one value to another");
    m.def(
        "copysign", [](const T& d, const T& b) { return T(xad::abs(d) * (b / xad::abs(b))); },
        "copy sign of one value to another");
    m.def(
        "trunc", [](const T& d) { return T(xad::trunc(d)); }, "cut off decimals");
};

void py_math(py::module& m)
{
    py::module_ m1 = m.def_submodule("math");

    add_math_functions<AReal>(m1);
    add_math_functions<FReal>(m1);
    add_math_functions<double>(m1);
};

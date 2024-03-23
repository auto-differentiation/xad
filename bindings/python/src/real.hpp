/*******************************************************************************

   Defines the bindings for the XAD active types.

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
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

using AReal = xad::AReal<double>;
using FReal = xad::FReal<double>;

inline void add_extra_methods(py::class_<AReal>& c)
{
    c.def("setAdjoint", &AReal::setAdjoint, "set adjoint of this variable");
    c.def("shouldRecord", &AReal::shouldRecord,
          "Check if the variable is registered on tape and should record");
    c.def("getSlot", &AReal::getSlot, "Get the slot of this variable on the tape");
}

inline void add_extra_methods(py::class_<FReal>&) {}

template <class T, class T1, class T2>
inline T py_fmod(const T1& x, const T2& y)
{
    auto res = T(xad::fmod(x, y));
    if ((res < 0 && y > 0) || (res > 0 && y < 0))
    {
        return res + y;
    }
    return res;
}

template <class T, class T1, class T2>
inline std::pair<T, T> py_divmod(const T1& x, const T2& y)
{
    T mod = py_fmod<T>(x, y);
    T div = (x - mod) / y;
    return {div, mod};
}

template <class T, class T1, class T2>
inline T py_floordiv(const T1& x, const T2& y)
{
    return xad::floor(x / y);
}

template <class T>
void py_real(py::module_& m)
{
    auto& c = py::class_<T>(m, "Real", py::dynamic_attr(), "active arithmetic type for first order adjoint mode")
                  .def(py::init<double>())
                  .def(py::init<>())
                  .def(py::self == py::self)
                  .def(py::self != py::self)
                  .def(py::self >= py::self)
                  .def(py::self <= py::self)
                  .def(py::self > py::self)
                  .def(py::self < py::self);

    add_extra_methods(c);

    c.def("__int__", [](const T& d) { return int(d.getValue()); })
        .def("__bool__", [](const T& d) { return bool(d); })
        .def("__neg__", [](const T& d) { return T(-d); })
        .def("__pos__", [](const T& d) { return d; })
        .def(
            "__add__", [](const T& a, double b) { return T(a + b); }, py::is_operator())
        .def(
            "__add__", [](const T& a, const T& b) { return T(a + b); }, py::is_operator())
        .def(
            "__radd__", [](const T& a, double b) { return T(a + b); }, py::is_operator())
        .def(
            "__mul__", [](const T& a, double b) { return T(a * b); }, py::is_operator())
        .def(
            "__mul__", [](const T& a, const T& b) { return T(a * b); }, py::is_operator())
        .def(
            "__rmul__", [](const T& a, double b) { return T(a * b); }, py::is_operator())
        .def(
            "__sub__", [](const T& a, double b) { return T(a - b); }, py::is_operator())
        .def(
            "__sub__", [](const T& a, const T& b) { return T(a - b); }, py::is_operator())
        .def(
            "__rsub__", [](const T& a, double b) { return T(b - a); }, py::is_operator())
        .def(
            "__truediv__", [](const T& a, double b) { return T(a / b); }, py::is_operator())
        .def(
            "__truediv__", [](const T& a, const T& b) { return T(a / b); }, py::is_operator())
        .def(
            "__rtruediv__", [](const T& a, double b) { return T(b / a); }, py::is_operator())
        .def("__repr__", [](const T& a) { return std::to_string(a.getValue()); })
        .def(
            "__rgt__", [](const T& a, double b) { return (b > a); }, py::is_operator())
        .def(
            "__gt__", [](const T& a, double b) { return (a > b); }, py::is_operator())
        .def(
            "__rlt__", [](const T& a, double b) { return (b < a); }, py::is_operator())
        .def(
            "__lt__", [](const T& a, double b) { return (a < b); }, py::is_operator())
        .def(
            "__rge__", [](const T& a, double b) { return (b >= a); }, py::is_operator())
        .def(
            "__ge__", [](const T& a, double b) { return (a >= b); }, py::is_operator())
        .def(
            "__rle__", [](const T& a, double b) { return (b <= a); }, py::is_operator())
        .def(
            "__le__", [](const T& a, double b) { return (a <= b); }, py::is_operator())
        .def(
            "__req__", [](const T& a, double b) { return (b == a); }, py::is_operator())
        .def(
            "__eq__", [](const T& a, double b) { return (a == b); }, py::is_operator())
        .def(
            "__rne__", [](const T& a, double b) { return (b != a); }, py::is_operator())
        .def(
            "__ne__", [](const T& a, double b) { return (a != b); }, py::is_operator())
        .def("__round__",
             [](const T& x, int ndigits)
             {
                 double factor = std::pow(10, ndigits);
                 return T(xad::round(x * factor) / factor);
             })
        .def("__round__", [](const T& x) { return int(xad::round(x)); })
        .def("__ceil__", [](const T& x) { return int(xad::ceil(x)); })
        .def("__floor__", [](const T& x) { return int(xad::floor(x)); })
        .def("__trunc__", [](const T& x) { return int(xad::trunc(x)); })
        .def("__abs__", [](const T& x) { return T(xad::abs(x)); })
        .def("__pow__", [](const T& x, const T& y) { return T(xad::pow(x, y)); })
        .def("__pow__", [](const T& x, int y) { return T(xad::pow(x, y)); })
        .def("__pow__", [](const T& x, double y) { return T(xad::pow(x, y)); })
        .def("__rpow__", [](const T& x, const T& y) { return T(xad::pow(y, x)); })
        .def("__rpow__", [](const T& x, int y) { return T(xad::pow(y, x)); })
        .def("__rpow__", [](const T& x, double y) { return T(xad::pow(y, x)); })
        .def("__mod__", [](const T& x, const T& y) { return py_fmod<T>(x, y); })
        .def("__mod__", [](const T& x, int y) { return py_fmod<T>(x, y); })
        .def("__mod__", [](const T& x, double y) { return py_fmod<T>(x, y); })
        .def("__rmod__", [](const T& y, const T& x) { return py_fmod<T>(x, y); })
        .def("__rmod__", [](const T& y, int x) { return py_fmod<T>(x, y); })
        .def("__rmod__", [](const T& y, double x) { return py_fmod<T>(x, y); })
        .def("__divmod__", [](const T& x, const T& y) { return py_divmod<T>(x, y); })
        .def("__divmod__", [](const T& x, double y) { return py_divmod<T>(x, y); })
        .def("__divmod__", [](const T& x, int y) { return py_divmod<T>(x, y); })
        .def("__rdivmod__", [](const T& y, const T& x) { return py_divmod<T>(x, y); })
        .def("__rdivmod__", [](const T& y, double x) { return py_divmod<T>(x, y); })
        .def("__rdivmod__", [](const T& y, int x) { return py_divmod<T>(x, y); })
        .def("__floordiv__", [](const T& x, const T& y) { return py_floordiv<T>(x, y); })
        .def("__floordiv__", [](const T& x, double y) { return py_floordiv<T>(x, y); })
        .def("__floordiv__", [](const T& x, int y) { return py_floordiv<T>(x, y); })
        .def("__rfloordiv__", [](const T& y, const T& x) { return py_floordiv<T>(x, y); })
        .def("__rfloordiv__", [](const T& y, double x) { return py_floordiv<T>(x, y); })
        .def("__rfloordiv__", [](const T& y, int x) { return py_floordiv<T>(x, y); })
        // to set/get derivatives
        .def(
            "getValue", [](const T& self) { return self.getValue(); }, "get the underlying value")
        .def(
            "setDerivative", [](T& self, double v) { self.setDerivative(v); },
            "set the adjoint of this variable")
        .def(
            "getDerivative", [](const T& self) { return self.getDerivative(); },
            "get the adjoint of this variable");
    c.def(
         "conjugate", [](const T& x) { return x; }, "complex conjugate")
        .def(
            "real", [](const T& x) { return x; }, "real part")
        .def(
            "imag", [](const T&) { return T(0.0); }, "imaginary part");
    // properties
    c.def_property_readonly(
        "value", [](const T& self) { return self.getValue(); },
        "read-only property to get the value");
    c.def_property(
        "derivative", [](const T& self) { return self.getDerivative(); },
        [](T& self, double v) { self.setDerivative(v); },
        "read-write property to get/set derivatives");
}

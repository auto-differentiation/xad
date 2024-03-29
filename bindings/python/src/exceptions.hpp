/*******************************************************************************

   Exports all XAD exceptions to Python.

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

namespace py = pybind11;

void py_exceptions(py::module_& m)
{

    py::module_ exceptions = m.def_submodule("exceptions");
    auto& xad_exception = py::register_exception<xad::Exception>(exceptions, "XadException");
    xad_exception.doc() = "Base class for all exceptions raised by XAD";
    py::register_exception<xad::TapeAlreadyActive>(exceptions, "TapeAlreadyActive", xad_exception)
        .doc() =
        "Raised when activating a tape when this or another tape is already active in the current "
        "thread";
    py::register_exception<xad::OutOfRange>(exceptions, "OutOfRange", xad_exception).doc() =
        "raised when setting a derivative at a slot that is out of range of the recorded variables";
    py::register_exception<xad::DerivativesNotInitialized>(exceptions, "DerivativesNotInitialized",
                                                           xad_exception)
        .doc() =
        "Raised when setting derivatives on the tape without a recording and registered outputs";
    py::register_exception<xad::NoTapeException>(exceptions, "NoTapeException", xad_exception)
        .doc() =
        "raised if an opteration that requires an active tape is performed while not tape is "
        "active";
}

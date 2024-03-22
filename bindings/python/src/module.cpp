/*******************************************************************************

   Main pybind module definition for the extension module.

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
#include <pybind11/pybind11.h>
#include "exceptions.hpp"
#include "math.hpp"
#include "real.hpp"
#include "tape.hpp"

namespace py = pybind11;

void py_adj_1st(py::module_ &m)
{
    py::module_ adj = m.def_submodule("adj_1st");
    py_real<AReal>(adj);
    py_tape(adj);
}

void py_fwd_1st(py::module_ &m)
{
    py::module_ fwd = m.def_submodule("fwd_1st");
    py_real<FReal>(fwd);
}

PYBIND11_MODULE(_xad_autodiff, m)
{
    py_adj_1st(m);
    py_fwd_1st(m);
    py_math(m);
    py_exceptions(m);
}
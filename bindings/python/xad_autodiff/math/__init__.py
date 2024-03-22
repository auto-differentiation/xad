##############################################################################
#
#  Math module for the XAD Python bindings
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

"""XAD math module - mimics the standard math module, but allows using XAD active types
   as arguments. Note that it's also possible to call the functions contained with 
   float arguments (passive type), to allow seamless integration with active and passive
   data types.
"""

from typing import Union, List
from .._xad_autodiff.math import (
    sqrt,
    pow,
    log10,
    log,
    ldexp,
    exp,
    exp2,
    expm1,
    log1p,
    log2,
    modf,
    ceil,
    floor,
    frexp,
    fmod,
    min,
    max,
    fmax,
    fmin,
    abs,
    fabs,
    smooth_abs,
    smooth_max,
    smooth_min,
    tan,
    atan,
    tanh,
    atan2,
    atanh,
    cos,
    acos,
    cosh,
    acosh,
    sin,
    asin,
    sinh,
    asinh,
    cbrt,
    erf,
    erfc,
    nextafter,
    remainder,
    degrees,
    radians,
    copysign,
    trunc,
)


__all__ = [
    "sqrt",
    "pow",
    "log10",
    "log",
    "ldexp",
    "exp",
    "exp2",
    "expm1",
    "log1p",
    "log2",
    "modf",
    "ceil",
    "floor",
    "frexp",
    "fmod",
    "min",
    "max",
    "fmax",
    "fmin",
    "abs",
    "fabs",
    "smooth_abs",
    "smooth_max",
    "smooth_min",
    "tan",
    "atan",
    "tanh",
    "atan2",
    "atanh",
    "cos",
    "acos",
    "cosh",
    "acosh",
    "sin",
    "asin",
    "sinh",
    "asinh",
    "cbrt",
    "erf",
    "erfc",
    "nextafter",
    "remainder",
    "degrees",
    "radians",
    "copysign",
    "trunc",
    "hypot",
    "dist",
    "pi",
    "e",
    "tau",
    "inf",
    "nan",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    
]

import xad_autodiff as xad
import math as _math


def hypot(*inputs: List[Union["xad.adj_1st.Real", "xad.fwd_1st.Real", float, int]]):
    return sqrt(sum(pow(x, 2) for x in inputs))


def dist(p: Union["xad.adj_1st.Real", "xad.fwd_1st.Real", float, int], q):
    return sqrt(sum(pow(px - qx, 2) for px, qx in zip(p, q)))

def isclose(a, b, *args, **kwargs):
    return _math.isclose(xad.value(a), xad.value(b), *args, **kwargs)

def isfinite(x):
    return _math.isfinite(xad.value(x))

def isinf(x):
    return _math.isinf(xad.value(x))

def isnan(x):
    return _math.isnan(xad.value(x))

# constants
pi = _math.pi
e = _math.e
tau = _math.tau
inf = _math.inf
nan = _math.nan
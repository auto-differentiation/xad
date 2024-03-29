##############################################################################
#
#  First order forward mode module for the XAD Python bindings
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

from typing import Tuple, Type
from xad_autodiff._xad_autodiff.fwd_1st import Real

__all__ = ["Real"]


setattr(Real, "value", property(Real.getValue, doc="get the underlying float value of the object"))
setattr(
    Real,
    "derivative",
    property(
        Real.getDerivative, Real.setDerivative, doc="get/set the derivative of the object"
    ),
)

# additional methods inserted on the python side
def _as_integer_ratio(x: Real) -> Tuple[int, int]:
    """Returns a rational representation of the float with numerator and denominator in a tuple"""
    return x.value.as_integer_ratio()

Real.as_integer_ratio = _as_integer_ratio

def _fromhex(cls: Type[Real], hexstr: str) -> Real:
    """Initialize from a hex expression"""
    return cls(float.fromhex(hexstr))

Real.fromhex = classmethod(_fromhex)

def _getnewargs(x: Real) -> Tuple[float]:
    return (x.value, )

Real.__getnewargs__ = _getnewargs

def _hash(x: Real) -> int:
    return hash(x.value)

Real.__hash__ = _hash

def _hex(x: Real) -> str:
    return x.value.hex()

Real.hex = _hex

def _is_integer(x: Real) -> bool:
    return x.value.is_integer()

Real.is_integer = _is_integer

def _format(x: Real, spec) -> str:
    return format(x.value, spec)

Real.__format__ = _format
##############################################################################
#
#  XAD Python bindings
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

"""Python bindings for the XAD comprehensive library for automatic differentiation"""

from typing import Any, Union
from ._xad_autodiff import adj_1st
from ._xad_autodiff import fwd_1st

__all__ = ["value", "derivative"]


def value(x: Union[adj_1st.Real, fwd_1st.Real, Any]) -> float:
    """Get the value of an XAD active type - or return the value itself otherwise

    Args:
        x (Real | any): Argument to get the value of

    Returns:
        float: The value stored in the variable
    """
    if isinstance(x, adj_1st.Real) or isinstance(x, fwd_1st.Real):
        return x.getValue()
    else:
        return x


def derivative(x: Union[adj_1st.Real, fwd_1st.Real]) -> float:
    """Get the derivative of an XAD active type - forward or adjoint mode

    Args:
        x (Real): Argument to extract the derivative information from

    Returns:
        float: The derivative
    """
    if isinstance(x, adj_1st.Real) or isinstance(x, fwd_1st.Real):
        return x.getDerivative()
    else:
        raise TypeError("type " + type(x).__name__ + " is not an XAD active type")

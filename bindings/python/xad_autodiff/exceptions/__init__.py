##############################################################################
#
#  Exceptions module for the XAD Python bindings
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

from .._xad_autodiff.exceptions import (
    XadException,
    TapeAlreadyActive,
    OutOfRange,
    DerivativesNotInitialized,
    NoTapeException,
)

__all__ = [
    "XadException",
    "TapeAlreadyActive",
    "OutOfRange",
    "DerivativesNotInitialized",
    "NoTapeException",
]
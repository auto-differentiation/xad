##############################################################################
#
#  Sample for first-order adjoint calculation with Python
#
#    Computes
#      y = f(x0, x1, x2, x3)
#    and its first order derivatives
#      dy/dx0, dy/dx1, dy/dx2, dy/dx3
#    using adjoint mode.
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

import xad_autodiff.adj_1st as xadj


#  input values
x0 = 1.0
x1 = 1.5
x2 = 1.3
x3 = 1.2

# set independent variables
x0_ad = xadj.Real(x0)
x1_ad = xadj.Real(x1)
x2_ad = xadj.Real(x2)
x3_ad = xadj.Real(x3)

with xadj.Tape() as tape:
    # and register them
    tape.registerInput(x0_ad)
    tape.registerInput(x1_ad)
    tape.registerInput(x2_ad)
    tape.registerInput(x3_ad)

    # start recording derivatives
    tape.newRecording()

    # calculate the output
    y = x0_ad + x1_ad - x2_ad * x3_ad

    # register and seed adjoint of output
    tape.registerOutput(y)
    y.derivative = 1.0

    # compute all other adjoints
    tape.computeAdjoints()

    # output results
    print(f"y = {y}")
    print(f"first order derivatives:\n")
    print(f"dy/dx0 = {x0_ad.derivative}")
    print(f"dy/dx1 = {x1_ad.derivative}")
    print(f"dy/dx2 = {x2_ad.derivative}")
    print(f"dy/dx3 = {x3_ad.derivative}")

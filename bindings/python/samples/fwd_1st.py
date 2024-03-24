##############################################################################
#
#  Sample for 1st order forward mode in Python.
#
#    Computes
#      y = f(x0, x1, x2, x3)
#    and it's first order derivative w.r.t. x0 using forward mode:
#      dy/dx0
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


import xad_autodiff.fwd_1st as xfwd

#  input values
x0 = 1.0
x1 = 1.5
x2 = 1.3
x3 = 1.2

# set independent variables
x0_ad = xfwd.Real(x0)
x1_ad = xfwd.Real(x1)
x2_ad = xfwd.Real(x2)
x3_ad = xfwd.Real(x3)

# compute derivative w.r.t. x0
# (if other derivatives are needed, the initial derivatives have to be reset
# and the function run again)
x0_ad.derivative = 1.0

# run the algorithm with active variables
y = 2 * x0_ad + x1_ad - x2_ad * x3_ad

# output results{
print(f"y = {y.value}")
print("first order derivative:")
print(f"dy/dx0 = {y.derivative}")

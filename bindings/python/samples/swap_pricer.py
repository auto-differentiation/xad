##############################################################################
#
#  Computes the discount rate sensitivities of a simple swap pricer
#  using adjoint mode.
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

from random import randint
from typing import List
from xad_autodiff import math
import xad_autodiff.adj_1st as xadj


def calculate_price_swap(
    disc_rates: List[xadj.Real],
    is_fixed_pay: bool,
    mat: List[float],
    float_rates: List[float],
    fixed_rate: float,
    face_value: float,
):
    """Calculates the Swap price, given maturities (in years), float and fixed rates
    at the given maturities, and the face value"""

    # discounted fixed cashflows
    b_fix = sum(face_value * fixed_rate / math.pow(1 + r, T) for r, T in zip(disc_rates, mat))
    # notional exchange at the end
    b_fix += face_value / math.pow(1.0 + disc_rates[-1], mat[-1])
    # discounted float cashflows
    b_flt = sum(
        face_value * f / math.pow(1 + r, T) for f, r, T in zip(float_rates, disc_rates, mat)
    )
    # notional exchange at the end
    b_flt += face_value / math.pow(1.0 + disc_rates[-1], mat[-1])

    return b_flt - b_fix if is_fixed_pay else b_fix - b_flt


# initialise input data
n_rates = 30
face_value = 10000000.0
fixed_rate = 0.03
is_fixed_pay = True
rand_max = 214
float_rates = [0.01 + randint(0, rand_max) / rand_max * 0.1 for _ in range(n_rates)]
disc_rates = [0.01 + randint(0, rand_max) / rand_max * 0.06 for _ in range(n_rates)]
maturities = list(range(1, n_rates + 1))

disc_rates_d = [xadj.Real(r) for r in disc_rates]

with xadj.Tape() as tape:
    #  set independent variables
    tape.registerInputs(disc_rates_d)

    #  start recording derivatives
    tape.newRecording()

    v = calculate_price_swap(
        disc_rates_d, is_fixed_pay, maturities, float_rates, fixed_rate, face_value
    )

    #  seed adjoint of output
    tape.registerOutput(v)
    v.derivative = 1.0

    #  compute all other adjoints
    tape.computeAdjoints()

    #  output results
    print(f"v = {v.value:.2f}")
    print("Discount rate sensitivities for 1 basispoint shift:")
    for i, rate in enumerate(disc_rates_d):
        print(f"dv/dr{i} = {rate.derivative * 0.0001:.2f}")

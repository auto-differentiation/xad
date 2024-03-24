##############################################################################
#
#  Pytests for math functions and their derivatives
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

import sys
import pytest
from xad_autodiff.adj_1st import Tape, Real as Areal
from xad_autodiff.fwd_1st import Real as Freal
from xad_autodiff import math as ad_math
import math

# This is a list of math functions with their expected outcomes and derivatives,
# used in parametrised tests, for unary functions.
#
# The format is a list of tuples, where each tuple has the following entries:
# - XAD math function: Callable
# - parameter value for the function: float
# - expected result: float
# - expected derivative value: float
#
PARAMETERS_FOR_UNARY_FUNC = [
    (ad_math.sin, math.pi / 4, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    (ad_math.cos, math.pi / 4, math.cos(math.pi / 4), -1 * math.sin(math.pi / 4)),
    (ad_math.tan, 0.5, math.tan(0.5), 2 / (1 + math.cos(2 * 0.5))),
    (ad_math.atan, 0.5, math.atan(0.5), 1 / (1 + math.pow(0.5, 2))),
    (ad_math.acos, 0.5, math.acos(0.5), -1 / math.sqrt(1 - math.pow(0.5, 2))),
    (ad_math.asin, 0.5, math.asin(0.5), 1 / math.sqrt(1 - math.pow(0.5, 2))),
    (ad_math.tanh, 0.5, math.tanh(0.5), 1 - math.pow(math.tanh(0.5), 2)),
    (ad_math.cosh, 0.5, math.cosh(0.5), math.sinh(0.5)),
    (ad_math.sinh, 0.5, math.sinh(0.5), math.cosh(0.5)),
    (ad_math.atanh, 0.5, math.atanh(0.5), 1 / (1 - math.pow(0.5, 2))),
    (ad_math.asinh, 0.5, math.asinh(0.5), 1 / math.sqrt(1 + math.pow(0.5, 2))),
    (ad_math.acosh, 1.5, math.acosh(1.5), 1 / math.sqrt(math.pow(1.5, 2) - 1)),
    (ad_math.sqrt, 4, math.sqrt(4), 1 / (2 * math.sqrt(4))),
    (ad_math.log10, 4, math.log10(4), 1 / (4 * math.log(10))),
    (ad_math.log, 4, math.log(4), 1 / 4),
    (ad_math.exp, 4, math.exp(4), math.exp(4)),
    (ad_math.expm1, 4, math.expm1(4), math.exp(4)),
    (ad_math.log1p, 4, math.log1p(4), 1 / (5)),
    (ad_math.log2, 4, math.log2(4), 1 / (4 * math.log(2))),
    (ad_math.abs, -4, abs(-4), -1),
    (ad_math.fabs, -4.4, 4.4, -1),
    (ad_math.smooth_abs, -4.4, 4.4, -1),
    (
        ad_math.erf,
        -1.4,
        math.erf(-1.4),
        (2 / math.sqrt(math.pi)) * math.exp(-1 * math.pow(-1.4, 2)),
    ),
    (
        ad_math.erfc,
        -1.4,
        math.erfc(-1.4),
        (-2 / math.sqrt(math.pi)) * math.exp(-1 * math.pow(-1.4, 2)),
    ),
    (ad_math.cbrt, 8, 2.0, (1 / 3) * (math.pow(8, (-2 / 3)))),
    (ad_math.trunc, 8.1, math.trunc(8.1), 0),
    (ad_math.ceil, 3.7, math.ceil(3.7), 0),
    (ad_math.floor, 3.7, math.floor(3.7), 0),
]


# This is a list of math functions with their expected outcomes and derivatives,
# used in parametrised tests, for binary functions.
#
# The format is a list of tuples, where each tuple has the following entries:
# - XAD math function: Callable
# - parameter1 value for the function: float
# - parameter2 value for the function: float
# - expected result: float
# - expected derivative1 value: float
# - expected derivative2 value: float
#
PARAMETERS_FOR_BINARY_FUNC = [
    (ad_math.min, 3, 4, 3, 1, 0),
    (ad_math.min, 4, 3, 3, 0, 1),
    (ad_math.max, 3, 4, 4, 0, 1),
    (ad_math.max, 4, 3, 4, 1, 0),
    (ad_math.fmin, 3.5, 4.3, 3.5, 1, 0),
    (ad_math.fmin, 4.3, 3.5, 3.5, 0, 1),
    (ad_math.fmax, 3.5, 4.3, 4.3, 0, 1),
    (ad_math.fmax, 4.3, 3.5, 4.3, 1, 0),
    (ad_math.smooth_min, 3.5, 4.3, 3.5, 1, 0),
    (ad_math.smooth_min, 4.3, 3.5, 3.5, 0, 1),
    (ad_math.smooth_max, 3.5, 4.3, 4.3, 0, 1),
    (ad_math.smooth_max, 4.3, 3.5, 4.3, 1, 0),
    (ad_math.remainder, 5, 2, math.remainder(5, 2), 1, -2),
    (ad_math.fmod, 6, 2, math.fmod(6, 3), 1, -3),
]

_binary_with_scalar_funcs = [
    (ad_math.pow, math.pow),
    (ad_math.min, min),
    (ad_math.max, max),
    (ad_math.fmin, min),
    (ad_math.fmax, max),
    (ad_math.atan2, math.atan2),
    (ad_math.remainder, math.remainder),
    (ad_math.copysign, math.copysign),
]

if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor >= 9):
    # introduced in Python 3.9
    PARAMETERS_FOR_BINARY_FUNC.append((ad_math.nextafter, 3.5, 4.3, math.nextafter(3.5, 4.3), 1, 0))
    _binary_with_scalar_funcs.append((ad_math.nextafter, math.nextafter))


@pytest.mark.parametrize("func,x,y,xd", PARAMETERS_FOR_UNARY_FUNC)
def test_unary_math_functions_for_adj(func, x, y, xd):
    assert func(x) == pytest.approx(y)
    x_ad = Areal(x)

    with Tape() as tape:
        tape.registerInput(x_ad)
        tape.newRecording()

        y_ad = func(x_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize("func,x,y,yd", PARAMETERS_FOR_UNARY_FUNC)
def test_unary_math_functions_for_fwd(func, x, y, yd):
    x_ad = Freal(x)
    x_ad.setDerivative(1.0)

    y_ad = func(x_ad)

    assert y_ad == pytest.approx(y)
    assert y_ad.getDerivative() == pytest.approx(yd)


@pytest.mark.parametrize("ad_func, func", _binary_with_scalar_funcs)
@pytest.mark.parametrize("value", [3, 3.1])
def test_binary_function_with_scalar_param(value, ad_func, func):
    assert ad_func(4.1, value) == pytest.approx(func(4.1, value))
    assert ad_func(4, value) == pytest.approx(func(4, value))
    assert ad_func(Freal(4.1), value) == pytest.approx(func(4.1, value))
    assert ad_func(Freal(4), value) == pytest.approx(func(4, value))
    assert ad_func(value, Freal(4.1)) == pytest.approx(func(value, 4.1))
    assert ad_func(value, Freal(4)) == pytest.approx(func(value, 4))


@pytest.mark.parametrize(
    "func, y, derv",
    [
        (0, math.pow(4, 3), pytest.approx(3 * math.pow(4, 3 - 1))),
        (1, math.pow(3, 4), pytest.approx(math.log(3) * math.pow(3, 4))),
    ],
)
def test_pow_for_adj(func, y, derv):
    x_ad = Areal(4.0)
    with Tape() as tape:
        tape.registerInput(x_ad)
        tape.newRecording()

        if func == 0:
            y_ad = ad_math.pow(x_ad, 3)
        else:
            y_ad = ad_math.pow(3, x_ad)

        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(derv)


@pytest.mark.parametrize(
    "func, y, derv",
    [
        (0, math.pow(4, 3), pytest.approx(3 * math.pow(4, 3 - 1))),
        (1, math.pow(3, 4), pytest.approx(math.log(3) * math.pow(3, 4))),
    ],
)
def test_pow_for_fwd(func, y, derv):
    x_ad = Freal(4.0)
    x_ad.setDerivative(1.0)

    if func == 0:
        y_ad = ad_math.pow(x_ad, 3)
    else:
        y_ad = ad_math.pow(3, x_ad)

    assert y_ad == y
    assert y_ad.getDerivative() == pytest.approx(derv)


@pytest.mark.parametrize("func,x1, x2,y,xd1, xd2", PARAMETERS_FOR_BINARY_FUNC)
def test_binary_math_functions_for_adj(func, x1, x2, y, xd1, xd2):
    x1_ad = Areal(x1)
    x2_ad = Areal(x2)
    with Tape() as tape:
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.newRecording()
        y_ad = func(x1_ad, x2_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(y)
        assert x1_ad.getDerivative() == pytest.approx(xd1)
        assert x2_ad.getDerivative() == pytest.approx(xd2)


@pytest.mark.parametrize("func,x1, x2,y,xd1, xd2", PARAMETERS_FOR_BINARY_FUNC)
@pytest.mark.parametrize("deriv", [1, 2])
def test_binary_math_functions_for_fwd(func, x1, x2, y, xd1, xd2, deriv):
    x1_ad = Freal(x1)
    x2_ad = Freal(x2)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    else:
        x2_ad.setDerivative(1.0)

    y_ad = func(x1_ad, x2_ad)

    assert y_ad == pytest.approx(y)
    if deriv == 1:
        assert y_ad.getDerivative() == pytest.approx(xd1)
    else:
        assert y_ad.getDerivative() == pytest.approx(xd2)


@pytest.mark.parametrize(
    "func,x,y,xd",
    [
        (ad_math.modf, 3.23, math.modf(3.23), 1),
        (ad_math.frexp, 3, math.frexp(3), 1 / math.pow(2, 2)),
    ],
)
def test_modf_frexp_functions_for_adj(func, x, y, xd):
    x_ad = Areal(x)
    with Tape() as tape:
        tape.registerInput(x_ad)
        tape.newRecording()

        y_ad = func(x_ad)
        tape.registerOutput(y_ad[0])
        y_ad[0].setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize(
    "func,x,y,xd",
    [
        (ad_math.modf, 3.23, math.modf(3.23), 1),
        (ad_math.frexp, 3, math.frexp(3), 1 / math.pow(2, 2)),
    ],
)
def test_modf_frexp_functions_for_fwd(func, x, y, xd):
    x_ad = Freal(x)
    x_ad.setDerivative(1.0)

    y_ad = func(x_ad)

    assert y_ad == pytest.approx(y)
    assert y_ad[0].getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize(
    "func, y, xd",
    [
        (ad_math.degrees, math.degrees(3), 180 / math.pi),
        (ad_math.radians, math.radians(3), math.pi / 180),
    ],
)
def test_degrees_radians_adj(func, y, xd):
    with Tape() as tape:
        x_ad = Areal(3.0)
        tape.registerInput(x_ad)
        tape.newRecording()

        y_ad = func(x_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad.getValue() == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize(
    "func, y, xd",
    [
        (ad_math.degrees, math.degrees(3), 180 / math.pi),
        (ad_math.radians, math.radians(3), math.pi / 180),
    ],
)
def test_degrees_radians_fwd(func, y, xd):
    x_ad = Freal(3)
    x_ad.setDerivative(1.0)

    y_ad = func(x_ad)

    assert y_ad.getValue() == pytest.approx(y)
    assert y_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize("Real", [Areal, Freal])
def test_copysign(Real):
    assert ad_math.copysign(Real(-3.1), 4) == pytest.approx(math.copysign(-3.1, 4))
    assert ad_math.copysign(Real(4), -3.1) == pytest.approx(math.copysign(4, -3.1))
    assert ad_math.copysign(Real(-3.1), Real(4)) == pytest.approx(math.copysign(-3.1, 4))


def test_copysign_derivative_for_adj():
    with Tape() as tape:
        x1_ad = Areal(-3.1)
        x2_ad = Areal(4)
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.newRecording()

        y_ad = ad_math.copysign(x1_ad, x2_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(math.copysign(-3.1, 4))
        assert x1_ad.getDerivative() == pytest.approx(-1)
        assert x2_ad.getDerivative() == pytest.approx(0)


@pytest.mark.parametrize("deriv", [1, 2])
def test_copysign_derivative_for_fwd(deriv):
    x1_ad = Freal(-3.1)
    x2_ad = Freal(4)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    else:
        x2_ad.setDerivative(1.0)

    y_ad = ad_math.copysign(x1_ad, x2_ad)

    assert y_ad == pytest.approx(math.copysign(-3.1, 4))
    if deriv == 1:
        assert y_ad.getDerivative() == pytest.approx(-1)
    else:
        assert y_ad.getDerivative() == pytest.approx(0)


def test_sum_adj():
    with Tape() as tape:
        x1_ad = Areal(-3.1)
        x2_ad = Areal(4)
        x3_ad = Areal(2.4)
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.registerInput(x3_ad)
        tape.newRecording()

        y_ad = sum([x1_ad, x2_ad, x3_ad])
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(3.3)
        assert x1_ad.getDerivative() == pytest.approx(1)
        assert x2_ad.getDerivative() == pytest.approx(1)
        assert x3_ad.getDerivative() == pytest.approx(1)


@pytest.mark.parametrize("deriv", [1, 2])
def test_sum_for_fwd(deriv):
    x1_ad = Freal(-3.1)
    x2_ad = Freal(4)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    else:
        x2_ad.setDerivative(1.0)

    y_ad = sum([x1_ad, x2_ad])

    assert y_ad == pytest.approx(sum([-3.1, 4]))
    assert y_ad.getDerivative() == pytest.approx(1)


def test_hypot_adj():
    with Tape() as tape:
        x1_ad = Areal(-3.1)
        x2_ad = Areal(4)
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.newRecording()

        y_ad = ad_math.hypot(x1_ad, x2_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(math.hypot(-3.1, 4))
        assert x1_ad.getDerivative() == pytest.approx(-3.1 / math.hypot(-3.1, 4))
        assert x2_ad.getDerivative() == pytest.approx(4 / math.hypot(-3.1, 4))


@pytest.mark.parametrize("deriv", [1, 2])
def test_hypot_for_fwd(deriv):
    x1_ad = Freal(-3.1)
    x2_ad = Freal(4)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    else:
        x2_ad.setDerivative(1.0)

    y_ad = ad_math.hypot(x1_ad, x2_ad)

    assert y_ad == pytest.approx(math.hypot(-3.1, 4))
    if deriv == 1:
        assert y_ad.getDerivative() == pytest.approx(-3.1 / math.hypot(-3.1, 4))
    else:
        assert y_ad.getDerivative() == pytest.approx(4 / math.hypot(-3.1, 4))


def test_dist_for_adj():
    with Tape() as tape:
        x1_ad = Areal(-3.1)
        x2_ad = Areal(4)
        x3_ad = Areal(2.4)
        x4_ad = Areal(1)
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.registerInput(x3_ad)
        tape.registerInput(x4_ad)
        tape.newRecording()

        y_ad = ad_math.dist([x1_ad, x2_ad], [x3_ad, x4_ad])
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad == pytest.approx(math.dist([-3.1, 4], [2.4, 1]))
        assert x1_ad.getDerivative() == pytest.approx(-5.5 / math.dist([-3.1, 4], [2.4, 1]))
        assert x2_ad.getDerivative() == pytest.approx(3 / math.dist([-3.1, 4], [2.4, 1]))
        assert x3_ad.getDerivative() == pytest.approx(5.5 / math.dist([-3.1, 4], [2.4, 1]))
        assert x4_ad.getDerivative() == pytest.approx(-3 / math.dist([-3.1, 4], [2.4, 1]))


@pytest.mark.parametrize("deriv", [1, 2, 3, 4])
def test_dist_for_fwd(deriv):
    x1_ad = Freal(-3.1)
    x2_ad = Freal(4)
    x3_ad = Freal(2.4)
    x4_ad = Freal(1)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    elif deriv == 2:
        x2_ad.setDerivative(1.0)
    elif deriv == 3:
        x3_ad.setDerivative(1.0)
    else:
        x4_ad.setDerivative(1.0)

    y_ad = ad_math.dist([x1_ad, x2_ad], [x3_ad, x4_ad])

    assert y_ad == pytest.approx(math.dist([-3.1, 4], [2.4, 1]))
    if deriv == 1:
        assert y_ad.getDerivative() == pytest.approx(-5.5 / math.dist([-3.1, 4], [2.4, 1]))
    elif deriv == 2:
        assert y_ad.getDerivative() == pytest.approx(3 / math.dist([-3.1, 4], [2.4, 1]))
    elif deriv == 3:
        assert y_ad.getDerivative() == pytest.approx(5.5 / math.dist([-3.1, 4], [2.4, 1]))
    else:
        assert y_ad.getDerivative() == pytest.approx(-3 / math.dist([-3.1, 4], [2.4, 1]))

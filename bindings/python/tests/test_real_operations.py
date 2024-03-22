##############################################################################
#
#  Pytests for operations and derivatives on the active types
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

from pytest import approx, raises
import pytest
from xad_autodiff.adj_1st import Real as AReal, Tape
from xad_autodiff.fwd_1st import Real as FReal
from xad_autodiff import value, derivative
import math as m


# This is a list of math functions with their expected outcomes and derivatives,
# used in parametrised tests, for binary arithmetic functions with one active operand.
#
# The format is a list of tuples, where each tuple has the following entries:
# - math function: Callable (lambda), with one parameter
# - parameter1 value for the function: float
# - expected result: float
# - expected derivative1 value: float
#
PARAMETERS_FOR_BINARY_ARITHMETICS_1_ACTIVE_OPERAND = [
    (lambda a: 2 * a, 3, 6, 2),
    (lambda a: 2 + a, 3, 5, 1),
    (lambda a: 2 - a, 3, -1, -1),
    (lambda a: 2 / a, 3, 2 / 3, -2 / 9),
    (lambda a: a * 3.6, 3, 10.8, 3.6),
    (lambda a: a + 3.9, 3, 6.9, 1),
    (lambda a: a - 4.3, 3, -1.3, 1),
    (lambda a: a / 2, 3, 1.5, 1 / 2),
]

# This is a list of math functions with their expected outcomes and derivatives,
# used in parametrised tests, for unary arithmetic functions (+x, -x).
#
# The format is a list of tuples, where each tuple has the following entries:
# - math function: Callable (lambda), with one parameter
# - parameter1 value for the function: float
# - expected result: float
# - expected derivative1 value: float
#
PARAMETERS_FOR_UNARY_ARITHMETICS = [(lambda a: +a, 3, 3, 1), (lambda a: -a, 3, -3, -1)]

# This is a list of math functions with their expected outcomes and derivatives,
# used in parametrised tests, for binary arithmetic functions with two active operands.
#
# The format is a list of tuples, where each tuple has the following entries:
# - math function: Callable (lambda, 2 parameters)
# - parameter1 value for the function: float
# - parameter2 value for the function: float
# - expected result: float
# - expected derivative1 value: float
# - expected derivative2 value: float
#
PARAMETERS_FOR_BINARY_ARITHMETICS_2_ACTIVE_OPERANDS = [
    (lambda a, b: a * b, 5.0, 2.0, 10, 2, 5),
    (lambda a, b: a + b, 5.0, 2.0, 7, 1, 1),
    (lambda a, b: a - b, 5.0, 2.0, 3, 1, -1),
    (lambda a, b: a / b, 5.0, 2.0, 2.5, 0.5, -1.25),
]


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_initialize_from_float(ad_type):
    assert ad_type(0.3).getValue() == approx(0.3)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_initialize_from_int(ad_type):
    assert ad_type(1).getValue() == 1


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_add_float(ad_type):
    real = ad_type(0.4) + 0.3
    assert real.getValue() == approx(0.7)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_add_int(ad_type):
    real = ad_type(1) + 2
    assert real.getValue() == 3


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_float(ad_type):
    real = ad_type(0.3) - 0.4
    assert real.getValue() == approx(-0.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_int(ad_type):
    real = ad_type(1) - 2
    assert real.getValue() == -1


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_add_to_float(ad_type):
    real = 0.3 + ad_type(0.4)
    assert real.getValue() == approx(0.7)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_to_float(ad_type):
    real = 2.5 - ad_type(2.0)
    assert real.getValue() == approx(0.5)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_add_to_int(ad_type):
    real = 2 + ad_type(1)
    assert real.getValue() == 3


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_to_int(ad_type):
    real = 2 - ad_type(1)
    assert real.getValue() == 1


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_add_real(ad_type):
    real = ad_type(2) + ad_type(1)
    assert real.getValue() == 3


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_real(ad_type):
    real = ad_type(2) - ad_type(1)
    assert real.getValue() == 1


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mul_float(ad_type):
    real = ad_type(0.2) * 0.5
    assert real.getValue() == approx(0.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mul_int(ad_type):
    real = ad_type(1) * 2
    assert real.getValue() == 2


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mul_to_float(ad_type):
    real = 0.5 * ad_type(0.2)
    assert real.getValue() == approx(0.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mul_to_int(ad_type):
    real = 2 * ad_type(1)
    assert real.getValue() == 2


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mul_real(ad_type):
    real = ad_type(0.2) * ad_type(0.5)
    assert real.getValue() == approx(0.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_div_float(ad_type):
    real = ad_type(0.2) / 0.5
    assert real.getValue() == approx(0.4)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_div_int(ad_type):
    real = ad_type(1) / 2
    assert real.getValue() == approx(0.5)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_div_to_float(ad_type):
    real = 0.5 / ad_type(0.2)
    assert real.getValue() == approx(2.5)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_div_to_int(ad_type):
    real = 2 / ad_type(1)
    assert real.getValue() == 2


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_div_real(ad_type):
    real = ad_type(0.2) / ad_type(0.5)
    assert real.getValue() == approx(0.4)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_addition_assignment_int(ad_type):
    real = ad_type(0.2)
    real += 1
    assert real.getValue() == approx(1.2)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_addition_assignment_float(ad_type):
    real = ad_type(0.2)
    real += 1.9
    assert real.getValue() == approx(2.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_addition_assignment_real(ad_type):
    real = ad_type(0.2)
    real += ad_type(0.5)
    assert real.getValue() == approx(0.7)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_assignment_int(ad_type):
    real = ad_type(0.2)
    real -= 1
    assert real.getValue() == approx(-0.8)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_assignment_float(ad_type):
    real = ad_type(0.2)
    real -= 1.9
    assert real.getValue() == approx(-1.7)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_sub_assignment_real(ad_type):
    real = ad_type(0.2)
    real -= ad_type(0.5)
    assert real.getValue() == approx(-0.3)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_comparison_real_to_real(ad_type):
    a = ad_type(0.2)
    b = ad_type(0.5)
    assert (b > a) is True
    assert (a > b) is False
    assert (b >= a) is True
    assert (a >= b) is False
    assert (b < a) is False
    assert (a < b) is True
    assert (b <= a) is False
    assert (a <= b) is True
    c = ad_type(0.2)
    assert (a != b) is True
    assert (a != c) is False
    assert (a == c) is True
    assert (b == c) is False


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_comparison_real_to_float(ad_type):
    a = ad_type(0.2)
    b = 0.5
    assert (b > a) is True
    assert (a > b) is False
    assert (b >= a) is True
    assert (a >= b) is False
    assert (b < a) is False
    assert (a < b) is True
    assert (b <= a) is False
    assert (a <= b) is True
    c = 0.2
    assert (a != b) is True
    assert (a != c) is False
    assert (a == c) is True
    assert (b == c) is False


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_comparison_real_to_int(ad_type):
    a = ad_type(2)
    b = 5
    assert (b > a) is True
    assert (a > b) is False
    assert (b >= a) is True
    assert (a >= b) is False
    assert (b < a) is False
    assert (a < b) is True
    assert (b <= a) is False
    assert (a <= b) is True
    c = 2
    assert (a != b) is True
    assert (a != c) is False
    assert (a == c) is True
    assert (b == c) is False


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_rounding(ad_type):
    assert round(ad_type(2.345), 2) == pytest.approx(2.35)
    assert round(ad_type(2.345), 1) == pytest.approx(2.3)
    assert round(ad_type(2.345), 0) == pytest.approx(2.0)
    assert round(ad_type(2.345)) == pytest.approx(2.0)
    assert type(round(ad_type(2.3))) == type(round(2.3))


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
@pytest.mark.parametrize(
    "func", [m.ceil, m.floor, m.trunc, int], ids=["ceil", "floor", "trunc", "int"]
)
def test_truncation_funcs(ad_type, func):
    assert func(ad_type(2.345)) == func(2.345)
    assert func(ad_type(2.845)) == func(2.845)
    assert func(ad_type(-2.845)) == func(-2.845)
    assert func(ad_type(0.0)) == func(0.0)
    assert type(func(ad_type(1.1))) == int


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_abs(ad_type):
    assert abs(ad_type(2.345)) == pytest.approx(2.345)
    assert abs(ad_type(-2.345)) == pytest.approx(2.345)
    assert abs(ad_type(0.0)) == 0.0


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_bool(ad_type):
    assert bool(ad_type(1.0)) is bool(1.0)
    assert bool(ad_type(0.0)) is bool(0.0)
    assert bool(ad_type(1.0)) is bool(-1.0)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_mod(ad_type):
    assert ad_type(2.7) % 2 == 2.7 % 2
    assert ad_type(2.7) % ad_type(2.0) == 2.7 % 2.0
    assert 2.7 % ad_type(2.0) == 2.7 % 2.0
    assert 2 % ad_type(2.0) == 2 % 2.0
    assert ad_type(-2.7) % 2 == -2.7 % 2
    assert ad_type(-2.7) % ad_type(2.0) == -2.7 % 2.0
    assert -2.7 % ad_type(2.0) == -2.7 % 2.0
    assert -2 % ad_type(2.0) == -2 % 2.0
    assert ad_type(2.7) % -2 == 2.7 % -2
    assert ad_type(2.7) % ad_type(-2.0) == 2.7 % -2.0
    assert 2.7 % ad_type(-2.0) == 2.7 % -2.0
    assert 2 % ad_type(-2.0) == 2 % -2.0


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_divmod(ad_type):
    assert divmod(ad_type(2.7), 2) == divmod(2.7, 2)
    assert divmod(ad_type(2.7), ad_type(2.0)) == divmod(2.7, 2.0)
    assert divmod(2.7, ad_type(2.0)) == divmod(2.7, 2.0)
    assert divmod(2, ad_type(2.0)) == divmod(2, 2.0)
    assert divmod(ad_type(-2.7), 2) == divmod(-2.7, 2)
    assert divmod(ad_type(-2.7), ad_type(2.0)) == divmod(-2.7, 2.0)
    assert divmod(-2.7, ad_type(2.0)) == divmod(-2.7, 2.0)
    assert divmod(-2, ad_type(2.0)) == divmod(-2, 2.0)
    assert divmod(ad_type(2.7), -2) == divmod(2.7, -2)
    assert divmod(ad_type(2.7), ad_type(-2.0)) == divmod(2.7, -2.0)
    assert divmod(2.7, ad_type(-2.0)) == divmod(2.7, -2.0)
    assert divmod(2, ad_type(-2.0)) == divmod(2, -2.0)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_floordiv(ad_type):
    assert ad_type(2.7) // 2 == 2.7 // 2
    assert ad_type(2.7) // ad_type(2.0) == 2.7 // 2.0
    assert 2.7 // ad_type(2.0) == 2.7 // 2.0
    assert 2 // ad_type(2.0) == 2 // 2.0
    assert ad_type(-2.7) // 2 == -2.7 // 2
    assert ad_type(-2.7) // ad_type(2.0) == -2.7 // 2.0
    assert -2.7 // ad_type(2.0) == -2.7 // 2.0
    assert -2 // ad_type(2.0) == -2 // 2.0
    assert ad_type(2.7) // -2 == 2.7 // -2
    assert ad_type(2.7) // ad_type(-2.0) == 2.7 // -2.0
    assert 2.7 // ad_type(-2.0) == 2.7 // -2.0
    assert 2 // ad_type(-2.0) == 2 // -2.0


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_pow_operator(ad_type):
    assert ad_type(2.7) ** 2 == pytest.approx(2.7**2)
    assert ad_type(2.7) ** 2.4 == pytest.approx(2.7**2.4)
    assert ad_type(2.7) ** ad_type(2.4) == pytest.approx(2.7**2.4)
    assert 2.7 ** ad_type(2.4) == pytest.approx(2.7**2.4)
    assert 2 ** ad_type(2.4) == pytest.approx(2**2.4)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_hash_method(ad_type):
    assert hash(ad_type(2.7)) == hash(2.7)
    assert hash(ad_type(-2.7)) == hash(-2.7)
    assert hash(ad_type(0)) == hash(0)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_getnewargs_method(ad_type):
    assert ad_type(1.2).__getnewargs__() == (1.2,)
    assert ad_type(1).__getnewargs__() == (1.0,)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_as_integer_ratio(ad_type):
    assert ad_type(1.2).as_integer_ratio() == (1.2).as_integer_ratio()
    assert ad_type(-21.2).as_integer_ratio() == (-21.2).as_integer_ratio()


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_conjugate(ad_type):
    assert ad_type(1.2).conjugate() == (1.2).conjugate()
    assert ad_type(-21.2).conjugate() == (-21.2).conjugate()


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_fromhex(ad_type):
    assert ad_type.fromhex("0x3.a7p10") == float.fromhex("0x3.a7p10")


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_hex(ad_type):
    assert ad_type(1.23).hex() == (1.23).hex()


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_imag(ad_type):
    assert ad_type(1.23).imag() == pytest.approx(0.0)
    assert ad_type(-1.23).imag() == pytest.approx(0.0)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_real(ad_type):
    assert ad_type(1.23).real() == pytest.approx(1.23)
    assert ad_type(-1.23).real() == pytest.approx(-1.23)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_isinteger(ad_type):
    assert ad_type(1.23).is_integer() is False
    assert ad_type(-1.23).is_integer() is False
    assert ad_type(21.0).is_integer() is True
    assert ad_type(-1234.0).is_integer() is True


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_format(ad_type):
    assert f"{ad_type(1.23):10.5f}" == f"{1.23:10.5f}"
    assert "{:10.5f}".format(ad_type(1.23)) == "{:10.5f}".format(1.23)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_value_function(ad_type):
    assert value(3) == 3
    assert value(3.2) == approx(3.2)
    assert value("3") == "3"
    assert value(ad_type(3.1)) == approx(3.1)


@pytest.mark.parametrize("ad_type", [AReal, FReal], ids=["adj", "fwd"])
def test_value_property_get(ad_type):
    assert ad_type(3.1).value == approx(3.1)


def test_derivative_property_get_fwd():
    x = FReal(1.2)
    x.setDerivative(1.0)
    assert x.derivative == 1.0


def test_derivative_property_set_fwd():
    x = FReal(1.2)
    x.derivative = 1.0
    assert x.getDerivative() == 1.0


def test_derivative_property_get_adj():
    x = AReal(1.2)
    with Tape() as tape:
        tape.registerInput(x)
        tape.newRecording()
        y = x
        tape.registerOutput(y)
        y.setDerivative(1.0)
        assert y.derivative == 1.0


def test_derivative_property_set_adj():
    x = AReal(1.2)
    with Tape() as tape:
        tape.registerInput(x)
        tape.newRecording()
        y = x
        tape.registerOutput(y)
        y.derivative = 1.0
        assert y.getDerivative() == 1.0


def test_derivative_function():
    x = AReal(3.2)

    with Tape() as tape:
        tape.registerInput(x)
        tape.newRecording()

        y_ad = x
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()
        assert derivative(x) == x.getDerivative()
        with raises(TypeError):
            derivative(1)


def test_should_record():
    x = AReal(42.0)
    assert x.shouldRecord() is False
    with Tape() as tape:
        tape.registerInput(x)
        assert x.shouldRecord() is True


def test_set_adjoint():
    x = AReal(42.0)
    with Tape() as tape:
        tape.registerInput(x)
        tape.newRecording()
        y = 4 * x
        tape.registerOutput(x)
        y.setAdjoint(1.0)
        tape.computeAdjoints()
        assert derivative(x) == 4.0


@pytest.mark.parametrize("func,x,y,xd", PARAMETERS_FOR_UNARY_ARITHMETICS)
def test_unary_arithmetics_adj(func, x, y, xd):
    x_ad = AReal(x)

    with Tape() as tape:
        tape.registerInput(x_ad)
        tape.newRecording()

        y_ad = func(x_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad.getValue() == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize("func,x,y,xd", PARAMETERS_FOR_UNARY_ARITHMETICS)
def test_unary_arithmetics_fwd(func, x, y, xd):
    x_ad = FReal(x)
    x_ad.setDerivative(1.0)

    y_ad = func(x_ad)

    assert y_ad == y
    assert y_ad.getDerivative() == xd


@pytest.mark.parametrize("func,x,y,xd", PARAMETERS_FOR_BINARY_ARITHMETICS_1_ACTIVE_OPERAND)
def test_binary_arithmetics_fwd(func, x, y, xd):
    x1_ad = FReal(x)
    x1_ad.setDerivative(1.0)
    y_ad = func(x1_ad)
    assert y_ad.getValue() == pytest.approx(y)
    assert y_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize("func,x,y,xd", PARAMETERS_FOR_BINARY_ARITHMETICS_1_ACTIVE_OPERAND)
def test_binary_arithmetics_adj(func, x, y, xd):
    x_ad = AReal(x)

    with Tape() as tape:
        tape.registerInput(x_ad)
        tape.newRecording()

        y_ad = func(x_ad)
        tape.registerOutput(y_ad)
        y_ad.setDerivative(1.0)

        tape.computeAdjoints()

        assert y_ad.getValue() == pytest.approx(y)
        assert x_ad.getDerivative() == pytest.approx(xd)


@pytest.mark.parametrize(
    "func,x1, x2,y,xd1, xd2", PARAMETERS_FOR_BINARY_ARITHMETICS_2_ACTIVE_OPERANDS
)
def test_binary_with_2_active_operands_adj(func, x1, x2, y, xd1, xd2):
    x1_ad = AReal(x1)
    x2_ad = AReal(x2)

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


@pytest.mark.parametrize(
    "func,x1, x2,y,xd1, xd2", PARAMETERS_FOR_BINARY_ARITHMETICS_2_ACTIVE_OPERANDS
)
@pytest.mark.parametrize("deriv", [1, 2])
def test_binary_with_2_active_operands_fwd(func, x1, x2, y, xd1, xd2, deriv):
    x1_ad = FReal(x1)
    x2_ad = FReal(x2)
    if deriv == 1:
        x1_ad.setDerivative(1.0)
    else:
        x2_ad.setDerivative(1.0)

    y_ad = func(x1_ad, x2_ad)
    assert y_ad.getValue() == pytest.approx(y)
    if deriv == 1:
        assert y_ad.getDerivative() == pytest.approx(xd1)
    else:
        assert y_ad.getDerivative() == pytest.approx(xd2)

##############################################################################
#
#  Test the adjoint tape in Python.
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


import pytest
from xad_autodiff import derivative, exceptions, value
from xad_autodiff.adj_1st import Tape, Real


def test_active_tape():
    tape = Tape()
    assert tape.isActive() is False
    tape.activate()
    assert tape.isActive() is True
    tape.deactivate()
    assert tape.isActive() is False


def test_tape_using_with():
    with Tape() as tape:
        assert tape.isActive() is True
    tape = Tape()
    assert tape.isActive() is False
    with tape:
        assert tape.isActive() is True
    assert tape.isActive() is False


def test_get_active():
    t = Tape()
    assert Tape.getActive() is None
    t.activate()
    assert Tape.getActive() is not None
    assert Tape.getActive() == t


def test_get_position():
    with Tape() as t:
        assert t.getPosition() == 0
        x1 = Real(1.0)
        t.registerInput(x1)
        x2 = 1.2 * x1
        x1.setDerivative(1.0)
        t.registerOutput(x2)
        t.computeAdjoints()
        assert t.getPosition() >= 0


def test_clear_derivative_after():
    with Tape() as tape:
        x1 = Real(1.0)
        tape.registerInput(x1)
        x2 = 1.2 * x1
        pos = tape.getPosition()
        x3 = 1.4 * x2 * x1
        x4 = x2 + x3
        tape.registerOutput(x4)
        x4.setDerivative(1.0)
        x3.setDerivative(1.0)
        x2.setDerivative(1.0)
        x1.setDerivative(1.0)
        tape.clearDerivativesAfter(pos)

        assert derivative(x2) == 1.0
        assert derivative(x1) == 1.0
        with pytest.raises(exceptions.OutOfRange) as e:
            derivative(x3)
        assert "given derivative slot is out of range - did you register the outputs?" in str(e)
        with pytest.raises(exceptions.OutOfRange) as e:
            derivative(x4)
        assert "given derivative slot is out of range - did you register the outputs?" in str(e)


def test_reset_to_and_compute_adjoints_to_usage():
    i = Real(2.0)
    with Tape() as tape:
        tape.registerInput(i)
        tape.newRecording()
        pos = tape.getPosition()
        values = []
        deriv = []
        for p in range(1, 10):
            v = p * i
            tape.registerOutput(v)
            v.setDerivative(1.0)
            tape.computeAdjointsTo(pos)
            values.append(value(v))
            deriv.append(derivative(i))
            tape.resetTo(pos)
            tape.clearDerivatives()

        assert values == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        assert deriv == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_derivative():
    with Tape() as t:
        x = Real(1.0)
        t.registerInput(x)
        assert t.derivative(x) == 0.0


def test_get_derivative():
    with Tape() as t:
        x = Real(1.0)
        t.registerInput(x)
        assert t.getDerivative(x) == 0.0


def test_set_derivative_value():
    with Tape() as t:
        x = Real(1.0)
        t.registerInput(x)
        t.setDerivative(x, 1.0)
        assert t.derivative(x) == 1.0
        with pytest.raises(exceptions.OutOfRange):
            derivative(t.setDerivative(1231, 0.0))


def test_set_derivative_slot():
    with Tape() as t:
        x = Real(1.0)
        t.registerInput(x)
        slot = x.getSlot()
        assert isinstance(slot, int)
        t.setDerivative(slot, 1.0)
        assert t.derivative(x) == 1.0
        assert t.getDerivative(slot) == 1.0

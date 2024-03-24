##############################################################################
#
#  Test exceptions bindings.
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
from xad_autodiff.adj_1st import Real, Tape
from xad_autodiff.exceptions import (
    XadException,
    TapeAlreadyActive,
    OutOfRange,
    DerivativesNotInitialized,
    NoTapeException,
)


@pytest.mark.parametrize("exception", [TapeAlreadyActive, XadException])
def test_exceptions_tape_active(exception):
    with Tape() as t:
        with pytest.raises(exception) as e:
            # when it's already active
            t.activate()
        assert "A tape is already active for the current thread" in str(e)


@pytest.mark.parametrize("exception", [OutOfRange, XadException])
def test_exceptions_outofrange(exception):
    with Tape() as t:
        x = Real(1.0)
        t.registerInput(x)
        assert t.derivative(x) == 0.0
        with pytest.raises(exception) as e:
            t.derivative(12312)
        assert "given derivative slot is out of range - did you register the outputs?" in str(e)


@pytest.mark.parametrize("exception", [DerivativesNotInitialized, XadException])
def test_exceptions_adjoints_not_initialized(exception):
    with Tape() as t:
        with pytest.raises(exception) as e:
            x = Real(1.0)
            t.registerInput(x)
            t.newRecording()
            y = x * x
            t.registerOutput(y)
            t.computeAdjoints()
        assert "At least one derivative must be set before computing adjoint" in str(e)


@pytest.mark.parametrize("exception", [NoTapeException, XadException])
def test_exceptions_no_tape_exception(exception):
    with pytest.raises(exception) as e:
        x = Real(1.0)
        x.setDerivative(1.0)
    assert "No active tape for the current thread" in str(e)

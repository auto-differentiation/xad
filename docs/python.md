---
description: >
  Python bindings for the XAD automatic differentiation tool.
hide:
  - navigation
---

# Python Bindings

The [Python bindings for XAD](https://pypi.org/project/xad-autodiff/) are available on PyPi for all major platforms and operating systems.
There are published with each XAD release, using the same versioning scheme as the C++ version.

## Installation

The XAD Python bindings can be installed as usual using `pip` or any other package manager:

```
pip install xad-autodiff
```

## Usage

The following example for first-order adjoint mode illustrates how to use it:

```python
import xad_autodiff.adj_1st as xadj


# set independent variables
x0_ad = xadj.Real(1.0)
x1_ad = xadj.Real(1.5)
x2_ad = xadj.Real(1.3)
x3_ad = xadj.Real(1.2)

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
```

The Python bindings follow largely the same syntax and workflow as in C++.

### Modules and Naming

| Module | Description | Contents |
|--------|-------------|---------|
| `xad_autodiff`  | The main module, which contain global functions and subpackages | `value`, `derivative` |
| `xad_autodiff.exceptions` | Contains all exceptions, with the same names as described in [Exceptions](ref/exceptions.md) | e.g. `NoTapeException` |
| `xad_autodiff.math` | Mirrors Python's `math` module, with functions for XAD's active types. | e.g. `sin`, `exp` |
| `xad_autodiff.fwd_1st` | Active type for first-order forward mode | `Real` |
| `xad_autodiff.adj_1st` | Active type for first-order adjoint mode as well as the corresponding tape type | `Real`, `Tape` |

### Differences to C++

- Only first order forward mode (module `xad_autodiff.fwd_1st`) and first order adjoint mode are supported (module `xad_autodiff.adj_1st`)
- The active type is called `Real` in all modes
- In adjoint mode, a newly constructed `Tape` object is not automatically activated on construction. It can be activated using `tape.activate()` later, but we recommend using a `with` block as illustrated in the example above.
- The math functions in `xad_autodiff.math` have been designed as a drop-in replacement for the standard Python `math` module. They not only support calls with XAD's active type, but also with regular `float` variables.
- Checkpointing and external function features are not yet supported in Python.
- The `x.getDerivative()` and `x.setDerivative()` methods of active types are also available as the Python property `x.derivative` with both set and get functionality.
- The `x.getValue()` method of active types is also available as the read-only property `x.value`
- Since Python does not allow setting references from function return values, the C++ syntax using the global function `derivative(y) = 1.0` is not possible in Python. Instead, use `y.setDerivative(1.0)` or the property setter `y.derivative = 1.0`. 
- Complex numbers are not yet supported in the Python bindings.

## Examples

Please see the `bindings/python/samples` folder as a starting point to illustrate how to use the Python bindings.


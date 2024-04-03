---
title: "Python Guide & Performance Benchmark"
description: "Efficiently perform automatic differentiation in Python and benefit from huge performance gain for financial risk assessments using QuantLib-Risks powered by XAD."
hide:
  - toc
---


The following example for first-order adjoint mode illustrates how to use it:

```python
import xad.adj_1st as xadj


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

**Modules and Naming**

| Module | Description | Contents |
|--------|-------------|---------|
| `xad`  | The main module, which contain global functions and subpackages | `value`, `derivative` |
| `xad.exceptions` | Contains all exceptions, with the same names as described in [Exceptions](../ref/exceptions.md) | e.g. `NoTapeException` |
| `xad.math` | Mirrors Python's `math` module, with functions for XAD's active types. | e.g. `sin`, `exp` |
| `xad.fwd_1st` | Active type for first-order forward mode | `Real` |
| `xad.adj_1st` | Active type for first-order adjoint mode as well as the corresponding tape type | `Real`, `Tape` |

**Notes**

- First order forward mode (module `xad.fwd_1st`) and first order adjoint mode are supported (module `xad.adj_1st`)
- The active type is called `Real` in all modes
- In adjoint mode, a newly constructed `Tape` object is not automatically activated on construction. It can be activated using `tape.activate()` later, but we recommend using a `with` block as illustrated in the example above.
- The math functions in `xad.math` have been designed as a drop-in replacement for the standard Python `math` module. They not only support calls with XAD's active type, but also with regular `float` variables.
- Checkpointing and external function features are not yet supported in Python.
- The `x.getDerivative()` and `x.setDerivative()` methods of active types are also available as the Python property `x.derivative` with both set and get functionality.
- The `x.getValue()` method of active types is also available as the read-only property `x.value`
- Use `y.setDerivative(1.0)` or the property setter `y.derivative = 1.0` to seed and
  access derivatives. 
- Complex numbers are not yet supported in the Python bindings.


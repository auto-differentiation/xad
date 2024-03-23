[![Python](https://img.shields.io/pypi/pyversions/xad-autodiff.svg)](https://auto-differentiation.github.io/python)
[![PyPI version](https://badge.fury.io/py/xad-autodiff.svg)](https://pypi.org/project/xad-autodiff/)


XAD is a cutting-edge library designed for automatic differentiation, tailored for both novices and experts. This library excels in production environments, offering unparalleled performance with an emphasis on ease of use. [Automatic differentiation](https://auto-differentiation.github.io/aad/), a critical technique for computing derivatives within computer programmes, is made efficient and straightforward with XAD. Whether you're performing simple arithmetic or complex mathematical functions, XAD ensures accurate and automatic derivative calculations.

Python developers will find the Python bindings for XAD incredibly useful, featuring:

-   Support for both forward and adjoint modes at the first order.
-   Robust exception-safety guarantees.
-   Unmatched performance, proven in extensive production deployments.
-   Discover how XAD can revolutionise your computational tasks. Dive into our comprehensive [documentation](https://auto-differentiation.github.io/python) and start integrating XAD into your projects today.

## Getting Started

Install:

```
pip install xad-autodiff
```


Calculate first-order derivatives in adjoint mode:

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

For more information, see the [Documentation](https://auto-differentiation.github.io/python).

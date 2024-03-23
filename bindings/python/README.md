[![Python](https://img.shields.io/pypi/pyversions/xad-autodiff.svg)](https://auto-differentiation.github.io/python)
[![PyPI version](https://badge.fury.io/py/xad-autodiff.svg)](https://pypi.org/project/xad-autodiff/)


XAD is a library designed for [automatic differentiation](https://auto-differentiation.github.io/aad/), aimed at both beginners and advanced users. It is intended for use in production environments, emphasizing performance and ease of use. The library facilitates the computation of derivatives within computer programs, making the process efficient and straightforward for a wide range of mathematical functions, from simple arithmetic to complex calculations, ensuring accurate and automatic derivative computations.

The Python bindings for XAD offer the following features:

- Support for both forward and adjoint modes at the first order.
- Strong exception-safety guarantees.
- High performance, as demonstrated in extensive production use.

For more details and to integrate XAD into your projects, consult the comprehensive [documentation](https://auto-differentiation.github.io/python).

## Application Areas

Automatic differentiation has many application areas, for example:

-   **Machine Learning and Deep Learning:** Training neural networks or other 
    machine learning models.
-   **Optimization:** Solving optimization problems in engineering and finance.
-   **Numerical Analysis:** Enhancing numerical solution methods for 
    differential equations.
-   **Scientific Computing:** Simulating physical systems and processes.
-   **Risk Management and Quantitative Finance:** Assessing and hedging risk in
    financial models.
-   **Computer Graphics:** Optimizing rendering algorithms.
-   **Robotics:** Improving control and simulation of robotic systems.
-   **Meteorology:** Enhancing weather prediction models.
-   **Biotechnology:** Modeling biological processes and systems.

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

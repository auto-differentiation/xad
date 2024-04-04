---
title: "Python QuantLib Integration: QuantLib-Risks"
description: "Boost QuantLib's risk evaluations in Python with XAD's automatic differentiation for unparalleled efficiency and accuracy."
hide:
  - toc
---

# Python QuantLib Integration: QuantLib-Risks

The Python package [QuantLib-Risks](https://pypi.org/project/QuantLib-Risks) is now available on PyPI. This version integrates automatic differentiation capabilities into QuantLib through its dependency on [XAD's Python bindings](../installation/python.md). This integration significantly boosts the efficiency of performing high-performance risk assessments within QuantLib from Python.

The key advantage brought by QuantLib-Risks is its ability to expediently ascertain how the pricing of derivatives is influenced by various input variables, notably market quotes.

The user interface mimics the official [QuantLib Python package](https://pypi.org/project/QuantLib), with the addition of replacing it's `Real` type with `xad.adj_1st.Real`, which can be tracked on an adjoint automatic differentiation tape and derivatives 
can be calculated with [xad](../tutorials/python.md).

## Installation

```
pip install QuantLib-Risks
```

## Usage Illustration

```python
import QuantLib_Risks as ql
from xad.adj_1st import Tape

with Tape() as t:
    rate = ql.Real(0.2)
    tape.registerInput(rate)
    
    # quantlib pricing code, setting up an option
    quote = ql.SimpleQuote(rate)
    ...
    npv = option.NPV()
    

    tape.registerOutput(npv)
    npv.derivative = 1.0
    tape.computeAdjoints()

    print(f"price = {npv}")
    print(f"delta = {rate.derivative}")
```


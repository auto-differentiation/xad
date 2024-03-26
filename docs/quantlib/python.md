---
description: >-
  QuantLib-Risks enables high performance risk evaluations with QuantLib Python bindings
  using XAD automatic differentiation.
---

# Python QuantLib Integration: QuantLib-Risks

The Python package [QuantLib-Risks](https://pypi.org/project/QuantLib-Risks), a fork of the QuantLib Python bindings, is now available on PyPI. This version integrates automatic differentiation capabilities through its dependency on [XAD's Python bindings](../python.md). This integration significantly boosts the efficiency of performing high-performance risk assessments within QuantLib from Python.

The key advantage brought by QuantLib-Risks is its ability to expediently ascertain how the pricing of derivatives is influenced by various input variables, notably market quotes.

The user interface mimics the official [QuantLib Python package](https://pypi.org/project/QuantLib), with the addition of replacing it's `Real` type with `xad-autodiff.adj_1st.Real`, which can be tracked on an adjoint automatic differentiation tape and derivatives 
can be calculated [xad-autodiff](../python.md).


## Performance

To gauge the performance impact of calculating sensitivities, we leverage the [multi-curve bootstrapping example](https://github.com/auto-differentiation/QuantLib-Risks/blob/v1.33/Python/examples/multicurve-bootstrapping.py). This setup incorporates a wide array of quotes to construct a term structure for swap pricing. It prices a forward-starting 5-year swap, set to commence 15 months into the future, with a calculation of 69 sensitivities covering all market quotes used in curve construction and select swap parameters like nominal, fixed rate, and spread.

Performance metrics are drawn from averaging execution times over 20 runs for stability. Initial benchmarks using the standard QuantLib package clock in at 198ms for pricing alone. Switching to `QuantLib-Risks` for simultaneous pricing and sensitivity analysis results in an execution time of 370ms, demonstrating that *all sensitivities can be obtained within approximately 1.87x of the original pricing time*.

Comparatively, a traditional bump-and-reval approach for sensitivities would necessitate 70 pricer code executions (one for valuation and one for each variable bump), translating to 70x the pure pricing time. *Thus, `QuantLib-Risks` achieves a 37.4x speed advantage over bump and reval*.

Summary of performance benchmarks:

| Sensitivities | Valuation (QuantLib) | AAD (QuantLib-Risks) | Bumping (estimate) | AAD vs Valuation | Bumping vs AAD |
|---:|---:|---:|---:|---:|---:|
| 69 | 198ms | 370ms | 13,860ms | 1.87x | 37.4x |

Benchmark configuration details include:
- `QuantLib` version: 1.33
- `QuantLib-Risks` version: 1.33.1
- `xad-autodiff` version: 1.5.0
- Operating on Ubuntu 22.04 with GCC 11.4.0
- Hardware specs include 128GB RAM and an Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz.


## Installation

```
pip install QuantLib-Risks
```

## Usage Illustration

```python
import QuantLib_Risks as ql
from xad_autodiff.adj_1st import Tape

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


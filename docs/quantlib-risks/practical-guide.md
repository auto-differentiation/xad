---
title: "QuantLib-Risks: Enhance Derivative Analysis"
description: "Optimise financial modeling and derivative pricing with QuantLib-Risks. Achieve accurate, fast sensitivity analysis."
---

# Swap Pricing and Curve Construction: A Practical Guide to Sensitivity Analysis Using Python and QuantLib

## Preliminary Notes

Before we dive deeper into our discussion on automatic differentiation (AD) within QuantLib, 
let's set the foundation with several essential insights. 
This note aims to bridge the gap for readers who may be newer to the concept of AD 
in financial modelling and QuantLib, 
or those seeking a nuanced understanding of its application:

- *Demystifying QuantLib-Risks:* Our exploration starts with an illustrative example 
  intended to make the principles of AD within QuantLib more accessible. 
  This example, while simplified for educational purposes, 
  lays the groundwork for appreciating the broader, more complex applications of AD.
- *Implicit Function Theorem in Practice:* Although our example does not employ the implicit
  function theorem, it's important to acknowledge this theorem's significance 
  in practical applications for model calibration. 
  In real-world scenarios, understanding and applying the implicit function theorem 
  is crucial for effective calibration processes in QuantLib with AD.
- *Approaching Discontinuities with Caution:* For the sake of simplicity, 
  our discussion will overlook discontinuities. 
  However, it's essential to recognise that discontinuities pose significant challenges 
  in AD applications. 
  A thorough understanding and careful handling of these are necessary in practical, 
  more complex scenarios. 
  See [Handling Discontinuities](../tutorials/smoothed_math.md) for more details.
- *Tailored Approaches for Effective Implementation:*
  While our example applies XAD in a broad sense, 
  it's critical to understand that in practice, 
  a one-size-fits-all approach is not feasible.
  Each algorithm requires careful evaluation and a customised methodology
  for successful application.
  

## Overview

In this tutorial, we'll explore how to leverage the capabilities of [QuantLib-Risks](https://pypi.org/project/QuantLib-Risks), the QuantLib Python bindings enhanced with automatic differentiation via [XAD](https://pypi.org/project/xad), for efficient risk assessment in financial analysis. This powerful combination allows for rapid determination of how derivative pricing is influenced by various market inputs, crucial for effective risk management.

## Getting Started with QuantLib-Risks for Swap Pricing

### Prerequisites

- Familiarity with Python and basic financial derivatives.
- Installation of [QuantLib-Risks](https://pypi.org/project/QuantLib-Risks) and [xad](https://pypi.org/project/xad) from PyPI.

### Example Overview

We'll base our example on QuantLib's standard [swap pricing example](https://github.com/lballabio/QuantLib-SWIG/blob/v1.33/Python/examples/swap.py). This constructs a financial landscape through a series of market quotes encompassing deposits, FRAs, futures, and swap rates, serving as the building blocks for bootstrapping two distinct curves: `depoFuturesSwapCurve` and `depoFraSwapCurve`. The former curve is built from deposit, future, and swap rate quotes, while the latter emerges from deposits, FRAs, and swap rate quotes.

It then constructs and prices two types of swaps: a `spot` swap, extending over five years from the spot date, and a `forward` swap, also spanning five years but commencing one year into the future. The output from this example is summarised in the following table:

|Swap | Curve | NPV |
|-----|-------|-----|
| `spot` | `depoFuturesSwapCurve` | 19,066.26 |
| `spot` | `depoFraSwapCurve` | 19,066.26 |
| `forward` | `depoFuturesSwapCurve` | 40,533.04 |
| `forward` | `depoFraSwapCurve` | 37,144.28 |



## Sensitivity Analysis with QuantLib-Risks

The core of our tutorial focuses on sensitivity analysis, crucial for understanding how changes in market parameters impact pricing and for devising appropriate hedging strategies.

The full example demonstrating this analysis is available on [GitHub](https://github.com/auto-differentiation/QuantLib-Risks-Py/blob/main/Python/examples/swap-adjoint.py).


### Setup and Imports

First, ensure [QuantLib-Risks](https://pypi.org/project/QuantLib-Risks) is installed (which depends on [xad](https://pypi.org/project/xad)): 

```text
pip install QuantLib-Risks
```

Then, import the necessary modules and activate the automatic differentiation tape:

```python
import QuantLib_Risks as ql
from xad.adj_1st import Tape

tape = Tape()
tape.activate()
```

### Preparing Input Variables

The input market quotes are wrapped in `ql.Real` instances for automatic differentiation
and registered with the tape for sensitivity analysis.
These form the independent input variables.
For example, futures quotes setup looks like this:

```python
futures_in = {
    ql.Date(19, 12, 2001): ql.Real(96.2875),
    ql.Date(20, 3, 2002):  ql.Real(96.7875),
    ql.Date(19, 6, 2002):  ql.Real(96.9875),
    ql.Date(18, 9, 2002):  ql.Real(96.6875),
    ql.Date(18, 12, 2002): ql.Real(96.4875),
    ql.Date(19, 3, 2003):  ql.Real(96.3875),
    ql.Date(18, 6, 2003):  ql.Real(96.2875),
    ql.Date(17, 9, 2003):  ql.Real(96.0875),
}
tape.registerInputs(futures_in.values())
```

### Performing Calculations

Before commencing calculations, we activate the tape to record derivatives:

```python
tape.newRecording()
```

Then we proceed with your calculations as usual in QuantLib, 
creating `ql.SimpleQuote` instances, creating curve bootstrapping helpers,
building the curves, and finally creating the swaps based on these curves. 
This step doesn't differ from the standard QuantLib process.

### Calculating Sensitivities

To extract sensitivities, perform the following steps after computing the NPV:

1. Register the NPV as an output with the tape.
2. Seed the output's adjoint and compute adjoints to propagate sensitivities.
3. Access the derivatives of inputs to understand sensitivities to market quotes.

This process can be neatly encapsulated in a function to show sensitivities for various market quotes:

```python
def show(swap):
    npv = swap.NPV()         # calculate NPV value
    tape.registerOutput(npv) # register output
    tape.clearDerivatives()  # clear previous derivatives (to allow multiple calls)
    npv.derivative = 1.0     # seed output adjoint
    tape.computeAdjoints()   # roll back the tape to calculate input adjoints
    
    print("NPV         = {:.2f}".format(v))
    print("Fair spread = {:.4f} %".format(swap.fairSpread()*100))
    print("Fair rate   = {:.4f} %".format(swap.fairRate()*100))

    print("\nSensitivities to deposit quotes, 1bp shift:")
    for k, v in deposits_in.items():
        print(f"  {ql.Period(k[0], k[1])}: {v.derivative * 0.0001:.2f}")
    print("Sensitivities to FRA quotes, 1bp shift:")
    for k, v in FRAs_in.items():
        print(f"  {k[0]}M - {k[1]}M: {v.derivative * 0.0001:.2f}")
    print("Sensitivities to futures quotes, 1c shift:")
    for k, v in futures_in.items():
        print(f"  {k}: {v.derivative * 0.01:.2f}")
    print("Sensitivities to swap rate quotes, 1bp shift:")
    for k, v in swaps_in.items():
        print(f"  {ql.Period(k[0], k[1])}: {v.derivative * 0.0001:.2f}")
```

### Analysing Results

Using the `show` function, we analyse the sensitivities of swaps to different market quotes. This will highlight the impact of various inputs on the swap's NPV, helping in understanding risk exposures and in the construction of hedges.

Our analysis of sensitivities reveals:

- **5-Year Spot Swap on Deposit/Futures/Swap Curve**: Only the 5-year swap rate quote affects this swap's price, with other quotes having no impact. A 1 basis point shift in the 5Y swap rate results in a price change of 443.4. This is expected due to the direct use of this rate in curve construction.

```text
NPV         = 19066.26
Fair spread = -0.4174 %
Fair rate   = 4.4300 %

Sensitivities to deposit quotes, 1bp shift:
  3M: -0.00
Sensitivities to FRA quotes, 1bp shift:
  3M - 6M: 0.00
  6M - 9M: 0.00
  9M - 12M: 0.00
Sensitivities to futures quotes, 1c shift:
  December 19th, 2001: 0.00
  March 20th, 2002: 0.00
  June 19th, 2002: 0.00
  September 18th, 2002: 0.00
  December 18th, 2002: 0.00
  March 19th, 2003: 0.00
  June 18th, 2003: -0.00
  September 17th, 2003: 0.00
Sensitivities to swap rate quotes, 1bp shift:
  2Y: 0.00
  3Y: -0.00
  5Y: 443.40
  10Y: 0.00
  15Y: 0.00
```
  
- **5-Year Spot Swap on Deposit/FRA/Swap Curve**: This is pricing the same 5-year spot swap on a different curve, which  includes the same 5Y swap rate quote. Sensitivities for all quotes except the 5Y swap rate are zero, as anticipated, and the price is identical.

```text
NPV         = 19066.26
Fair spread = -0.4174 %
Fair rate   = 4.4300 %

Sensitivities to deposit quotes, 1bp shift:
  3M: -0.00
Sensitivities to FRA quotes, 1bp shift:
  3M - 6M: -0.00
  6M - 9M: -0.00
  9M - 12M: -0.00
Sensitivities to futures quotes, 1c shift:
  December 19th, 2001: 0.00
  March 20th, 2002: 0.00
  June 19th, 2002: 0.00
  September 18th, 2002: 0.00
  December 18th, 2002: 0.00
  March 19th, 2003: 0.00
  June 18th, 2003: 0.00
  September 17th, 2003: 0.00
Sensitivities to swap rate quotes, 1bp shift:
  2Y: -0.00
  3Y: 0.00
  5Y: 443.40
  10Y: 0.00
  15Y: 0.00
```

- **1-Year Forward-Starting Swap on Deposit/Futures/Swap Curve**: Recall that the that pricing date is 6 November 2001 and this swap starts in November 2002, maturing in November 2007. The sensitivities show dependencies on 3M deposit and future rates up to one year from the pricing date, which determine the rate at the time the swap starts. We see a further dependency on both the 5-year and 10-year swap rates, as the swap's maturity falls between these 2 rates. We see no dependency on FRA quotes, as they are not part of the curve.

```text
NPV         = 40533.04
Fair spread = -0.9241 %
Fair rate   = 4.9520 %

Sensitivities to deposit quotes, 1bp shift:
  3M: -11.42
Sensitivities to FRA quotes, 1bp shift:
  3M - 6M: 0.00
  6M - 9M: 0.00
  9M - 12M: 0.00
Sensitivities to futures quotes, 1c shift:
  December 19th, 2001: 24.49
  March 20th, 2002: 24.94
  June 19th, 2002: 24.53
  September 18th, 2002: 13.48
  December 18th, 2002: 0.00
  March 19th, 2003: -0.00
  June 18th, 2003: 0.00
  September 17th, 2003: 0.00
Sensitivities to swap rate quotes, 1bp shift:
  2Y: 0.00
  3Y: -0.00
  5Y: 347.43
  10Y: 174.31
  15Y: 0.00
```
  
- **1-Year Forward-Starting Swap on Deposit/FRA/Swap Curve**: Displays dependencies similar to the previous curve but relies on FRA quotes instead of futures, with sensitivities matching the previous curve for the 5-year and 10-year swap rates.

```text
NPV         = 37144.28
Fair spread = -0.8469 %
Fair rate   = 4.8724 %

Sensitivities to deposit quotes, 1bp shift:
  3M: -25.30
Sensitivities to FRA quotes, 1bp shift:
  3M - 6M: -24.23
  6M - 9M: -27.69
  9M - 12M: -21.64
Sensitivities to futures quotes, 1c shift:
  December 19th, 2001: 0.00
  March 20th, 2002: 0.00
  June 19th, 2002: 0.00
  September 18th, 2002: 0.00
  December 18th, 2002: 0.00
  March 19th, 2003: 0.00
  June 18th, 2003: 0.00
  September 17th, 2003: 0.00
Sensitivities to swap rate quotes, 1bp shift:
  2Y: -0.00
  3Y: -0.00
  5Y: 347.43
  10Y: 174.31
  15Y: 0.00
```


## Performance Evaluation

To gauge the performance impact of calculating sensitivities, we leverage the [multi-curve bootstrapping example](https://github.com/auto-differentiation/QuantLib-Risks-Py/blob/main/Python/examples/multicurve-bootstrapping.py). This setup, while similar to the swap pricer above, incorporates a wider array of quotes to construct the term structure for swap pricing. 
It involves pricing a forward-starting 5-year swap, set to commence 15 months into the future, with a calculation of 69 sensitivities covering all market quotes used in curve construction and select swap parameters like nominal, fixed rate, and spread.

Performance metrics are drawn from averaging execution times over 20 runs for stability. Initial benchmarks using the standard QuantLib package take 198ms for pricing alone. Switching to `QuantLib-Risks` for simultaneous pricing and sensitivity analysis results in an execution time of 370ms, demonstrating that all sensitivities can be obtained within approximately 1.87x of the original pricing time.

Comparatively, a traditional bump-and-reval approach for sensitivities would necessitate 70 pricer code executions (one for valuation and one for each variable bump), translating to 70x the pure pricing time. Thus, `QuantLib-Risks` achieves a 37.4x speed advantage over bump and reval.

Summary of performance benchmarks:

- Number of sensitivities: 69
- Valuation time (QuantLib): 198ms
- Sensitivity calculation (QuantLib-Risks): 370ms
- Bump-and-reval time (estimate): 13,860ms
- **QuantLib-Risks sensitivities overhead vs QuantLib valuation: 1.87x**
- **QuantLib-Risks speedup vs bump-and-reval: 37.4x**

Benchmark configuration:

- `QuantLib` version: 1.33
- `QuantLib-Risks` version: 1.33.3
- `xad` version: 1.5.2
- Operating on Ubuntu 22.04
- Hardware specs include 128GB RAM and an Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz.

## Getting Hands-On

To try these examples yourself:

1. Access the code samples on [GitHub](https://github.com/auto-differentiation/QuantLib-Risks-Py/blob/main/Python/examples/).
2. Experiment with the examples in an online notebook via [Binder](https://mybinder.org/v2/gh/auto-differentiation/QuantLib-Risks-Py/main?labpath=Python%2Fexamples)
3. Experiment locally after installing `QuantLib-Risks` with pip:
    ```text
    pip install QuantLib-Risks
    ```

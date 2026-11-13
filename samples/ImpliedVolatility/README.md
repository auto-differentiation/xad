# ImpliedVolatility

Solves the Black-Scholes formula for the implied volatility of a European call with
Newton's method in plain `double`. It then computes how the implied volatility changes
with the option price, spot and rate using the implicit function theorem: one adjoint
sweep of XAD at the converged volatility, without differentiating through the solver.

The market data is fixed: spot `100`, strike `105`, rate `5%` and maturity `1` year. The
option price is generated in the sample from a volatility of `20%`, which Newton's method
recovers. No data file is needed.

## How it works

**Solving.** Newton's method finds the root of

    F(vol) = C(vol; spot, rate) - price

where `C` is the Black-Scholes call price. The derivative `dF/dvol` is the analytic vega.
It starts from `vol = 50%` and stops when the step falls below `1e-14`. A step never more
than halves the volatility, so it stays positive. The pricer is written once as a
template, so the same code runs with `double` and with XAD's active type.

**Differentiating.** Running Newton on active types and differentiating through the
iterations would give the derivative of the last iterate rather than of the root. That
derivative depends on the starting guess and on when the loop stops. For solvers such as
bisection it is zero, since the iterates are piecewise constant in the inputs. It also
records every iteration on the tape.

The sample uses the implicit function theorem instead. At the root,
`F(vol*, p) = 0` holds for all nearby inputs `p`, so differentiating that identity gives

    dvol*/dp = -(dF/dp) / (dF/dvol)

Only the partial derivatives of `F` at the single point `(vol*, p)` are needed. One adjoint
sweep of `F` gives them all, for the price, spot and rate, and `dF/dvol` (the vega). The
tape holds a single evaluation of the pricer, however many iterations Newton needed.

**An equivalent alternative.** Instead of applying the formula above by hand, you can
run one Newton step on active types, starting from the converged root:

    vol1 = vol* - F(vol*, p) / (dF/dvol)(vol*, p)

At the root `F = 0`, so `vol1 = vol*`, and differentiating the step gives exactly
`-(dF/dp) / (dF/dvol)`. The terms involving the derivative of `dF/dvol` are all
multiplied by `F = 0`. This needs `dF/dvol` itself as an active expression (here the
analytic vega). The sample uses the single evaluation of `F` because it shows the
theorem directly.

For a solver whose derivatives come from XAD at every iteration, see
[`samples/CurveFit`](../CurveFit), which fits a curve with Gauss-Newton using XAD
Jacobians.

## Checks

Since `dF/dprice = -1`, `dF/dspot = delta` and `dF/drate = rho`, the sensitivities have a
closed form:

    dvol/dprice = 1 / vega
    dvol/dspot  = -delta / vega
    dvol/drate  = -rho / vega

The sample compares the XAD results with these to a relative tolerance of `1e-12`. It
also compares them with central finite differences of the full Newton solve, re-solved
for each input moved up and down by `h = 1e-5 * max(1, |input|)`, to a relative tolerance
of `1e-6`. It returns a non-zero exit code if Newton's method does not converge or any
check fails.

## Build and run

From the repository root:

    cmake -S . -B build
    cmake --build build --target ImpliedVolatility
    build/samples/ImpliedVolatility/ImpliedVolatility

Or, from the build directory, `ctest -R ImpliedVolatility`.

## Expected output

The last digits of the price errors and relative differences vary between compilers and
platforms. A representative run prints:

    European call: spot 100.00, strike 105.00, rate 5.00%, maturity 1.00 years
    market price 8.0213522351 (generated from vol = 20.00%)

    Newton's method
    iter                vol    price error
       0  0.500000000000000   1.176849e+01
       1  0.250000000000000   1.980850e+00
       2  0.199927160282269  -2.889595e-03
       3  0.199999999867878  -5.241361e-09
       4  0.200000000000000   7.105427e-15

    implied vol = 0.200000000000000

    at the root: F = 0.000e+00, dF/dvol = 39.6705238084 (vega)

    sensitivities of the implied vol
        dvol/d   implicit (XAD)      closed form     finite diff.   rel. diff   rel. diff
                                                                       closed          FD
         price     0.0252076329     0.0252076329     0.0252076329   2.753e-16   3.016e-11
          spot    -0.0136682928    -0.0136682928    -0.0136682928   1.269e-16   1.139e-09
          rate    -1.1646299743    -1.1646299743    -1.1646299758   1.907e-16   1.272e-09

# Black-Scholes Greeks Sample

Prices a European call and put with the Black-Scholes formula and computes the
Greeks with XAD in both forward and adjoint mode, checking them against the
closed-form Greeks.

The pricer is written once, as a template on the number type:

```c++
template <class T>
T blackScholes(bool isCall, const T& spot, double strike, const T& vol, const T& rate,
               const T& maturity);
```

and is called with `double` for the price, `xad::FReal<double>` for forward
mode and `xad::AReal<double>` for adjoint mode. The strike is a plain `double`,
since no Greek is needed for it.

## Greeks

| Greek | Definition                                              |
| ----- | ------------------------------------------------------- |
| delta | dV/dS, sensitivity to the spot                          |
| vega  | dV/dsigma, sensitivity to the volatility                |
| rho   | dV/dr, sensitivity to the interest rate                 |
| theta | dV/dt = -dV/dT, change in value as calendar time passes |

## Forward and Adjoint Mode

- **Forward mode** (`xad::FReal<double>`): the tangent of one input is set to 1
  and a single evaluation gives the derivative of the price with respect to
  that input. Four Greeks need four passes.
- **Adjoint mode** (`xad::AReal<double>` with an `xad::Tape<double>`): the inputs
  are registered on the tape, the price is recorded once, and one reverse pass
  from the price gives all four Greeks at once.

With many inputs and one output, as is typical for pricing, adjoint mode gives
all sensitivities for a cost that does not grow with the number of inputs.

## Check

Both results are compared with the closed-form Greeks. The difference must stay
below $10^{-12}$ (absolute, or relative for values larger than 1), otherwise the
sample exits with a non-zero code and `ctest` reports a failure.

## Build and Run

From the repository root:

    cmake -S . -B build
    cmake --build build --target BlackScholesGreeks
    build/samples/BlackScholesGreeks/BlackScholesGreeks

Or run via `ctest`:

    ctest --test-dir build -R sample_BlackScholesGreeks --output-on-failure

## Expected Output

    Black-Scholes, spot = 100, strike = 105, vol = 0.2, rate = 0.03, maturity = 1.5

                           price       delta        vega         rho       theta
    call exact          9.577478    0.542606   48.581343   67.024681   -4.579250
    call forward        9.577478    0.542606   48.581343   67.024681   -4.579250
    call adjoint        9.577478    0.542606   48.581343   67.024681   -4.579250
    put exact           9.957214   -0.457394   48.581343  -83.544922   -1.567858
    put forward         9.957214   -0.457394   48.581343  -83.544922   -1.567858
    put adjoint         9.957214   -0.457394   48.581343  -83.544922   -1.567858

    largest difference to the closed form: 4.24e-16

# HestonCalibration

Calibrates the five Heston parameters to a grid of 35 European call prices
(5 maturities x 7 strikes) with Levenberg-Marquardt. The semi-analytic pricer works on
`std::complex<xad::AReal<double>>`, and the Jacobian of the residuals comes from XAD's
adjoint mode. Finite differences are used only to check it once, at the starting point.

The Heston model has stochastic variance `v` with parameters `kappa` (mean reversion),
`theta` (long-run variance), `sigma` (volatility of variance), `rho` (correlation with
the spot) and `v0` (initial variance). The sample uses a spot of 100 and zero interest
rate and dividend yield.

The "observed" prices are generated in the sample from known parameters
(`kappa = 2`, `theta = 0.04`, `sigma = 0.35`, `rho = -0.7`, `v0 = 0.05`) at maturities of
0.25, 0.5, 1, 2 and 3 years and strikes of 80, 90, 95, 100, 105, 110 and 120. They are
noise-free, so the fit must recover the parameters. No data file is needed.

## How it works

**Pricer:** the call price follows from the characteristic function `phi` of `ln(S_T / S_0)`
through Lewis' formula

    C = S - sqrt(S K) / pi * int_0^inf Re[exp(i u k) phi(u - i/2)] / (u^2 + 1/4) du,   k = ln(S / K)

and the integral is evaluated with 200-point Gauss-Legendre quadrature on `[0, 100]`. The
characteristic function is written in the "little Heston trap" form of Albrecher et al.
(with `exp(-d T)` instead of `exp(+d T)`), so the complex logarithm stays continuous
for long maturities. The pricer is a template over the real type: the same code runs
with `double` and with `xad::AReal<double>`, where the complex arithmetic (`sqrt`,
`exp`, `log`, division) is differentiated by XAD.

**Jacobian:** the residuals `r_i = model_i - observed_i` are recorded once on the tape.
With 5 parameters and 35 residuals, forward mode would need 5 passes, but the adjoint
variant shows the pattern that matters for larger models: one backward sweep per
residual gives one row of the 35 x 5 Jacobian, without re-evaluating the pricer.

**Levenberg-Marquardt:** each iteration solves

    (J^T J + lambda diag(J^T J)) dp = -J^T r

`lambda` is increased tenfold until the step keeps the parameters valid (`kappa`, `theta`,
`sigma`, `v0` positive, `|rho| < 1`) and lowers the sum of squared residuals, and is
reduced again after every accepted step.

The sample returns a non-zero exit code if the XAD and finite-difference Jacobians
differ by more than `1e-5` (relative), if Levenberg-Marquardt does not converge, or if
a fitted parameter differs from its true value by more than `1e-6`.

## Build and run

From the repository root:

    cmake -S . -B build
    cmake --build build --target HestonCalibration
    build/samples/HestonCalibration/HestonCalibration

Or, from the build directory, `ctest -R HestonCalibration`.

## Expected output

The last digits of the Jacobian check vary between compilers and platforms. A
representative run prints:

    Jacobian check at the starting point (max over the 35 residuals):
       param       max |XAD|        max |FD|    rel. diff
       kappa      1.01487313      1.01487313    1.817e-08
       theta     96.21993230     96.21993232    1.969e-10
       sigma      5.01064309      5.01064309    4.117e-09
         rho      4.00976329      4.00976329    5.844e-09
          v0     47.04222318     47.04222318    3.578e-10

    iter     kappa     theta     sigma       rho        v0        sum r^2
       0   1.00000   0.06000   0.50000  -0.30000   0.08000   1.266926e+02
       1   1.52332   0.03997   0.44143  -0.48808   0.04892   1.109131e+00
       2   1.95431   0.03962   0.37285  -0.62360   0.05015   5.908991e-02
       3   2.00309   0.03999   0.35050  -0.69488   0.05001   4.991142e-04
       4   2.00000   0.04000   0.34999  -0.70001   0.05000   4.959944e-10
       5   2.00000   0.04000   0.35000  -0.70000   0.05000   3.379754e-19
       6   2.00000   0.04000   0.35000  -0.70000   0.05000   9.289626e-27

       param        true      fitted
       kappa    2.000000    2.000000
       theta    0.040000    0.040000
       sigma    0.350000    0.350000
         rho   -0.700000   -0.700000
          v0    0.050000    0.050000

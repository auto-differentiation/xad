# CurveFit

Fits a Nelson-Siegel zero curve to 20 zero rates with Gauss-Newton. The Jacobian of
the residuals comes from XAD's vector forward mode; finite differences are used only
to check it once, at the starting point.

The Nelson-Siegel zero rate for maturity `t` (in years) has 4 parameters:

    z(t) = beta0 + beta1 * L(t) + beta2 * (L(t) - exp(-t / tau))
    L(t) = (1 - exp(-t / tau)) / (t / tau)

`beta0` is the long-term level, `beta1` the slope, `beta2` the curvature (hump) and
`tau` the decay time in years.

The "observed" rates are generated in the sample from known parameters
(`beta0 = 4.5%`, `beta1 = -2%`, `beta2 = 1.5%`, `tau = 2`) at maturities from 3 months
to 30 years, plus uniform noise of +/- 2 basis points. The noise comes from
`std::mt19937` with seed `42`, which gives the same numbers on every platform. No data
file is needed.

## How it works

The residuals are `r_i = z(t_i) - observed_i`. Each Gauss-Newton iteration solves

    (J^T J) dp = -J^T r

where `J` is the 20 x 4 Jacobian `dr_i / dp_k`. The model is written once as a template,
so the same code runs with `double` (for the residuals and the finite differences) and
with XAD's active type.

**Why forward mode:** forward mode costs one pass per input direction, adjoint mode one
backward sweep per output. Here there are 4 parameters and 20 residuals, so forward mode
is the better fit. `xad::fwd<double, 4>` carries all 4 directions at once: the parameters
are seeded with the unit vectors, and a single evaluation of the model gives the whole
Jacobian, without a tape.

**Starting point:** `tau = 1` year, and the betas that fit best for that `tau`. The model
is linear in the betas, so one Gauss-Newton step on them alone solves this exactly.
The Nelson-Siegel fit has local minima in `tau`: with a poor start for the betas,
Gauss-Newton can end up at a much smaller `tau` with residuals well above the noise.
Steps are halved until the residual norm goes down and `tau` stays positive.

The sample returns a non-zero exit code if the XAD and finite-difference Jacobians
differ by more than `1e-6` (relative), if Gauss-Newton does not converge, or if the RMS
residual of the fit is more than 1.5 times the noise RMS.

## Build and run

From the repository root:

    cmake -S . -B build
    cmake --build build --target CurveFit
    build/samples/CurveFit/CurveFit

Or, from the build directory, `ctest -R CurveFit`.

## Expected output

The last digits of the Jacobian check vary between compilers and platforms. A
representative run prints:

    Jacobian check at the starting point (max over the 20 residuals):
       param       max |XAD|        max |FD|    rel. diff
       beta0      1.00000000      1.00000000    1.000e-12
       beta1      0.88479687      0.88479687    2.595e-12
       beta2      0.29699708      0.29699708    8.157e-12
         tau      0.00639354      0.00639354    7.795e-10

    iter       beta0       beta1       beta2         tau            |r|
       0    0.045838   -0.021539    0.000136    1.000000   1.083756e-03
       1    0.045837   -0.021534    0.002698    1.119258   1.063643e-03
       2    0.045781   -0.021357    0.006624    1.332046   1.016827e-03
       3    0.045656   -0.021059    0.009128    1.502657   8.939597e-04
       4    0.045212   -0.020179    0.014345    1.914029   8.755798e-04
       5    0.045141   -0.020160    0.013589    1.890158   4.240398e-04
       6    0.045141   -0.020162    0.013572    1.888508   4.240164e-04
    ...

       param        true      fitted
       beta0    0.045000    0.045141
       beta1   -0.020000   -0.020162
       beta2    0.015000    0.013572
         tau    2.000000    1.888457

    RMS residual = 0.95 bp (noise RMS = 1.15 bp)

The fitted parameters differ from the true ones because of the noise; the RMS residual
of the fit is at the noise level.

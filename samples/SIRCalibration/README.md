# SIRCalibration

Fits the parameters of an SIR epidemic model to daily case counts. The model is
written as plain C++ loops, integrated with fixed-step RK4 over 150 days. Its
gradient comes from XAD's adjoint mode, not from finite differences; finite
differences are used only as a check.

The "observed" cases are generated in the sample from known parameters
(`beta = 0.35`, `gamma = 0.1`, `I0 = 20`) plus Gaussian noise with standard
deviation 200, using `std::mt19937` with seed `42`. No data file is needed.

The model has a lockdown on day 60, when `beta` is halved. The branch
`if (day >= LOCKDOWN_DAY)` is taken on a plain integer, so it does not affect the
tape.

## How it works

The state is the susceptible and infected populations. Each day is split into
4 RK4 steps:

    dS/dt = -beta * S * I / N
    dI/dt =  beta * S * I / N - gamma * I

The loss is the mean squared residual between the simulated and observed daily
cases, in units of the noise standard deviation. The fit uses `log(beta)`,
`log(gamma)` and `log(I0)` as parameters, so they stay positive.

Each gradient comes from one adjoint sweep over the whole simulation. The sample
checks it against central differences at the starting guess and returns a
non-zero exit code if the relative difference exceeds `1e-5`. The fit itself is
gradient descent with a backtracking line search.

## Build and run

From the repository root:

    cmake -S . -B build -DXAD_ENABLE_TESTS=ON
    cmake --build build --target SIRCalibration
    build/samples/SIRCalibration/SIRCalibration

Or, from the build directory, `ctest -R SIRCalibration`.

## Expected output

The structure is fixed; exact digits may vary between C++ standard-library
implementations. A representative run prints:

    initial guess: loss = 5555.036553
     param    adjoint grad         FD grad       rel. diff
         0       -5.821927       -5.821927       8.605e-08
         1        5.207147        5.207147       7.588e-08
         2       -0.617106       -0.617105       8.118e-07

    gradient descent on log-parameters
    ...

         param        true      fitted
          beta    0.350000    0.359237
         gamma    0.100000    0.106331
            I0   20.000000   17.347381

    final loss = 0.998423

The fitted loss is close to 1, the value expected when the model matches the true
parameters and the noise standard deviation is correct.

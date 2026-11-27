# KalmanFilter

Estimates the parameters of an AR(1)-plus-noise linear state-space model by
maximising the Kalman filter log-likelihood. The gradient with respect to all
model parameters is computed in a single adjoint pass with XAD, verified against
central finite differences, and used in a BFGS optimizer to recover the true
simulation parameters.

## Model

The simulated series has 500 time steps generated with `std::mt19937` (seed `42`)
from:

    x_t = phi * x_{t-1} + eta_t,    eta_t ~ N(0, sigmaEta^2)
    y_t = x_t + eps_t,              eps_t ~ N(0, sigmaEps^2)

with true parameters `phi = 0.85`, `sigmaEta^2 = 1.00`, and `sigmaEps^2 = 0.25`.
Both a contiguous 5-step gap and scattered observations (~4%) are marked as
missing.

To keep optimization unconstrained while enforcing stationarity and strictly
positive variances, the optimizer works on:

    phi        = tanh(u[0])   in (-1, 1)
    sigmaEta^2 = exp(u[1])    > 0
    sigmaEps^2 = exp(u[2])    > 0

## How it works

The Kalman filter negative log-likelihood is written once as a template for both
`double` and XAD's `xad::adj<double>::active_type`:

1. **Loop-invariant hoisting**: `phi`, `phi^2`, `sigmaEta^2`, and `sigmaEps^2`
   are computed once before the recursion; the `0.5` factor and `log(2*pi)`
   constant are folded after the loop to minimise tape nodes.
2. **Positive-definite covariance update**: at each observed step, the inverse
   innovation variance `invF = 1 / (pPred + sigmaEps^2)` is computed once and
   the posterior variance is updated as `(pPred * sigmaEps^2) * invF`, avoiding
   cancellation in `(1 - K) * pPred` and reducing divisions from three to one
   per step.
3. **Missing observations**: branching on a plain `unsigned char` mask skips the
   measurement update (`xHat = xPred`, `p = pPred`) and adds no likelihood term.
   Because the branch condition does not depend on active variables, XAD records
   only the executed operations and computes exact gradients through the
   branches.
4. **Low-latency optimization**: a single tape is reused across iterations via
   `tape.clearAll()`, and a stack-allocated 3x3 BFGS update with backtracking
   Armijo line search converges without heap allocations in the hot loop.

## Build and run

From the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target KalmanFilter
    build/samples/KalmanFilter/KalmanFilter

Or, from the build directory, `ctest -R KalmanFilter`. The sample returns a
non-zero exit code if the finite-difference gradient check fails or BFGS does
not converge.

## Expected output

The structure is fixed; exact digits may vary between C++ standard-library
implementations. A representative run prints:

    Kalman filter MLE (AR(1) + noise)
    Observations: 500 steps (476 observed, 24 missing)

    initial guess: mean NLL = 1.813048
             param    adjoint grad         FD grad       rel. diff
        atanh(phi)       -0.581052       -0.581052       3.836e-10
     log(sigEta^2)       -0.098517       -0.098517       1.169e-09
     log(sigEps^2)        0.011730        0.011730       2.182e-10

    BFGS optimization (maximizing log-likelihood)
    iter   0  mean NLL = 1.813048  |grad| = 5.895e-01
    iter   5  mean NLL = 1.661907  |grad| = 9.872e-03
    iter  10  mean NLL = 1.656978  |grad| = 1.729e-03
    iter  15  mean NLL = 1.656946  |grad| = 7.613e-06
    iter  18  mean NLL = 1.656946  |grad| = 9.515e-09

    converged in 18 iterations
             param        true      fitted
               phi    0.850000    0.858635
        sigmaEta^2    1.000000    0.778627
        sigmaEps^2    0.250000    0.543440

    final log-likelihood = -788.706123

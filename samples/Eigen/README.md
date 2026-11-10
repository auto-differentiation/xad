# Eigen

Uses XAD's adjoint mode with Eigen matrices to compute two gradients through Eigen's
decompositions. Each gradient comes from a single adjoint sweep and is checked against
central finite differences:

- a least-squares residual, solved with `ColPivHouseholderQR`, differentiated with respect
  to the matrix `A` and the right-hand side `b`
- `log|det(M)|`, computed with `PartialPivLU`, differentiated with respect to `M` and also
  compared with its analytical gradient `inverse(M)^T`

All inputs are fixed matrices written in the source, so every run gives the same output.

## Using XAD types in Eigen

Eigen matrices hold XAD's active type like any other scalar:

    typedef xad::adj<double>::active_type AD;
    Eigen::Matrix<AD, Eigen::Dynamic, Eigen::Dynamic> A = A0.cast<AD>();

Include `XAD/StdCompatibility.hpp` before `XAD/XAD.hpp` and the Eigen headers. It makes
XAD's math functions available in namespace `std`, where Eigen looks for them. Each
function is written once as a template, so the `double` version gives the finite
differences.

## Least squares

For an overdetermined `5x3` matrix `A` and a vector `b` outside its column space:

    x = argmin ||A x - b||           (A.colPivHouseholderQr().solve(b))
    L(A, b) = ||A x - b||^2

The tape records the whole QR factorisation and solve, so the adjoint sweep accounts for
how `x` changes with `A` and `b`. Since `dL/dx = 0` at the optimum, the result reduces to
`dL/dA = 2 r x^T` and `dL/db = -2 r`, where `r = A x - b`. The printed gradients
can be checked against this. The residual is non-zero on purpose: with a square system
it would be zero, and so would the gradient.

## Log-determinant

For a `4x4` matrix `M`, `log|det(M)|` is the sum of `log|U_ii|` over the diagonal of the LU
factor. This avoids the overflow and underflow of forming `det(M)` itself. Its gradient is

    d log|det(M)| / dM = inverse(M)^T

which the sample compares with the adjoint gradient as well as finite differences.

## Checks

Finite differences use step `h = 1e-6` on each entry:

    finite_difference = (f(x + h) - f(x - h)) / (2 * h)

For every entry, the sample computes `|adjoint - reference| / (1 + |reference|)` and prints
the largest value. The tolerance is `1e-6` against finite differences and `1e-10` against
`inverse(M)^T`. The sample returns a non-zero exit code if a gradient is not finite or any
check exceeds its tolerance.

## Build and run

The sample is built when Eigen 3.4 or later is available: either already installed and
found by CMake, or downloaded with `XAD_ENABLE_EIGEN_TESTS`. From the repository root:

    cmake -S . -B build -DXAD_ENABLE_EIGEN_TESTS=ON
    cmake --build build --target EigenSample
    build/samples/Eigen/EigenSample

Or, from the build directory, `ctest -R EigenSample`. Without Eigen, CMake prints
`Skipping Eigen sample (Eigen 3.4 not found)`. The target is called `EigenSample`
because the name `Eigen` is used by the downloaded Eigen library.

## Expected output

    least squares, A is 5x3, solved with ColPivHouseholderQR
      x = 0.258303 0.648878 1.106410
      ||A x - b||^2 = 27.983091
      dL/dA =
        [ 0.638160   1.603111   2.733484]
        [ 0.810890   2.037023   3.473354]
        [-0.608435  -1.528439  -2.606160]
        [-2.070071  -5.200191  -8.866912]
        [-1.322114  -3.321262  -5.663125]
      dL/db =
        [-2.470588  -3.139300   2.355510   8.014126   5.118468]

    least-squares checks
      dL/dA vs finite differences         max error 1.98e-09  (tolerance 1.00e-06)  OK
      dL/db vs finite differences         max error 1.30e-09  (tolerance 1.00e-06)  OK

    log-determinant, M is 4x4, factorised with PartialPivLU
      log|det(M)| = 5.581153
      d log|det(M)| / dM =
        [ 0.357227  -0.045407   0.030504  -0.199376]
        [-0.098162   0.191802  -0.079057   0.120187]
        [-0.009703   0.038153   0.161091  -0.039943]
        [-0.218557  -0.014413  -0.060857   0.459534]

    log-determinant checks
      gradient vs inverse(M)^T            max error 2.39e-17  (tolerance 1.00e-10)  OK
      gradient vs finite differences      max error 4.90e-10  (tolerance 1.00e-06)  OK

The exact error values in the checks may differ slightly between compilers.

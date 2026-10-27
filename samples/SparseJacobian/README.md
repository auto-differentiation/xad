# SparseJacobian

Computes the Jacobian of a function with a tridiagonal sparsity pattern using only
3 forward-mode passes, instead of one pass per input, and checks it against the dense
Jacobian from adjoint mode.

The function (written once as a template for any active type) is, with `x_{-1} = x_n = 0`:

    y_i = x_{i-1} - 2 x_i + x_{i+1} + x_{i-1} x_{i+1} + 0.1 exp(x_i)

Each `y_i` depends only on `x_{i-1}`, `x_i` and `x_{i+1}`. The sample uses `xad::fwd<double>`
with `derivative()` for the sparse passes, and `computeJacobian` (`XAD/Jacobian.hpp`,
adjoint mode) for the dense reference.

## Why 3 passes are enough

Column `j` gets colour `j % 3`. Columns of the same colour never affect the same row, so they
can be seeded together in one pass and their contributions stay separate. For n = 6:

    pass 0: columns 0, 3   (column 0 affects rows 0-1, column 3 affects rows 2-4)
    pass 1: columns 1, 4   (rows 0-2 and rows 3-5)
    pass 2: columns 2, 5   (rows 1-3 and rows 4-5)

In each pass, at most one of columns i-1, i, i+1 has the pass’s colour, so derivative(y[i]) is exactly that one entry of row i (lower, diagonal or upper band) and can be stored directly.

## Build and run

    cmake -S . -B build
    cmake --build build --target SparseJacobian
    build/samples/SparseJacobian/SparseJacobian

or, from the build directory, `ctest -R SparseJacobian`. The sample returns a non-zero
exit code if sparse and dense results differ by more than 1e-12.

## Expected output

The error lines are deterministic; timings and the checksum vary by machine:

    n = 10: max |sparse - dense| = 0 (OK)
    n = 1000: max |sparse - dense| = 0 (OK)
    median of 11 runs:
    sparse (3 forward passes): 0.1421 ms
    dense  (1000 adjoint passes): 6.6153 ms
    checksum (ignore): -42.4038

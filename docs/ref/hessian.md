# Hessian

## Overview

XAD implements a set of methods to compute the Hessian matrix of a function and its inputs in `XAD/Hessian.hpp`.

Note that the Hessian header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.


## Specialisations
Hessians can be computed in `fwd_adj` or `fwd_fwd` higher-order mode.

The `computeHessian()` method takes a set of variables packaged in a `std::vector<T>` and a function in the format `T foo(std::vector<T>)`.

If provided with `RowIterator`s, `computeHessian()` will write directly to them and return `void`. If no `RowIterator`s are provided, the Hessian will be written to a `std::vector<std::vector<T>>` and returned.

#### `fwd_adj`
This mode uses a [Tape](ref/tape.md) to compute second derivatives. This Tape will be instantiated within the method or set to the current active Tape using `Tape::getActive()` if none is passed as argument.

#### `fwd_fwd`
This mode does not require a Tape and can help reduce the overhead that comes with one.

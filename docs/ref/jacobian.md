# Jacobian

## Overview

XAD implements a set of methods to compute the Jacobian matrix of a function and its inputs in `XAD/Jacobian.hpp`.

Note that the Jacobian header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.


## Specialisations

Jacobians can be computed in `adj` or `fwd` mode.

The `computeJacobian()` method takes a set of variables packaged in a `std::vector<T>` and a function in the format `std::vector<T> foo(std::vector<T>)`.
If provided with `RowIterator`s, `computeHessian()` will write directly to them and return `void`. If no `RowIterator`s are provided, the Hessian will be written to a `std::vector<std::vector<T>>` and returned.

#### `adj`

This mode uses a [Tape](ref/tape.md) to compute derivatives. This Tape will be instantiated within the method or set to the current active Tape using `Tape::getActive()` if none is passed as argument.

#### `fwd_fwd`

This mode does not require a Tape and can help reduce the overhead that comes with one.

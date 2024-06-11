# Jacobian

## Overview

XAD implements a set of methods to compute the Jacobian matrix of a function in `XAD/Jacobian.hpp`.

Note that the Jacobian header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.

Jacobians can be computed in `adj` or `fwd` mode.

The `computeJacobian()` method takes a set of variables packaged in a
`std::vector<T>` and a function with signature
`std::vector<T> foo(std::vector<T>)`.

## Return Types

If provided with `RowIterators`, `computeHessian()` will write directly to
them and return `void`. If no `RowIterators` are provided, the Hessian will be
written to a `std::vector<std::vector<T>>` and returned.

## Specialisations

#### `adj`

```c++
template <class RowIterator, typename T>
void computeJacobian(const std::vector<AD> &vec,
                     std::function<std::vector<AD>(std::vector<AD> &)> foo,
                     RowIterator first, RowIterator last,
                     xad::Tape<T> *tape = xad::Tape<T>::getActive())
```

This mode uses a [Tape](ref/tape.md) to compute derivatives. This Tape will
be instantiated within the method or set to the current active Tape using
`Tape::getActive()` if none is passed as argument.

#### `fwd_fwd`

```c++
template <class RowIterator, typename T>
void computeJacobian(const std::vector<xad::FReal<T>> &vec,
                     std::function<std::vector<xad::FReal<T>>(std::vector<xad::FReal<T>> &)> foo,
                     RowIterator first, RowIterator last)
```

This mode does not require a Tape and can help reduce the overhead that
comes with one.

## Example Use

Given $f(x, y, z, w) = [sin(x + y) sin(y + z) cos(z + w) cos(w + x)]$, or

```c++
auto foo = [](std::vector<AD> &x) -> std::vector<AD>
{
    return {sin(x[0] + x[1]), sin(x[1] + x[2]), cos(x[2] + x[3]), cos(x[3] + x[0])};
};
```

with the derivatives of interest being

```c++
std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});
```

we'd like to compute the Jacobian

$$
J = \begin{bmatrix}
\frac{\partial \sin(x + y)}{\partial x} &
\frac{\partial \sin(x + y)}{\partial y} &
\frac{\partial \sin(x + y)}{\partial z} &
\frac{\partial \sin(x + y)}{\partial w} \\
\frac{\partial \sin(y + z)}{\partial x} &
\frac{\partial \sin(y + z)}{\partial y} &
\frac{\partial \sin(y + z)}{\partial z} &
\frac{\partial \sin(y + z)}{\partial w} \\
\frac{\partial \cos(z + w)}{\partial x} &
\frac{\partial \cos(z + w)}{\partial y} &
\frac{\partial \cos(z + w)}{\partial z} &
\frac{\partial \cos(z + w)}{\partial w} \\
\frac{\partial \cos(w + x)}{\partial x} &
\frac{\partial \cos(w + x)}{\partial y} &
\frac{\partial \cos(w + x)}{\partial z} &
\frac{\partial \cos(w + x)}{\partial w}
\end{bmatrix}
$$

First step is to setup the tape and active data types

```c++
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;
```

Note that if no tape is setup, one will be created when computing the Jacobian.
`fwd` mode is also supported in the same fashion. All that is left to do is
define our input values and our function, then call `computeJacobian()`:

```c++
    auto foo = [](std::vector<AD> &x) -> std::vector<AD>
    { return {sin(x[0] + x[1]),
              sin(x[1] + x[2]),
              cos(x[2] + x[3]),
              cos(x[3] + x[0])}; };

    auto jacobian = xad::computeJacobian<double>(x_ad, foo);
```

Note the signature of `foo()`. Any other signature will throw an error.

This computes the relevant matrix

$$
\begin{bmatrix}
1 & 0.0707372 & 0 & 0 \\
0 & 1 & 0.267499 & 0 \\
0 & 0 & 1 & 0.362358 \\
0.540302 & 0 & 0 & 1
\end{bmatrix}
$$

and prints it

```c++
    for (auto row : jacobian)
    {
        for (auto elem : row) std::cout << elem << " ";
        std::cout << std::endl;
    }
```

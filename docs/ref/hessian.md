# Hessian

## Overview

XAD implements a set of methods to compute the Hessian matrix of a function in `XAD/Hessian.hpp`.

Note that the Hessian header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.

Hessians can be computed in `fwd_adj` or `fwd_fwd` higher-order mode.

The `computeHessian()` method takes a set of variables packaged in a
`std::vector<T>` and a function with signature `T foo(std::vector<T>)`.

## Return Types

If provided with `RowIterators`, `computeHessian()` will write directly to
them and return `void`. If no `RowIterators` are provided, the Hessian will
be written to a `std::vector<std::vector<T>>` and returned (`T` is
usually `double`).

## Specialisations

### Forward over Adjoint Mode

```c++
template <typename RowIterator, typename T>
void computeHessian(
    const std::vector<AReal<FReal<T>>> &vec,
    std::function<AReal<FReal<T>>(std::vector<AReal<FReal<T>>> &)> foo,
    RowIterator first, RowIterator last,
    Tape<FReal<T>> *tape = Tape<FReal<T>>::getActive())
```

This mode uses a [Tape](tape.md) to compute second derivatives. This Tape
will be instantiated within the method or set to the current active Tape using
`Tape::getActive()` if none is passed as argument.

### Forward over Forward Mode

```c++
template <typename RowIterator, typename T>
void computeHessian(
    const std::vector<FReal<FReal<T>>> &vec,
    std::function<FReal<FReal<T>>(std::vector<FReal<FReal<T>>> &)> foo,
    RowIterator first, RowIterator last)
```

This mode does not require a Tape which can help reduce the overhead that comes
with it, at the expense of requiring more executions of the function given to
determine the full Hessian.

## Example Use

Given $f(x, y, z, w) = sin(x y) - cos(y z) - sin(z w) - cos(w x)$, or

```c++
auto foo = [](std::vector<AD> &x) -> AD
{
    return sin(x[0] * x[1]) - cos(x[1] * x[2])
         - sin(x[2] * x[3]) - cos(x[3] * x[0]);
};
```

with the derivatives of interest calculated at the point

```c++
std::vector<AD> x_ad({1.0, 1.5, 1.3, 1.2});
```

we'd like to compute the Hessian

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} &
\frac{\partial^2 f}{\partial x \partial y} &
\frac{\partial^2 f}{\partial x \partial z} &
\frac{\partial^2 f}{\partial x \partial w} \\
\frac{\partial^2 f}{\partial y \partial x} &
\frac{\partial^2 f}{\partial y^2} &
\frac{\partial^2 f}{\partial y \partial z} &
\frac{\partial^2 f}{\partial y \partial w} \\
\frac{\partial^2 f}{\partial z \partial x} &
\frac{\partial^2 f}{\partial z \partial y} &
\frac{\partial^2 f}{\partial z^2} &
\frac{\partial^2 f}{\partial z \partial w} \\
\frac{\partial^2 f}{\partial w \partial x} &
\frac{\partial^2 f}{\partial w \partial y} &
\frac{\partial^2 f}{\partial w \partial z} &
\frac{\partial^2 f}{\partial w^2}
\end{bmatrix}
$$

First step is to setup the tape and active data types

```c++
    typedef xad::fwd_adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    tape_type tape;
```

Note that if no tape is setup, one will be created when computing the Hessian.
`fwd_fwd` mode is also supported in the same fashion. All that is left to do
is define our input values and our function, then call `computeHessian()`:

```c++
    std::function<AD(std::vector<AD> &)> foo = [](std::vector<AD> &x) -> AD
    { return sin(x[0] * x[1]) - cos(x[1] * x[2])
           - sin(x[2] * x[3]) - cos(x[3] * x[0]); };

    auto hessian = computeHessian(x_ad, foo);
```

Note the signature of `foo()`. Any other signature will throw an error.

This computes the relevant matrix

$$
H = \begin{bmatrix}
-1.72257 & -1.42551 & 0 & 1.36687 \\
-1.42551 & -1.6231 & 0.207107 & 0 \\
0 & 0.207107 & 0.607009 & 1.54911 \\
1.36687 & 0 & 1.54911 & 2.05226
\end{bmatrix}
$$

and prints it

```c++
    for (auto row : hessian)
    {
        for (auto elem : row) std::cout << elem << " ";
        std::cout << std::endl;
    }
```

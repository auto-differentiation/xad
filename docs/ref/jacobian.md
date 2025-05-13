# Jacobian

## Overview

XAD implements a set of methods to compute the Jacobian matrix of a function in `XAD/Jacobian.hpp`.

Note that the Jacobian header is not automatically included with `XAD/XAD.hpp`.
Users must include it as needed.

Jacobians can be computed in `adj` or `fwd` mode.

The `computeJacobian()` function takes a set of variables packaged in a
`std::vector<T>` and a function with signature
`std::vector<T> foo(std::vector<T>)`, where `T` is either a forward-mode
or adjoint-mode active type (`FReal` or `AReal`).

Optionally, the function also takes the codomain of the input function as an
`unsigned int` and/or the active `Tape` to be used if `adj` mode is specified.

If the codomain is not passed, an extra function evaluation will be required.

## Return Types

If provided with `RowIterators`, `computeJacobian()` will write directly to
them and return `void`. If no `RowIterators` are provided, the Jacobian will be
written to a `std::vector<std::vector<T>>` and returned, where `T` is the
underlying passive type (usually `double`).

## Specialisations

### Adjoint Mode

```c++
template <typename RowIterator, typename T>
void computeJacobian(
    const std::vector<AReal<T>> &vec,
    std::function<std::vector<AReal<T>>(std::vector<AReal<T>> &)> foo,
    RowIterator first, RowIterator last,
    std::size_t codomain = 0U,
    Tape<T> *tape = Tape<T>::getActive())
```

This mode uses a [Tape](ref/tape.md) to compute derivatives. This Tape will
be instantiated within the method or set to the current active Tape using
`Tape::getActive()` if none is passed as argument.

### Forward Mode

```c++
template <typename RowIterator, typename T>
void computeJacobian(
    const std::vector<FReal<T>> &vec,
    std::function<std::vector<FReal<T>>(std::vector<FReal<T>> &)> foo,
    RowIterator first, RowIterator last,
    std::size_t codomain = 0U)
```

This mode does not require a Tape and can help reduce the overhead that
comes with one. It is recommended for functions that have a higher number
of outputs than inputs.

## Example Use

Given $f(x, y, z, w) = [sin(x + y), sin(y + z), cos(z + w), cos(w + x)]$, or

```c++
std::function<std::vector<AD>(std::vector<AD>&)> foo =
[](std::vector<AD> &x) -> std::vector<AD> {
    return {sin(x[0] + x[1]),
            sin(x[1] + x[2]),
            cos(x[2] + x[3]),
            cos(x[3] + x[0])};
};
```

with the derivatives calculated at the following point

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
    auto foo = [](std::vector<AD>& x) -> std::vector<AD>
    {
        return {sin(x[0] + x[1]),
                sin(x[1] + x[2]),
                cos(x[2] + x[3]),
                cos(x[3] + x[0])};
    };

    auto jacobian = computeJacobian(x_ad, foo);
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

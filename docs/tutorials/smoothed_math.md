---
title: "Smooth Handling of Discontinuities"
description: "Learn how XAD tackles discontinuities in functions using smoothed math for accurate derivatives."
hide:
  - toc
---

# Handling Discontinuities

Many functions have jumps or discontinuities
at which points no mathematical derivatives exist.
These are typically written as conditionals in the source code
or by using the math functions `abs`, `max`, or `min`.

XAD generally defines the derivatives of standard math functions
as the average of the left and right derivatives at the discontinuity points.
For example, the derivative of `abs(x)` at point `x = 0` is set to `0`,
as the left derivative is `-1` and the right derivative is `1`.

As this definition is not mathematically accurate,
and as this creates problems with higher order derivatives,
XAD provides a set of smoothed math functions which are differentiable
at all points and can be used as a replacement for the original function.
They are implemented to provide accurate derivatives outside
a small area around the discontinuity,
and approximate the original function using a spline within this area.

As an example, the [`smooth_abs`](../ref/smooth-math.md#smooth_abs) function is illustrated in the
figure below (with `c = 0.001`):

<figure markdown>
![smooth_abs function](../images/sabs.svg)
</figure>

Note that discontinuities may be hidden in conditional constructs in the
original code.
In order to benefit from the smoothed math functions,
the conditionals need to be replaced by functions.
For example:

=== "Python"

    ```python
    # original code
    y = 0.0
    if value > strike:
        y = value - strike
    
    # equivalent smoothed code
    y = math.smooth_max(0.0, value - strike)
    ```


=== "C++"

    ```c++
    // original code
    double y = 0;
    if (value > strike)
        y = value - strike;
        
    // equivalent smoothed code
    double y = smooth_max(0, value - strike);
    ```

A reference of all provided smoothed math functions is given in [Smoothed Math Functions](../ref/smooth-math.md).

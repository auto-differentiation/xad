---
description: Smoothed versions of non-differentiable functions provided for convenience with XAD.
---

# Smoothed Mathematical Functions

The functions in this section are smoothed equivalents of the original
math functions and can be used to allow computing derivatives around discontinuities.

#### `smooth_abs`

`#!c++ T smooth_abs(T x, T c = 0.001)` is a smoothed version of `abs`, defined as:
   
$$
\text{smooth\_abs}(x,c) = \left\{\begin{array}{lll}
|x| & & \text{if }x < c\text{ or } x > c\\[5pt]
x^2\left(\frac{2}{c}-\frac{1}{c^2} x\right) & & \text{if }0 \leq x \leq c\\[5pt]
x^2\left(\frac{2}{c}+\frac{1}{c^2} x\right) & & \text{if }-c \leq x < 0
\end{array}\right.
$$
   
- `x` is the input value
- `c` is the cut-off point for the spline-approximated area (default: `0.001`)
- __returns__: The smoothed absolute value, defined as above.
      
#### `smooth_max`

`#!c++ T smooth_max(T x, T y, T c = 0.001)` is a smoothed version of `max`, defined as:
   
$$
\text{smooth\_max}(x,y,c) = 0.5\left(x+y+\text{smooth\_abs}(x-y,c)\right) 
$$

- `x` First argument to max
- `y` Second argument to max
- `c` Cut-off point for the spline-approximated area (default: `0.001`)
- __returns__: The smoothed max function, defined as above
   
#### `smooth_min`

`#!c++ T smooth_min(T x, T y, T c = 0.001)` is a smoothed version of `min`, defined as:
   
$$
\text{smooth\_min}(x,y,c) = 0.5\left(x+y-\text{smooth\_abs}(x-y,c)\right) 
$$ 

- `x` First argument to min
- `y` Second argument to min
- `c` Cut-off point for the spline-approximated area (default: `0.001`)
- __returns__: The smoothed min function, defined as above
   

   
   
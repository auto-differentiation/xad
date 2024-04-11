# Global Functions

This section lists functions that are specific to the active data types
and tape management.
For mathematical functions, [Mathematical Operations](math.md).

#### `value`

`#!c++ value(T& x)` returns a reference (or const-reference if `x` is constant) to the value
stored in `x`.

If `x` is a XAD active data type, such as [`AReal`](areal.md) or [`FReal`](freal.md),
this function returns a reference to the stored value (which is assignable).

If `x` is a passive data type, this function simple returns the value itself.

This function is especially useful in generic code, as it is defined on any data type.

#### `derivative`

`#!c++ derivative(const T& x)` returns a reference (or const-reference if `x` is constant) to the derivative stored in `x`.

If `x` is a XAD active data type, such as [`AReal`](areal.md) or [`FReal`](freal.md),
this function returns a reference to the stored derivative (which is assignable).

If `x` is a passive data type, this function simply returns 0.

This function is especially useful in generic code, as it is defined on any data type.

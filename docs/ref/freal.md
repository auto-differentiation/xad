---
description: >
  The class FReal defines an active data type for forward mode automatic differentiation
  in XAD.
---

# Forward Mode Type `FReal`

## Overview

```c++
template <typename T>
class FReal : public Expression<T, FReal<T>>
```

The class `FReal` defines an active data type for forward mode for the underlying type `T`
which tracks derivatives without tape.
It is designed to behave exactly like the built-in type `double`,
with all mathematical operations defined for this custom type.

It consists of a value and a derivative, both of which are tracked through
operations on this class. The derivative of at least one independent
variable should be set to 1 before the computation starts to ensure derivative
propagation to the outputs.

!!! note "See also"

    [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

## Member Functions

### Types

#### `value_type`

The value-type of this class, i.e., `T`.

### Constructors and Destructors

```c++
FReal(const T& val = T(), const T& der = T()) // (1) construct from value(s)
FReal(const FReal& val)                // (2) copy-constructor
FReal(FReal&& o)                       // (3) move-constructor
FReal(const Expression<T,Expr>& expr)  // (4) from expression
~FReal()                               // (5) destructor
```

The constructors create new instances of this class.

Variant `(1)` creates an instance from a value, as well as optionally sets
its derivativ.

Variants `(2)` and `(3)` copy or move from other values.

Variant `(4)` creates a value from an expression template, evaluating the expression
and recording both the result and the derivative of the right hand side expression.
It gets triggered for example by expressions like this:

```c++
FReal<double> y = x + x*sin(x);
```

If `x` is an instance of `#!c++ FReal<double>` itself.

The destructor `(5)` destroys the object.

### Assignments

```c++
FReal& operator=(const T &val)     // (1) assign from a scalar value
FReal& operator=(const FReal& val) // (2) copy-assignment
FReal& operator=(FReal&& val)      // (3) move-assignment
FReal& operator=(const Expression<T,Expr>& expr)  // (4) from expression
```

These assignment operators for `FReal` behave similar to the equivalent
constructors above.

### Values and Derivatives

#### `getValue`

`#!c++ T getValue() const` returns the value as the underlying type (without tape information).

#### `getDerivative`

`#!c++ T getDerivative() const` returns the derivative.

#### `setDerivative`

`#!c++ void setDerivative(const T& a)` sets the derivative in the object.
Typically this is called on independent variables before the operation is started
(if not already initialised using the constructor).

#### `value`

`#!c++ T& value()` and `#!c++ const T& value() const` return a reference to the underlying
passive type value.
This can be used to assign a value to the variable without affecting the derivative, as `#!c++ x.value() = 1.2`.

#### `derivative`

`#!c++ T& derivative()` and `#!c++ const T& derivative() const` return a reference to the derivative value.
This can be used to assign a value to it as well, as `#!c++ x.derivative() = 1.0`,
which is equivalent to `setDerivative`.
It can also be used as a replacement for `getDerivative`.

## Other Operations

In addition, `FReal` instances support all other mathematical arithmetic operations,
such as `operator+=` and friends.
Also, since `FReal` is an [`Expression`](expressions.md),
all math functions defined for expressions also work on instances of this class.

---
description: >
  The class AReal defines an active data type for adjoint mode in XAD, 
  which tracks derivative information on a tape.
---

# Adjoint Mode Type `AReal`

## Overview

```c++
template <typename T>
class AReal : public Expression<T, AReal<T>>
```

The class `AReal` defines an active data type for adjoint mode for the underlying type `T`,
which tracks derivative information on a tape.
It is designed to behave exactly like the built-in type `double`,
with all mathematical operations defined for this custom type.

Derivatives will only be tracked on tape if the variable has been registered
or is dependent on other registered variables.
Hence creating and using variables without an active tape is not problematic.

!!! note "See also"

    [Tape](tape.md), [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

## Member Functions

### Types

#### `tape_type`

The type of the tape that is used to store operations on this class.

#### `slot_type`

The type used for storing this instance's slot in the tape.
This type is useful for checkpointing, where the slot of the inputs and
outputs needs to be stored in the checkpoint in order to retrieve or
increment their derivatives during adjoint computation.

#### `value_type`

The value-type of this class, i.e., `T`.

### Constructors and Destructors

```c++
AReal(const T& val = T())     // (1) construct from value or default-construct
AReal(const AReal& val)       // (2) copy-constructor
AReal(AReal&& o)              // (3) move-constructor
AReal(const Expression<T,Expr>& expr)  // (4) from expression
~AReal()                      // (5) destructor
```

The constructors create new instances of this class.

Variant `(1)` creates a value that is not connected to any tape (it can be registered
explicitly using `tape.registerInput()`).

Variants `(2)` and `(3)` copy or move from other values, taping the operation
if the source type has been registered with a tape.

Variant `(4)` creates a value from an expression template, evaluating the expression
and recording the operations on tape if any of the variables on the right-hand side
have been registered with a tape. It gets triggered for example by expressions like this:

```c++
AReal<double> y = x + x*sin(x);
```

If `x` is an instance of `#!c++ AReal<double>` itself.

The destructor `(5)` unregisters the variable from the tape if applicable.

### Assignments

```c++
AReal& operator=(const T &val)     // (1) assign from a scalar value
AReal& operator=(const AReal& val) // (2) copy-assignment
AReal& operator=(AReal&& val)      // (3) move-assignment
AReal& operator=(const Expression<T,Expr>& expr)  // (4) from expression
```

These assignment operators for `AReal` behave similar to the equivalent
constructors above.

### Values and Derivatives

#### `getValue`

`#!c++ T getValue() const` returns the value as the underlying type (without tape information).

#### `getDerivative`

`#!c++ T getDerivative() const` returns the derivative (adjoint) as stored on the
tape (typically after rolling back the operation).
It throws an instance of [`#!c++ NoTapeException`](exceptions.md) if the variable
has not been registered with an active tape.

#### `setDerivative`

`#!c++ void setDerivative(const T& a)` sets the derivative (adjoint) on the tape.
Typically this is called in the function outputs after recording the operation,
before rolling back the tape.

#### `setAdjoint`

Alias for `setDerivative`.

#### `value`

`#!c++ T& value()` and `#!c++ const T& value() const` return a reference to the underlying
passive type value.
This can be used to assign a value to the variable without tape recording, as `#!c++ x.value() = 1.2`.

#### `derivative`

`#!c++ T& derivative()` and `#!c++ const T& derivative() const` return a reference to the underlying adjoint value.
This can be used to assign a value to the adjoint, as `#!c++ x.derivative() = 1.0`,
which is equivalent to `setDerivative`.
It can also be used as a replacement for `getDerivative`.

#### `shouldRecord`

`#!c++ bool shouldRecord() const` checks if the variable has been registered with a tape and should therefore
be recorded.

## Other Operations

In addition, `AReal` instances support all other mathematical arithmetic operations,
such as `operator+=` and friends.
Also, since `AReal` is an [`Expression`](expressions.md),
all math functions defined for expressions also work on instances of this class.

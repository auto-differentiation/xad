# AD Mode Interface

XAD provides a set of interface structures that conveniently allow access
to the data types needed for different AD modes.
These are traits classes with member `#!c++ typedef` declarations
that allow easy access to mode-specific types.

For example, they can be used as:

```c++
using mode = xad::adj<double>;
using adouble = mode::active_type;

// use active type in functions, etc.
adouble my_function(adouble x) {...}

mode::tape_type tape;    // setup tape
```

## Mode Interface Classes

### First Order

#### `adj<T>`

Mode interface class for first order adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

#### `fwd<T>`

Mode interface class for first order forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

### Second Order

#### `fwd_adj<T>`

Mode interface class for second order forward over adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

#### `fwd_fwd<T>`

Mode interface class for second order forward over forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

#### `adj_fwd<T>`

Mode interface class for second order adjoint over forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

#### `adj_adj<T>`

Mode interface class for second order adjoint over adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`).

## Type Members

All mode classes above have type members as described below.

```c++
template <typename T>
struct mode {
  typedef implementation_defined active_type;   // active data type
  typedef T                      passive_type;  // fully unwrapped passive type
  typedef implementation_defined tape_type;     // tape (void for forward mode)
  typedef passive_type           value_type;    // alias for passive_type
    
  // for second-order only
  typedef implementation_defined inner_type;    // first-order active type
};
```

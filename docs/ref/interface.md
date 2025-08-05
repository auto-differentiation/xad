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

#### `adj<T, N = 1>`

Mode interface class for first order adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwd<T, N = 1>`

Mode interface class for first order forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `adjd<T, N = 1>`

Mode interface class for first order direct forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwdd<T, N = 1>`

Mode interface class for first order direct adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

### Second Order

#### `fwd_adj<T, N = 1>`

Mode interface class for second order forward over adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwd_fwd<T, N = 1>`

Mode interface class for second order forward over forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `adj_fwd<T, N = 1>`

Mode interface class for second order adjoint over forward mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `adj_adj<T, N = 1>`

Mode interface class for second order adjoint over adjoint mode, where `T` is the underlying scalar type (e.g. `#!c++ double`),
and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwd_fwdd<T, N = 1>`

Mode interface class for second order forward over direct forward mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adj_fwdd<T, N = 1>`

Mode interface class for second order adjoint over direct forward mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwd_adjd<T, N = 1>`

Mode interface class for second order forward over direct adjoint mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adj_adjd<T, N = 1>`

Mode interface class for second order adjoint over direct adjoint mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwdd_adj<T, N = 1>`

Mode interface class for second order direct forward over adjoint mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwdd_fwd<T, N = 1>`

Mode interface class for second order direct forward over forward mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwdd_fwdd<T, N = 1>`

Mode interface class for second order direct forward over direct forward mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adjd_fwd<T, N = 1>`

Mode interface class for second order direct adjoint over forward mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adjd_adj<T, N = 1>`

Mode interface class for second order direct adjoint over adjoint mode, where `T` is the underlying scalar type
(e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adjd_adjd<T, N = 1>`

Mode interface class for second order direct adjoint over direct adjoint mode, where `T` is the underlying
scalar type (e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `adjd_fwdd<T, N = 1>`

Mode interface class for second order direct adjoint over direct forward mode, where `T` is the underlying
scalar type (e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

#### `fwdd_adjd<T, N = 1>`

Mode interface class for second order direct forward over direct adjoint mode, where `T` is the underlying
scalar type (e.g. `#!c++ double`), and `N` adds the option of having a vector of derivatives of size `N`.

## Type Members

All mode classes above have type members as described below.

```c++
template <typename T, size_t N = 1>
struct mode {
  typedef implementation_defined active_type;   // active data type
  typedef T                      passive_type;  // fully unwrapped passive type
  typedef implementation_defined tape_type;     // tape (void for forward mode)
  typedef passive_type           value_type;    // alias for passive_type

  // for second-order only
  typedef implementation_defined inner_type;    // first-order active type
};
```

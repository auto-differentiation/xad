# Exceptions

Generally, functions that throw XAD exceptions give mostly the *strong guarantee*,
i.e. the state of the objects involved is unchanged and operations can
continue as if the throwing function was not called.

Some functions can only give the *weak guarantee*,
i.e., the object is in an undefined state but the application can recover
be re-initializing the object.
This is notably the case if the user-defined [`CheckpointCallback::computeAdjoint`](chkpt_cb.md#computeadjoint) function throws an exception.

XAD defines the following exception types.

#### `xad::Exception`

`#!c++ class Exception : public std::runtime_error`

Base class of all XAD exceptions, including the standard method `const char* what() const`
to return the message of the exception (inherited).

#### `xad::TapeAlreadyActive`

`#!c++ class TapeAlreadyActive : public Exception`

Exception that is thrown when a tape is attempted to be activated while
another one is already active for the current thread.

#### `OutOfRange`

`#!c++ class OutOfRange : public Exception`

Exception thrown when an argument is out of the acceptable range.

#### `DerivativesNotInitialized`

`#!c++ DerivativesNotInitialized : public Exception`

Exception thrown if adjoints are attempted to be computed without setting
at least one derivative first.

#### `NoTapeException`

`#!c++ NoTapeException : public Exception`

Exception thrown if a derivative of an [`AReal`](areal.md#derivative) object is created without an active tape for the current thread.

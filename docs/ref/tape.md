# Tape

## `Tape`

`#!c++ template <typename T> class Tape;`

Tape data type to record operations for adjoint computations,
using the underlying scalar type `T` (which may in turn be an active type
for higher-order derivative calculations).

Typical usage

```c++
Tape<double> tape;
// initialize independent variables
AReal<double> x1 = 1.2, x2 = 12.1;
// register independents with the tape
tape.registerInput(x1);
tape.registerInput(x2);
// start recording derivatives on tape
tape.newRecording();
AReal<double> y = sin(x1) + x1*x2;
// register output and set adjoint values
tape.registerOutput(y);
derivative(y) = 1.0;
// compute the adjoints of the independent variables
tape.computeAdjoints();
// output/use results
std::cout << "y = " << value(y) << "\n"
        << "dy/dx1 = " << derivative(x1) << "\n"
        << "dy/dx2 = " << derivative(x2) << "\n";
```

For usability, it is recommended to use the type definitions decribed in
[AD Mode Interfaces](interface.md) instead of using this tape type directly.

### Member Typedefs

#### `size_type`

Type for sizes

#### `slot_type`

Type used to represent a slot of a specific active variable

#### `position_type`

Type to represent a position in the tape (same as `slot_type`)

#### `active_type`

Active data type that records on this type of tape

#### `value_type`

The value type of the tape, i.e. `T`

#### `tape_type`

The tape's type itself - for generic code

#### `callback_type`

The callback type used for checkpoints, i.e. `#!c++ CheckpointCallback<tape_type>*`

### Construct, Destruct, and Assign

A tape can be created and moved, but it is not copyable.

```c++
explicit Tape(bool activate = true);   // (1) constructor
Tape(Tape&&);                          // (2) move-constructor
Tape& operator=(Tape&&);               // (3) move-assignment
~Tape();                               // (4) destructor
```

The constructor `(1)` constructs a new tape, and activates it if needed.
If `active` is `#!c++ true`, a global thread-local pointer is set to this constructed instance,
resulting all operations and instantiations of active data types that follow
to get automatically associated with this tape instance.

It may throw [`TapeAlreadyActive`](exceptions.md), if `activate` is true and another tape is already active for the current thread

The other constructors facilitate moving a tape, but it cannot be copied.

### Recording Control

#### `activate`

`#!c++ void activate()` sets a global thread-local pointer to this tape instance,
resulting all `registerInput` calls and operations of active data types depending
on such inputs to get associated with this tape instance.

It may throw [`TapeAlreadyActive`](exceptions.md) if another tape is
already active for the current thread.

#### `deactivate`

`#!c++ void deactivate()` resets the global thread-local pointer to NULL, hence deactivating this tape.

#### `isActive`

`#!c++ bool isActive() const` check if the current instance is the currently active tape.

#### `getActive`

`#!c++ static Tape* getActive()` get a pointer to the currently active tape,
or `!c++ nullptr` if no active tape has been set.

Note that this is a thread-local pointer - calling this function in different
threads gives different results.

#### `setActive`

`#!c++ static void setActive(Tape* t)` static function that sets the given tape as the
globally active one. This is equivalent to `t.activate()`.

It may throw [`TapeAlreadyActive`](exceptions.md) if another tape is
already active for the current thread.

#### `deactivateAll`

`#!c++ static void deactivateAll()` deactivates any currently active tapes.
Equivalent to `auto t = Tape::getActive(); if (t) t->deactivate();`.

#### `registerInput`

`#!c++ void registerInput(active_type& inp)` registers the given variable with the tape and start recording dependents of it. A call to this function or its overloads is required in order to calculate adjoints.

Other overloads are:

*   `#!c++ void registerInput(std::complex<active_type>& inp)` for complex values

#### `registerInputs`

`#!c++ template <typename Inner> void registerInputs(std::vector<Inner>& v)` is a convenience function to register all variables in a vector as an input.

`#!c++ template <typename It> void registerInputs(It first, It last)` is a convenience
iterator interface to register variables in a range with the tape.

#### `registerOutput`

`#!c++ void registerOutput(active_type& inp)` registers the given variable as an output with the tape. A call to this function or its overloads is required in order to allow seeding derivatives (adjoints).

Other overloads are:

*   `#!c++ void registerOutput(std::complex<active_type>& inp)` registers a complex-valued output.

#### `registerOutputs`

`#!c++ template <typename Inner> void registerOutputs(std::vector<Inner>& v)` is a convenience function to register all variables in a vector as an input.

`#!c++ template <typename It> void registerOutputs(It first, It last)` is a convenience iterator interface to register variables in a range with the tape.

#### `newRecording`

`#!c++ void newRecording()` starts recording derivatives.

This function should be called *after* the independent variables are
initialized and registered,
as the `computeAdjoints` method will roll back the adjoints until
the point where `newRecording` was called.

#### `computeAdjoints`

`#!c++ void computeAdjoints()` propagates adjoints by interpreting the operations on the tape.

This function should be called after the output derivatives (adjoints)
have been initialized to a non-zero value.

After this call, the derivatives of the independent variables are set and
can be obtained.

It throws [`DerivativesNotInitialized`](exceptions.md) if called without setting any derivative first.

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `getPosition`

`#!c++ position_type getPosition()` returns the current position in the tape as an opaque integer (its value is internal and should not be relied upon in client code).
This posiiton can later be used in the methods `clearDerivativesAfter`,
`resetTo`, and `computeAdjointsTo`.

#### `clearDerivativesAfter`

`#!c++ void clearDerivativesAfter(position_type pos)` clears all derivatives after the
given position in the tape (resets them to zero). Derivatives before this point keep their
value, meaning that further calls to `computeAdjoints` will potentially increment these
adjoints further.

#### `resetTo`

`#!c++ void resetTo(position_type pos)` resets the tape back to the given position. All statements recorded after this point will be discarded.

!!! warning

    If variables registered after the given postion (i.e. dependent variables computed
    after this position) are used again after a call to `resetTo`,
    the behaviour is undefined, as their slot in the tape is no longer valid.

#### `computeAdjointsTo`

`#!c++ void computeAdjointsTo(position_type pos)` works like `computeAdjoints`, but stops rolling back the adjoints at the given position in the tape.

#### `clearAll`

`#!c++ void clearAll()` clears the stored tape info and brings it back to its initial state.

While this clears the content, it leaves allocated memory untouched.
This may be a performance gain compared to repeated construction/destruction
of tapes of the same time, for example in a path-wise AD Monte-Carlo.

### Derivatives

#### `derivative`

`#!c++ T& derivative(slot_type s)` gets a reference to the derivative associated with the slot `s`.

It throws [`OutOfRange`](exceptions.md) if the given slot is not associated with a stored derivative.
(Note that it is only thrown in debug mode for performance reasons, otherwise the behaviour is undefined in this case)

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

The const version `#!c++ const T& derivative(slot_type s) const` gets a const reference to the derivative associated with the slot `s` and otherwise behaves the same.

#### `getDerivative`

`#!c++ T getDerivative(slot_type s) const` gets a copy of the value of the derivative associated with the slot `s`.

It throws [`OutOfRange`](exceptions.md) if the given slot is not associated with a stored derivative.
(Note that it is only thrown in debug mode for performance reasons, otherwise the behaviour is undefined in this case)

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `setDerivative`

`#!c++ void setDerivative(slot_type s, const T& v)` sets the value of the derivative associated with the slot `s` to `v`.

It throws [`OutOfRange`](exceptions.md) if the given slot is not associated with a stored derivative.
(Note that it is only thrown in debug mode for performance reasons, otherwise the behaviour is undefined in this case)

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `clearDerivatives`

`#!c++ void clearDerivatives()` resets all stored derivatives to 0 (but leaving the recorded data in place). This can be used to calculate derivatives w.r.t. multiple outputs,
as the same tape can be rolled back multiple times.

### Status

#### `printStatus`

`#!c++ void printStatus() const` prints the number of recorded operations, statements, and registered variables to stdout.

#### `getMemory`

`#!c++ std::size_t getMemory() const` returns the memory in bytes that is occupied by the tape.

### Checkpointing

#### `insertCallback`

`#!c++ void insertCallback(callback_type cb)` inserts a checkpoint callback into the tape.

During computing adjoints (`computeAdjoints`), this callback is called when the tape
reaches the current position,
allowing users to implement their own adjoint computation.

Note that the parameter is provided by pointer (`callback_type` is a pointer),
but the tape does not take ownership.
It is the responsibility of the user to free the memory for the callback object.
Alternatively, the [Checkpoint Callback Memory Management](#checkpoint-callback-memory-management) API can be used to have the
tape destroy the callbacks automatically.

#### `getAndResetOutputAdjoint`

`#!c++ T getAndResetOutputAdjoint(slot_type slot)` obtains the output adjoint stored
in `slot`and resets it to 0.

This function should be called by [`#!c++ CheckpointCallback<TapeType>::computeAdjoints`](chkpt_cb.md) to get the current value of the adjoint.
It also resets its adjoint to 0 on the tape to allow re-use of that variable.

It throws [`OutOfRange`](exceptions.md) if the given slot is not associated with a stored derivative.
(Note that it is only thrown in debug mode, otherwise the behaviour is undefined)

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `incrementAdjoint`

`#!c++ void incrementAdjoint(slot_type slot, const T& x)` increments the adjoint of the given slot by the value `x`.

This function should be called at the end of a [`CheckpointCallback<TapeType>::computeAdjoints`](chkpt_cb.md)
implementation, to update the input adjoints with the
computed adjoint increments.

It throws [`OutOfRange`](exceptions.md) if the given slot is not associated with a stored derivative.
(Note that it is only thrown in debug mode, otherwise the behaviour is undefined)

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `newNestedRecording`

`#!c++ void newNestedRecording()` starts a new nested recording that can be rolled-back on its own.
It must be ended with `endNestedRecording`.

It is intended for use within a [`CheckpointCallback<TapeType>::computeAdjoints`](chkpt_cb.md) implementation,
when from a checkpoint, the adjoints are computed using XAD in a nested recording.

To avoid forgetting the call to `endNestedRecording`, consider
using the RAII class [`ScopedNestedRecording`](#scopednestedrecording).

#### `endNestedRecording`

`#!c++ void endNestedRecording()` ends a nested recording.

### Checkpoint Callback Memory Management

#### `pushCallback`

`#!c++ void pushCallback(callback_type cb)` lets this tape handle the de-allocation of the given callback, destroying the dynamically-allocated `cb`.

When the tape is destructed, it also destructs all callbacks that have been
registered using this function.

Use this if checkpoints are created in a stateless function to avoid
having to track and destroy checkpoint callbacks manually.

#### `getLastCallback`

`#!c++ callback_type getLastCallback()` obtains the last [`CheckpointCallback`](chkpt_cb.md) object that has been pushed with `pushCallback`.

This can be useful if multiple subsequent checkpoints can be added to the
same checkpoint callback object.

Throws `OutOfRange`: if the callback stack is empty.

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

#### `getNumCallbacks`

`#!c++ size_type getNumCallbacks() const` gets the number of callback objects that have been pushed by `pushCallback`.

#### `haveCallbacks`

`#!c++ bool haveCallbacks() const` checks if there have been any checkpoint callbacks registered by `pushCallback`.

#### `popCallback`

`#!c++ void popCallback()` removes the callback object that has been last pushed by
`pushCallback`. It throws [`OutOfRange`](exceptions.md) if the stack of callbacks is empty

Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

## `ScopedNestedRecording`

```c++
template <typename TapeType>
class ScopedNestedRecording
```

Convenience RAII class to ensure that a call to `Tape<T>::newNestedRecording`
is always followed by the corresponding `Tape<T>::endNestedRecording`.

It should be constructed on the stack. On creation it starts a nested
recording on the corresponding tape, and on destruction it ends the nested
recording.
This is useful for checkpoint callbacks, i.e. within the implementation of
[`CheckpointCallback<TapeType>::computeAdjoints`](chkpt_cb.md).

### Construct and Destruct

```c++
ScopedNestedRecording(TapeType* t);
~ScopedNestedRecording();
```

The constructor starts a new nested recording on the given tape and track it with this object.
It calls `newNestedRecording` on the given tape.

The destructor calles `endNestedRecording` on the given tape automatically.

### Member Functions

#### `computeAdjoints`

`#!c++ void computeAdjoints()` computes adjoints within the nested recording.

#### `incrementAdjoint`

`#!c++ void incrementAdjoint(TapeType::slot_type slot, const TapeType::value_type& value)`
increments the adjoint given by the slot by the given value,
similar to [`Tape<T>::incrementAdjoint`](#incrementadjoint).

#### `getTape`

`#!c++ TapeType* getTape()` returns the underlying tape for this nested recording.

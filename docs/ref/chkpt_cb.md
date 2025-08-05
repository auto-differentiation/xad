# CheckpointCallback

```c++
template <typename TapeType>
class CheckpointCallback
```

Base class of any checkpoint callbacks associated with the given `TapeType`.

User-defined checkpoints should derive from this class and implement
`computeAdjoints` to implement the adjoint computation of the checkpoint.
An object of this class should then be registered with the tape using
[`Tape<T>::insertCallback`](tape.md#insertcallback).
During computing adjoints, XAD then calls this registered callback
once it reaches the corresponding point in the tape.

Implementers are responsible to store all needed information for the
checkpoint within this class.
That is:

*   The values of input variables,
*   Their slots in the tape,
*   The slots of the output variables on the tape,
*   Any other information that may be needed to propagate the adjoints
    of the outputs back to its input variables

#### `computeAdjoint`

`#!c++ virtual void computeAdjoint(TapeType* tape) = 0`

This function is called by XAD during rolling back the tape,
giving a reference to itself.
Gives *weak exception guarantee* - if this user-implemented function
throws an exception, the tape is in an undefined state and needs to
be reset/re-initialized before it can be used again.

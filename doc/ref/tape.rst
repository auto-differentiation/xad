.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-tape:

Tape
====

.. highlight:: cpp

Overview
--------

.. cpp:class:: template <typename T> Tape

   Tape data type to record operations for adjoint computations,
   using the underlying scalar type `T` (which may in turn be an active type
   for higher-order derivative calculations).

   Typical usage::

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

   For usability, it is recommended to use the type definitions in
   :ref:`ref-interface` instead of declaring this tape type directly.

Members
-------

.. cpp:namespace:: template <typename T> Tape

Types
^^^^^

.. cpp:type:: size_type

   Type for sizes

.. cpp:type:: slot_type

   Type used to represent a slot of a specific active variable

.. cpp:type:: position_type

   Type to represent a position in the tape (same as `slot_type`)

.. cpp:type:: AReal<T> active_type

   Active data type that records on this type of tape

.. cpp:type:: T value_type

   The value type of the tape, i.e. ``T``

.. cpp:type:: Tape<T> tape_type

   The tape's type itself - for generic code

.. cpp:type:: CheckpointCallback<tape_type>* callback_type

   The callback type used for checkpoints

Construct, Destruct, and Assign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A tape can be created and moved, but it is not copyable.

.. cpp:function:: explicit Tape(bool activate = true)

   Constructs a new tape, and activates it if needed.

   If ``active`` is ``true``, a global thread-local pointer is
   set to this constructed instance, resulting all operations and instantiations
   of active data types that follow
   to get automatically associated with this tape instance.

   :throws `TapeAlreadyActive`:
      if `activate` is true and another tape is already active
      for the current thread

.. cpp:function:: ~Tape()

   Destructor.

.. cpp:function:: Tape(Tape&&)

   Move-constructor.

.. cpp:function:: Tape& operator=(Tape&&)

   Move-assign.

Recording Control
^^^^^^^^^^^^^^^^^

.. cpp:function:: void activate()

   Sets a global thread-local pointer to this tape instance,
   resulting all ``registerInput`` calls and operations of 
   active data types depending on such inputs to get associated 
   with this tape instance.

   :throws `TapeAlreadyActive`:
      if another tape is already active for the current thread

.. cpp:function:: void deactivate()

   Resets the global thread-local pointer to NULL, hence deactivating this
   tape.

.. cpp:function:: bool isActive() const

   Check if the current instance is the currently active tape.

   :return: ``true`` if the this instance is active

.. cpp:function:: static Tape* getActive()

   Get a pointer to the currently active tape.

   Note that this is a thread-local pointer - calling this function in different
   threads gives different results.

   :return: Pointer to the currently-active thread-local tape - or ``nullptr``

.. cpp:function:: void registerInput(active_type& inp)

   Register the given variable with the tape and start recording dependents of it.
   A call to this function or its overloads is required in order to calculate adjoints.

.. cpp:function:: template <typename Inner> void registerInputs(std::vector<Inner>& v)

   Convenience function to register all variables in a vector as an input.

.. cpp:function:: template <typename It> void registerInputs(It first, It last)

   Convenience iterator interface to register variables in a range with the tape.

.. cpp:function:: void registerInput(std::complex<active_type>& inp)

   Register a complex-valued input.

.. cpp:function:: void registerOutput(active_type& inp)

   Register the given variable as an output with the tape.
   A call to this function or its overloads is required in order to allow seeding derivatives (adjoints).

.. cpp:function:: template <typename Inner> void registerOutputs(std::vector<Inner>& v)

   Convenience function to register all variables in a vector as an input.

.. cpp:function:: template <typename It> void registerOutputs(It first, It last)

   Convenience iterator interface to register variables in a range with the tape.

.. cpp:function:: void registerOutput(std::complex<active_type>& inp)

   Register a complex-valued output.


.. cpp:function:: void newRecording()

   Start recording derivatives.

   This function should be called *after* the independent variables are
   initialized and registered,
   as the :cpp:func:`computeAdjoints` method will roll back the adjoints until
   the point where :cpp:func:`newRecording` was called.

.. cpp:function:: void computeAdjoints()

   Propagates adjoints by interpreting the operations on the tape.

   This function should be called after the output derivatives (adjoints)
   have been initialized to a non-zero value.

   After this call, the derivatives of the independent variables are set and
   can be obtained.

   :throws `DerivativesNotInitialized`:
      If called without setting any derivative first.
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: position_type getPosition()

   Returns the current position in the tape as an opaque integer (its value is
   internal and should not be relied upon in client code). 
   This posiiton can later be used in the methods ``clearDerivativesAfter``,
   ``resetTo``, and ``computeAdjointsTo``.

.. cpp:function:: void clearDerivativesAfter(position_type pos)

   Clears all derivatives after the given position in the tape (resets them to zero).
   Derivatives before this point keep their value, meaning that further calls
   to ``computeAdjoints`` will potentially increment these adjoints further.

.. cpp:function:: void resetTo(position_type pos)

   Resets the tape back to the given position. All statements recorded after this point
   will be discarded.

   .. warning::

      If variables registered after the given postion (are dependent variables computed
      after this position) are used again after a call to ``resetTo``, 
      the behaviour is undefined, as their slot in the tape is no longer valid.

.. cpp:function:: void computeAdjointsTo(position_type pos)

   Like ``computeAdjoints``, but stops rolling back the adjoints at the given position
   in the tape.

.. cpp:function:: void clearAll()

   Clears the stored tape info and brings it back to its initial state.

   While this clears the content, it leaves allocated memory untouched.
   This may be a performance gain compared to repeated construction/destruction
   of tapes of the same time, for example in a path-wise AD Monte-Carlo.


Derivatives
^^^^^^^^^^^

.. cpp:function:: T& derivative(slot_type s)

   Get a reference to the derivative associated with the slot ``s``.

   :param s: The slot of the derivative
   :throws `OutOfRange`:
      if the given slot is not associated with a stored derivative. (Only
      thrown in debug mode,
      otherwise the behaviour is undefined in this case)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: const T& derivative(slot_type s) const

   Get a const reference to the derivative associated with the slot ``s``.

   :param s: The slot of the derivative
   :throws `OutOfRange`:
      if the given slot is not associated with a stored derivative. (Only
      thrown in debug mode,
      otherwise the behaviour is undefined in this case)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: T getDerivative(slot_type s) const

   Get the value of the derivative associated with the slot ``s``.

   :param s: The slot of the derivative
   :throws `OutOfRange`:
      if the given slot is not associated with a stored derivative. (Only
      thrown in debug mode,
      otherwise the behaviour is undefined in this case)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: void setDerivative(slot_type s, const T& v)

   Set the value of the derivative associated with the slot ``s``.

   :param s: The slot of the derivative
   :param v: The value to assign to the derivative
   :throws `OutOfRange`:
      if the given slot is not associated with a stored derivative. (Only
      thrown in debug mode,
      otherwise the behaviour is undefined in this case)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: void clearDerivatives()

   Resets all stored derivatives to 0 (but leaving the recorded data in place).
   This can be used to calculate derivatives w.r.t. multiple outputs,
   as the same tape can be rolled back multiple times.

Status
^^^^^^

.. cpp:function:: void printStatus() const

   Prints the number of recorded operations, statements, and registered variables
   to stdout.

.. cpp:function:: std::size_t getMemory() const

   Returns the memory in bytes that is occupied by the tape.

   :return: Memory in bytes

Checkpointing
^^^^^^^^^^^^^

.. cpp:function:: void insertCallback(callback_type cb)

   Insert a checkpoint callback into the tape.

   During computing adjoints (:cpp:func:`computeAdjoints`),
   this callback is called when the tape reaches the current position,
   allowing users to implement their own adjoint computation.

   Note that the parameter is provided by pointer, but the tape does not take
   ownership.
   It is the responsibility of the user to free the memory for the callback object.
   Alternatively, the :ref:`ref-tape-ckpt-mem` API can be used to have the
   tape destroy the callbacks automatically.

   :param cb: Pointer to a :cpp:class:`CheckpointCallback` instance.

.. cpp:function:: T getAndResetOutputAdjoint(slot_type slot)

   Obtains and resets the output adjoint to 0

   This function should be called by :cpp:func:`CheckpointCallback<TapeType>::computeAdjoints`
   to get the current value of the adjoint.
   It also resets its adjoint to 0 on the tape to allow re-use of that variable.

   :param slot: The slot of the output variable.
   :return: The value of the variable's derivative (i.e. its adjoint)
   :throws `OutOfRange`:
      If the given slot is not associated with a stored derivative.
      (Only thrown in debug mode, otherwise the behaviour is undefined)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.


.. cpp:function:: void incrementAdjoint(slot_type slot, const T& x)

   Increments the adjoint of the given slot by the value `x`.

   This function should be called at the end of a :cpp:func:`CheckpointCallback<TapeType>::computeAdjoints`
   implementation, to update the input adjoints with the
   computed adjoint increments.

   :param slot: Slot of the input variable to increment
   :param x: The value to be added to the adjoint
   :throws `OutOfRange`:
      If the given slot is not associated with a stored derivative.
      (Only thrown in debug mode, otherwise the behaviour is undefined)
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: void newNestedRecording()

   Starts a new nested recording that can be rolled-back on its own.
   It must be ended with :cpp:func:`endNestedRecording`.

   It is intended for use within a :cpp:func:`CheckpointCallback<TapeType>::computeAdjoints`
   implementation,
   when from a checkpoint, the adjoints are computed using XAD in a nested
   recording.

   To avoid forgetting the call to :cpp:func:`endNestedRecording`, consider
   using the RAII class :cpp:class:`ScopedNestedRecording`.

.. cpp:function:: void endNestedRecording()

   Ends a nested recording.

.. _ref-tape-ckpt-mem:

Checkpoint Callback Memory Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: void pushCallback(callback_type cb)

   Let this tape handle the de-allocation of the given callback.

   When the tape is destructed, it also destructs all callbacks that have been
   registered using this function.

   Use this if checkpoints are created in a stateless function to avoid
   having to track and destroy checkpoint callbacks manually.

   :param cb: Pointer to a dynamically-allocated checkpoint callback

.. cpp:function:: callback_type getLastCallback()

   Obtains the last :cpp:class:`CheckpointCallback` object that has been pushed
   with :cpp:func:`pushCallback`.

   This can be useful if multiple subsequent checkpoints can be added to the
   same checkpoint callback object.

   :return: Pointer to the last callback object that has been pushed.
   :throws `OutOfRange`: if the callback stack is empty
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.

.. cpp:function:: size_type getNumCallbacks() const

   Gets the number of callback objects that have been pushed by
   :cpp:func:`pushCallback`

   :return: Number of callback objects registered

.. cpp:function:: bool haveCallbacks() const

   Checks if there have been any checkpoint callbacks registered by
   :cpp:func:`pushCallback`

   :return: ``true`` if there is at least one pushed callback object.

.. cpp:function:: void popCallback()

   Removes the callback object that has been last pushed by
   :cpp:func:`pushCallback`

   :throws `OutOfRange`:
      if the stack of callbacks is empty
      Gives *strong exception safety guarantee* - tape state unchanged in case of exception.



Nested Tape
-----------

.. cpp:namespace:: 0

.. cpp:class:: template <typename TapeType> ScopedNestedRecording

   Convenience RAII class to ensure that a call to :cpp:func:`Tape<T>::newNestedRecording`
   is always followed by the corresponding :cpp:func:`Tape<T>::endNestedRecording`.

   It should be constructed on the stack. On creation it starts a nested
   recording on the corresponding tape, and on destruction it ends the nested
   recording.
   This is useful for checkpoint callbacks, i.e. within the implementation of
   :cpp:func:`CheckpointCallback<TapeType>::computeAdjoints`.

   .. cpp:function:: ScopedNestedRecording(TapeType* t)

      Start a new nested recording on the given tape and track it with this object.

      :param t: pointer to the associated tape

   .. cpp:function:: ~ScopedNestedRecording()

      Ends the nested recording with the associated tape

   .. cpp:function:: void computeAdjoints()

      Computes adjoints within the nested recording

   .. cpp:function:: void incrementAdjoint(TapeType::slot_type slot, const TapeType::value_type& value)

      Increment the adjoint given by the slot by the given value.
      See :cpp:func:`Tape<T>::incrementAdjoint` for details.

   .. cpp:function:: TapeType* getTape()

      Returns the underlying tape for this nested recording.

      :return: Pointer to the underlying tape

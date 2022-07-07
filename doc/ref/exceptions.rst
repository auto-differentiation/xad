.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
Exceptions
==========


.. highlight:: cpp

Generally, functions that throw XAD exceptions give mostly the *strong guarantee*,
i.e. the state of the objects involved is unchanged and operations can 
continue as if the throwing function was not called.

Some functions can only give the *weak guarantee*, 
i.e., the object is in an undefined state but the application can recover
be re-initializing the object.
This is notably the case if the user-defined :cpp:func:`CheckpointCallback::computeAdjoint`
function throws an exception.

XAD defines the following exception types.

.. cpp:class:: Exception : public std::runtime_error

   Base class of all XAD exceptions
   
   .. cpp:function:: const char* what() const
   
      Returns the message of the exception (inherited from :cpp:expr:`std::runtime_error`)
   
.. cpp:class:: TapeAlreadyActive : public Exception

   Exception that is thrown when a tape is attempted to be activated while
   another one is already active for the current thread.
   
.. cpp:class:: OutOfRange : public Exception

   Exception thrown when an argument is out of the acceptable range.
   
.. cpp:class:: DerivativesNotInitialized : public Exception

   Exception thrown if adjoints are attempted to be computed without setting
   at least one derivative first.
   
.. cpp:class:: NoTapeException : public Exception

   Exception thrown if an :cpp:class:`AReal` object is created without an active
   tape for the current thread (debug-mode only).
   
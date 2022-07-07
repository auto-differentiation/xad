.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-interface:

AD Mode Interface
=================

.. highlight:: cpp

XAD provides a set of interface structures that conveniently allow access
to the data types needed for different AD modes. 
These are described in the following.

First-Order Mode Interface
--------------------------

.. cpp:class:: template <typename T> adj

   Structure defining the data types for adjoint-mode first order differentiation.
   
   The template parameter `T` is the underlying basic type, typically `double`
   or `float`.
   
   .. cpp:type:: Tape<T> tape_type
   
      The data type for the tape in adjoint mode
   
   .. cpp:type:: AReal<T> active_type
   
      Active data type for adjoint mode differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as `passive_type`
      

.. cpp:class:: template <typename T> fwd

   Structure defining the data types for forward-mode (tangent-linear) first
   order differentiation.
   
   The template parameter `T` is the underlying basic type, typically `double`
   or `float`.
   
   .. cpp:type:: FReal<T> active_type
   
      Active data type for forward-mode differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as `passive_type`
      
   .. cpp:type:: void tape_type
   
      As forward mode does not require a tape, this is defined to `void`

Second-Order Mode Interface
---------------------------

Second order is performed by nesting two first-order modes.  

.. cpp:class:: template <typename T> fwd_adj

   Structure defining the data types for second order forward-over-adjoint mode.
   
   The template parameter ``T`` is the underlying basic type, typically ``double``
   or ``float``.
   
   .. cpp:type:: FReal<T> inner_type
   
      The type for the inner mode, applied for the second order. In this case this is 
      a forward AD type.
   
   .. cpp:type:: AReal<inner_type> active_type
   
      Active data type for forward-over-adjoint mode differentiation
   
   .. cpp:type:: Tape<inner_type> tape_type
   
      Tape type for forward-over-adjoint mode differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as ``passive_type``

.. cpp:class:: template <typename T> fwd_fwd

   Structure defining the data types for second order forward-over-forward mode.
   
   The template parameter `T` is the underlying basic type, typically ``double``
   or ``float``.
   
   .. cpp:type:: FReal<T> inner_type
   
      The type for the inner mode, applied for the second order. In this case this is 
      a forward AD type.
   
   .. cpp:type:: FReal<inner_type> active_type
   
      Active data type for forward-over-forward mode differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as ``passive_type``
      
   .. cpp:type:: void tape_type
   
      As not tape is required in this mode, this is defined as ``void``
      

.. cpp:class:: template <typename T> adj_fwd

   Structure defining the data types for second order adjoint-over-forward mode.
   
   The template parameter ``T`` is the underlying basic type, typically ``double``
   or ``float``.
   
   .. cpp:type:: AReal<T> inner_type
   
      The type for the inner mode, applied for the second order. In this case this is 
      a adjoint AD type.
   
   .. cpp:type:: FReal<inner_type> active_type
   
      Active data type for adjoint-over-forward mode differentiation
   
   .. cpp:type:: Tape<T> tape_type
   
      Tape type for adjoint-over-forward mode differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as ``passive_type``

.. cpp:class:: template <typename T> adj_adj

   Structure defining the data types for second order adjoint-over-adjoint mode.
   
   The template parameter ``T`` is the underlying basic type, typically ``double``
   or ``float``.
   
   .. cpp:type:: AReal<T> inner_type
   
      The type for the inner mode, applied for the second order. In this case this is 
      a adjoint AD type.
   
   .. cpp:type:: AReal<inner_type> active_type
   
      Active data type for adjoint-over-adjoint mode differentiation
   
   .. cpp:type:: Tape<T> inner_tape_type
   
      Tape type for the second order (inner) differentiation

   .. cpp:type:: Tape<inner_type> outer_tape_type
   
      Tape type for the first order (outer) differentiation
   
   .. cpp:type:: T passive_type
   
      Passive data type
   
   .. cpp:type:: passive_type value_type
   
      Same as ``passive_type``
      

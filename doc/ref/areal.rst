.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-areal:

Adjoint Mode Type ``AReal``
===========================

Overview
--------

.. highlight:: cpp

.. cpp:class:: template <typename T> AReal : public Expression<T, AReal<T> >

   Defines and active data type version of the underlying type `T` 
   that tracks derivatives on a tape for adjoint computation.

   Derivatives will only be tracked on tape if the variable has been registered
   or is dependent on other registered variables.
   Hence creating and using variables without an active tape is not problematic.
   
   .. seealso:: :ref:`ref-tape`, :ref:`ref-global`, :ref:`ref-interface`, :ref:`ref-math` 
   
Members
-------

.. cpp:namespace:: template <typename T> AReal

Types
^^^^^

.. cpp:type:: tape_type

   The type of the tape that is used to store operations on this class.

.. cpp:type:: slot_type

   The type used for storing this instance's slot in the tape. 
   This type is useful for checkpointing, where the slot of the inputs and 
   outputs needs to be stored in the checkpoint in order to retrieve or 
   increment their derivatives during adjoint computation.
   
.. cpp:type:: T value_type

   The value-type of this class, i.e., ``T``.

   
Construct, Destruct, and Assign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      
.. cpp:function:: AReal(const T& val = T())

   Constructs a new instance of this class, given its initial passive value.

   Ensure that there is an active tape instance for the current thread, 
   otherwise creating instances of this class results in undefined behaviour.
   This error condition is only checked in debug builds, where it throws
   an :cpp:class:`NoTapeException`. Release builds do not perform this check.

   :param val: Value to initialize the  the instance with.
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)
   

.. cpp:function:: AReal(const AReal& val)

   Copy-constructor
   
   :param val: Value to initialize the instance with.
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)
   
.. cpp:function:: AReal(AReal&& o)

   Move-constructor
   
   :param o: Other object to move from
   
.. cpp:function:: AReal(const Expression<T,Expr>& expr)

   Construct from an expression. This constructor gets called from statements 
   like this, where the right-hand side involves and active data type::
   
      AReal<double> y = x + x*sin(x);
   
   :param expr: The expression to construct from
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)

.. cpp:function:: AReal& operator=(const T& val)

   Assign from a passive value.
   
   :param val: Value to be assigned to this object.
   :return: A reference to ``this``
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)
   
.. cpp:function:: AReal& operator=(const AReal& val)

   Assign from another `AReal` object.
   
   :param val: Value to be assigned to this object.
   :return: A reference to ``this``
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)

.. cpp:function:: AReal& operator=(AReal&& val)

   Move-assignment
   
   :param val: Value to be moved into this object
   :return: A reference to ``this``
   
.. cpp:function:: AReal& operator=(const Expression<T,Expr>& expr)

   Assign an expression
   
   :param expr: Expression to be assigned to this object.
   :return: A reference to ``this``
   :throws `NoTapeException`: If there is no active tape for the current thread 
                              (debug mode only)

.. cpp:function:: ~AReal()

   Destructor. 
      
Values and Derivatives
^^^^^^^^^^^^^^^^^^^^^^
   
.. cpp:function:: T getValue() const

   Get the value of this object, as the underlying type.
   
   :return: The value of this object
   
.. cpp:function:: const T& value() const

   Get a const reference to the value of this object.
   
   :return: The value of this object

.. cpp:function:: T& value()

   Get a reference to the value of this object, i.e. it is assignable
   
   :return: Reference to the value of this object

.. cpp:function:: T getDerivative() const

   Get the stored derivative of this object.
   
   :return: The derivative (adjoint) of this object

.. cpp:function:: const T& derivative() const

   Get a const reference to the stored derivative of this object.
   
   :return: The derivative (adjoint) of this object
   :throw `OutOfRange`: If the derivatives have not been initialized yet

.. cpp:function:: T& derivative()

   Get a reference to the stored derivative of this object, i.e., it is assignable.
   
   :return: A reference to the derivative (adjoint) of this object

.. cpp:function:: void setDerivative(const T& a)

   Sets the derivative of this object. This is the same as calling ``derivative() = a``.
   
   :param a: The value to assign to the derivative.
   
   
.. cpp:function:: void setAdjoint(const T& a)

   Synonym for ``setDerivative(a)``.

.. cpp:function:: bool shouldRecord() const

   Checks if the variable has been registered with a tape and should therefore
   be recorded.


Other Operations
----------------

In addition, :cpp:class:`AReal` supports all other mathematical arithmetic operations, 
such as ``operator+=`` and friends. 
Also, as :cpp:class:`AReal` is an :cpp:class:`Expression`, 
all free math functions defined for expressions also work on instances of this class.



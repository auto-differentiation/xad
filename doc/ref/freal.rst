.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-freal:

Forward Mode Type ``FReal``
===========================

.. highlight:: cpp

Overview
--------

.. cpp:class:: template <typename T> FReal : public Expression<T, FReal<T> >

   Defines and active data type version of the underlying type `T` 
   that tracks derivatives for forward mode differentiation (without tape)
   
   It consists of a value and a derivative, both of which are tracked through
   operations on this class. The derivative of at least one independent
   variable should be set to 1 before the computation starts to ensure derivative
   propagation to the outputs.
   
   .. seealso:: :ref:`ref-global`, :ref:`ref-interface`, :ref:`ref-math` 
   
  
Members
-------

.. cpp:namespace:: template<typename T> FReal

Types
^^^^^

.. cpp:type:: T value_type

   The value-type of this class, i.e., ``T``.

   
Construct, Destruct, and Assign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      
.. cpp:function:: FReal(const T& val = T(), const T& der = T())

   Constructs a new instance of this class, given its initial passive value
   and derivative.

   :param val: Value to initialize the  the instance with.
   :param der: Derivative to initialize with (0 if not given).

.. cpp:function:: FReal(const FReal& val)

   Copy-constructor
   
   :param val: Value to initialize the instance with.
   
.. cpp:function:: FReal(FReal&& o)

   Move-constructor
   
   :param o: Other object to move from
   
.. cpp:function:: FReal(const Expression<T,Expr>& expr)

   Construct from an expression. This constructor gets called from statements 
   like this, where the right-hand side involves and active data type::
   
      FReal<double> y = x + x * sin(x);
   
   :param expr: The expression to construct from

.. cpp:function:: FReal& operator=(const T& val)

   Assign from a passive value. Sets the value to ``val`` and the derivative
   to zero.
   
   :param val: Value to be assigned to this object.
   :return: A reference to ``this``
   
.. cpp:function:: FReal& operator=(const FReal& val)

   Copy-assign from another `FReal` object.
   
   :param val: Value to be assigned to this object.
   :return: A reference to ``this``

.. cpp:function:: FReal& operator=(FReal&& val)

   Move-assignment
   
   :param val: Value to be moved into this object
   :return: A reference to ``this``
   
.. cpp:function:: FReal& operator=(const Expression<T,Expr>& expr)

   Assign an expression
   
   :param expr: Expression to be assigned to this object.
   :return: A reference to ``this``

      
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
   
   :return: The derivative stored in this object

.. cpp:function:: const T& derivative() const

   Get a const reference to the stored derivative of this object.
   
   :return: The derivative stored in this object

.. cpp:function:: T& derivative()

   Get a reference to the stored derivative of this object, i.e., it is assignable.
   
   :return: A reference to the derivative in this object

.. cpp:function:: void setDerivative(const T& a)

   Sets the derivative of this object. This is the same as calling ``derivative() = a``.
   
   :param a: The value to assign to the derivative.
   

Other Operations
----------------

In addition, :cpp:class:`FReal` supports all other mathematical arithmetic operations, 
such as ``operator+=`` and friends. 
Also, as :cpp:class:`FReal` is an :cpp:class:`Expression`, 
all free math functions defined for expressions also work on instances of this class.

.. seealso:: :ref:`ref-expression`, :ref:`ref-math`



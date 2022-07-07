.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-expression:

Expressions
===========

.. highlight:: cpp

Expression Template
-------------------

.. cpp:class:: template <typename T, typename Derived> Expression

   Represents a mathematical expression in a type. 
   Active data types, such as :cpp:class:`AReal` and :cpp:class:`FReal`, 
   as well as all mathematical expressions inherit from this class. 
   Therefore all mathematical operations are defined on this type, rather
   than any specific derived class.
   
   The derived classes are typically created transparently to the user.

   Note that this class uses the CRTP pattern, where ``Derived`` is
   the derived class itself, so that static polymorphism can be used.
   
   
All global arithmetic operations defined in C++ are specialized 
for :cpp:class:`Expression`, 
so that ``double`` or ``float`` can be replaced seamlessly with a XAD data type.
This also includes comparisons.

.. seealso:: :ref:`ref-math`
   

Expression Traits
-----------------

XAD also defines expression traits to find out information about expressions
in a templated context.
This is typically only needed when custom functions dealing with the XAD 
expressions are added.

.. cpp:enum:: Direction

   Enum to indicate the direction of algorithmic differentiation associated with a type.

   .. cpp:enumerator:: DIR_NONE

      Not an algorithmic differentiation type

   .. cpp:enumerator:: DIR_FORWARD

      Forward mode AD type

   .. cpp:enumerator:: DIR_REVERSE

      Reverse mode AD type

.. cpp:class:: template <typename T> ExprTraits

   Main traits class to find out information about an expression type.

   .. cpp:member:: static const bool isExpr

      True if the type is in fact an expression (or any XAD active variable)

   .. cpp:member:: static const int numVariables

      Number of variables that are port of the expression

   .. cpp:member:: static const bool isForward

      Boolean to represent if forward-mode AD

   .. cpp:member:: static const bool isReverse

      Boolean to represent if adjoint mode AD

   .. cpp:member:: static const bool isLiteral

      True if type is an elementary AD type (e.g. :cpp:class:`AReal`)

   .. cpp:type:: nested_type

      The underlying double type of the expression, e.g. ``double`` for :cpp:class:`AReal\<double>`,
      or :cpp:class:`AReal\<FReal\<double>>` (unwrapping all layers)

   .. cpp:type:: value_type

      The underlying active type of the expression, e.g. :cpp:class:`AReal\<double>`
      for a complex expression involving reverse mode active variables.

   .. cpp:type:: scalar_type

      The scalar type of the expression, unwrapping one layer in case of higher-order
      derivatives. For example, returns ``double`` for :cpp:class:`AReal\<double>`,
      and ``FReal<double>`` for :cpp:class:`AReal\<FReal\<double>>` .
.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-global:

Global Functions
================

.. highlight:: cpp

This section lists functions that are specific to the active data types
and tape management. 
For mathematical function, see :ref:`ref-math`.


.. cpp:function:: value(T& x)

   Returns a reference (or const-reference if ``x`` is constant) to the value
   stored in x. 
   
   If ``x`` is a XAD active data type, 
   such as :cpp:class:`AReal` or :cpp:class:`FReal`,
   this function returns a reference to the stored value (which is assignable).
   
   If ``x`` is a passive data type, this function simple returns the value itself.
   
   This function is especially useful in generic code, as it is defined on any
   data type.
   
   :param x: An active or passive variable
   :return: Reference to the value stored in ``x``

.. cpp:function:: derivative(const T& x)

   Returns a reference (or const-reference if ``x`` is constant) to the derivative
   stored in x. 
   
   If ``x`` is a XAD active data type, 
   such as :cpp:class:`AReal` or :cpp:class:`FReal`,
   this function returns a reference to the stored derivative (which is assignable).
   
   If ``x`` is a passive data type, this function simply returns 0.
   
   This function is especially useful in generic code, as it is defined on any
   data type.
   
   :param x: An active or passive variable
   :return: Reference to the derivative associated with ``x``
   

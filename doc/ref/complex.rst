.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-complex:

Complex
=======

.. highlight:: cpp

Overview
--------

XAD implements specialisations of :cpp:expr:`std::complex` for the XAD active data types
:cpp:class:`AReal` and :cpp:class:`FReal`.
They are are provided in the header ``XAD/Complex.hpp``,
along with all the mathematical operations defined in the standard.

Note that the complex header is not automatically included with ``XAD/XAD.hpp``.
Users must include it as needed.

.. cpp:class:: template <typename T> std::complex<AReal<T> >

    Specialisation for adjoint mode data type (in ``std`` namespace).

.. cpp:class:: template <typename T> std::complex<FReal<T> >

    Specialisation for forward mode data type (in ``std`` namespace).


Members
-------

.. cpp:namespace:: template <typename T> std::complex<AReal<T> >

All standard complex members are implemented.

Below are the non-standard additions and changes of the interface only,
using the placeholder :cpp:class:`AReal\<T>` as a placeholder inner type.
The same functions are also defined for :cpp:class:`FReal\<T>`.

.. cpp:function:: XReal<T>& real()

    This function returns a reference rather than a copy of the real part,
    to allow for easy access and adjusting of derivatives using :cpp:func:`derivative`.

.. cpp:function:: XReal<T>& imag()

    Returns a reference rather than a copy.

.. cpp:function:: const XReal<T>& real() const

    Returns a reference rather than a copy.

.. cpp:function:: const XReal<T>& imag() const

    Returns a reference rather than a copy.

.. cpp:function:: void setDerivative(const T& real_derivative, const T& imag_derivative = T())

    Sets the derivatives (either :math:`\dot{x}` or :math:`\bar{x}`) for both the real 
    and imaginary parts.

.. cpp:function:: void setAdjoint(const T& real_derivative, const T& imag_derivative = T())

    Alias for :cpp:func:`setDerivative`

.. cpp:function:: std::complex<T> getDerivative() const

    Gets the derivatives (either :math:`\dot{x}` or :math:`\bar{x}`) for both the real 
    and imaginary parts, represented as a complex of the underlying (double) type.

.. cpp:function:: std::complex<T> getAdjoint() const

    Alias for :cpp:func:`getDerivative`


Non-Member Functions
--------------------

.. cpp:namespace:: 0

.. cpp:function:: template <typename T> \
    std::complex<T> derivative(const std::complex<AReal<T> >& z)

    Returns the adjoints of the ``z`` variable, represented 
    as a complex number of the underlying double type.

    Note that since the return type is not a reference, setting
    derivatives should be done by using the member function 
    :cpp:func:`template <typename T> std::complex<AReal<T> >::setDerivative`
    or using the ``real`` and ``imag`` member functions instead.

.. cpp:function:: template <typename T> \
    std::complex<T> derivative(const std::complex<FReal<T> >& z)

    Returns the derivatives of the ``z`` variable, represented 
    as a complex number of the underlying double type.

    Note that since the return type is not a reference, setting
    derivatives should be done by using the member function 
    :cpp:func:`template <typename T> std::complex<FReal<T> >::setDerivative`
    or using the ``real`` and ``imag`` member functions instead.

.. cpp:function:: template <typename T> \
    std::complex<T> value(const std::complex<AReal<T> >& z)

    Returns the value of the ``z`` variable (underlying double type), 
    represented as a complex number.

.. cpp:function:: template <typename T> \
    std::complex<T> value(const std::complex<FReal<T> >& z)

    Returns the value of the ``z`` variable (underlying double type), 
    represented as a complex number.

.. cpp:function:: template <typename T> \
    AReal<T>& real(std::complex<AReal<T> >& z)

    Access to the real part by reference.

.. cpp:function:: template <typename T> \
    FReal<T>& real(std::complex<FReal<T> >& z)

    Access to the real part by reference.

.. cpp:function:: template <typename T> \
    AReal<T>& imag(std::complex<AReal<T> >& z)

    Access to the imaginary part by reference.

.. cpp:function:: template <typename T> \
    FReal<T>& imag(std::complex<FReal<T> >& z)

    Access to the imaginary part by reference.


Math Operations
---------------

All arithmetic operators and mathematical functions in the C++11 standard
have been specialised with the XAD complex data types as well.
This also includes the stream read and write operations.

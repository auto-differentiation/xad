.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-math:

Mathematical Operations
=======================

.. highlight:: cpp

In the following, the data types ``T`` 
refer to arithmetic data types on which the operation is defined mathematically.
This includes the active XAD types as well as the standard passive types.

The functions listed here are defined in the ``xad`` namespace, 
and C++ ADL rules (argument-dependent lookup) typically find these functions
automatically if they are applied to XAD types.
However, for this the calls must be unqualified, i.e. without a namespace specifier.

Alternatively, fully qualified names work as usual (e.g. :cpp:expr:`xad::sin(x)`),
also for ``float`` and ``double``.

For convenience, if the header ``XAD/StdCompatibility.hpp`` is included,
the XAD variables are imported into the :cpp:expr:`std` namespace,
so that existing calls to :cpp:expr:`std::sin` and similar functions are working as expected.


Absolute Values, Max, Min, and Rounding
---------------------------------------


.. cpp:function:: T abs(T x)

   Computes the absolute value of ``x``. 

   Note that for defined second-order derivatives, this computes ``(x>0)-(x<0)``
   
.. cpp:function:: T max(T x, T y)

   Returns the maximum of ``x`` and ``y``.
   
   Note that for well-defined second order derivative, this is implemented as
   ``(x + y + abs(x-y)) / 2``

.. cpp:function:: T fmax(T x, T y)

   Synonym for :cpp:func:`max`

.. cpp:function:: T min(T x, T y)

   Returns the minimum of ``x`` and ``y``.
   
   Note that for well-defined second order derivative, this is implemented as
   ``(x + y - abs(x-y)) / 2``

.. cpp:function:: T fmin(T x, T y)

   Synonym for :cpp:func:`min`

.. cpp:function:: T floor(T x)

   Rounds towards negative infinity
   
.. cpp:function:: T ceil(T x)

   Rounds towards positive infinity
   
.. cpp:function:: T trunc(T x)

   Rounds towards 0

.. cpp:function:: T round(T x)

   Round to the nearest integer value

.. cpp:function:: long lround(T x)

   Like :cpp:func:`round`, but converting the result to a ``long`` type.

.. cpp:function:: long long llround(T x)

   Like :cpp:func:`round`, but converting the result to a ``long long`` type.
   
.. cpp:function:: T fmod(T x, T y)

   The floating-point remainder of the division operation ``x/y``, i.e.
   exactly the value ``x - n*y``, where ``n`` is ``x/y`` with its fractional 
   part truncated.
   
.. cpp:function:: T remainder(T x, T y)

   The IEEE floating-point remainder of the division operation ``x/y``, i.e.
   exactly the value ``x - n*y``, where the value ``n`` is the integral value 
   nearest the exact value ``x/y``. 
   When ``abs(n-x/y) = 0.5``, the value n is chosen to be even.

   In contrast to :cpp:func:`fmod`, 
   the returned value is not guaranteed to have the same sign as ``x``.
   
.. cpp:function:: T remquo(T x, T y, int* n)

   Same as :cpp:func:`remainer`, but returns the integer factor `n` in
   addition.

.. cpp:function:: T modf(T x, T* iptr)

   Decomposes ``x`` into integral and fractional parts, each with the same
   type and sign as ``x``. The integral part is stored in ``iptr``.


.. cpp:function:: T nextafter(T from, T to)

   Returns the next representable value of ``from`` in the direction of ``to``.

   Mathmatically, the difference of ``from`` to the return value is very small.
   For derivatives, we therefore consider them both the same and calculate
   derivative accordingly.

.. cpp:function:: T copysign(T x, T y)

   Copies the sign of the floating point value ``y`` to the value ``x``, correctly
   treating positive/negative zero, NaN, and Inf values. It uses :cpp::func:`signbit`
   internally to determine the sign of ``y``.


Trigonometric Functions
-----------------------

.. cpp:function:: T degrees(T x)

   Converts the given value in radians to degrees
   
.. cpp:function:: T radians(T x)

   Converts the given value in degrees to radians
   
.. cpp:function:: T cos(T x)

   Computes the cosine of ``x``
   
.. cpp:function:: T sin(T x)

   Computes the sine of ``x``
   
.. cpp:function:: T tan(T x)

   Computes the tangent of ``x``

.. cpp:function:: T asin(T x)

   Computes the inverse sine of ``x``
   
.. cpp:function:: T acos(T x)

   Computes the inverse cosine of ``x``
   
.. cpp:function:: T atan(T x)

   Computes the inverse tangent of ``x``

.. cpp:function:: T atan2(T x, T y)

   Computes the four-quadrant inverse tangent of a point located at ``(x, y)``.

.. cpp:function:: T sinh(T x)
   
   Computes the hyperbolic sine of ``x``
   
.. cpp:function:: T cosh(T x)

   Computes the hyperbolic cosine of ``x``
   
.. cpp:function:: T tanh(T x)

   Computes the tangent of ``x``

.. cpp:function:: T asinh(T x)

   Computes the inverse hyperbolic sine of ``x``
   
.. cpp:function:: T acosh(T x)

   Computes the inverse hyperbolic cosine of ``x``

.. cpp:function:: T atanh(T x)

   Computes the inverse hyperbolic tangent of ``x``
   
         

Powers, Exponentials, and Logarithms
------------------------------------

.. cpp:function:: T log(T x)

   Computes the natural logarithm of ``x``
   
.. cpp:function:: T log10(T x)

   Computes the base 10 logarithm of ``x``  
   
.. cpp:function:: T log2(T x)

   Computes the base 2 logarithm of ``x`` 
   
.. cpp:function:: T exp(T x)

   Computes the exponential of ``x`` (base e)
   
.. cpp:function:: T expm1(T x)

   Computes ``exp(x) - 1`` with higher precision around 0
   
.. cpp:function:: T exp2(T x)

   Computes 2 to the power of ``x``
   
.. cpp:function:: T log1p(T x)

   Computes ``log(1 + x)`` with higher precision around 0
   
.. cpp:function:: T sqrt(T x)

   Computes the square root of ``x``
   
.. cpp:function:: T cbrt(T x)

   Computes the cubic root of ``x``
   
.. cpp:function:: T pow(T x, T y)

   Computes ``x`` to the power of ``y``

.. cpp:function:: T ldexp(T x, int exp)

   Multiplies ``x`` by two to the power of ``exp``

.. cpp:function:: T frexp(T arg, int* exp)

   Decomposes the given floating point value arg into a normalised fraction
   and an integral power of two. 
   
.. cpp:function:: int ilogb(T arg)

   Returns the integral part of the logarithm of ``abs(x)``, 
   using ``FLT_RADIX`` as base for the log.

.. cpp:function:: T scalbn(T arg, int exp)

   Calculates ``arg * pow(FLT_RADIX, exp)``.


Error Functions
---------------

.. cpp:function:: T erf(T x)

   Computes the error function of ``x``, if provided by the compiler's math
   library.
   
.. cpp:function:: T erfc(T x)
   
   Computes the complementary error function of ``x``, if provided by the compiler's math
   library.
   

Floating Point Classification
-----------------------------

.. cpp:function:: bool isinf(T x)

   Check if value is infinity (positive or negative)

.. cpp:function:: bool isnan(T x)

   Check if value is NaN

.. cpp:function:: bool isfinite(T x)

   Check if value is finite (not infinite and not NaN)

.. cpp:function:: bool signbit(T x)

   Returns true if ``x`` is negative and false otherwise. Also detects sign bit of zeros.

.. cpp:function:: bool isnormal(T x)

   Checks if the value is a normal floating point number, i.e.
   not zero, subnormal, infinite, or NaN.
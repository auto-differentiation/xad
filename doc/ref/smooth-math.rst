.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-smooth-math:

Smoothed Mathematical Functions
===============================

.. highlight:: cpp

The functions in this section are smoothed equivalents of the original
math functions that can be used to allow computing derivatives around discontinuities.

.. cpp:function:: T smooth_abs(T x, T c = 0.001)

   Smoothed version of :cpp:func:`abs`, defined as:
   
   .. math::
      \text{smooth\_abs}(x,c) = \left\{\begin{array}{lll}
      |x| & & \text{if }x < c\text{ or } x > c\\[5pt]
      x^2\left(\frac{2}{c}-\frac{1}{c^2} x\right) & & \text{if }0 \leq x \leq c\\[5pt]
      x^2\left(\frac{2}{c}+\frac{1}{c^2} x\right) & & \text{if }-c \leq x < 0
      \end{array}\right.
   
   :param x: The input value
   :param c: Cut-off point for the spline-approximated area (default: ``0.001``)
   :return: The smoothed absolute value, defined as above
      
.. cpp:function:: T smooth_max(T x, T y, T c = 0.001)

   Smoothed version of the :cpp:func:`max`, defined as:
   
   .. math::
      \text{smooth\_max}(x,y,c) = 0.5\left(x+y+\text{smooth\_abs}(x-y,c)\right) 
      
   :param x: First argument to max
   :param y: Second argument to max
   :param c: Cut-off point for the spline-approximated area (default: ``0.001``)
   :return: The smoothed max function, defined as above
   
.. cpp:function:: T smooth_min(T x, T y, T c = 0.001)

   Smoothed version of the :cpp:func:`min`, defined as:
   
   .. math::
      \text{smooth\_min}(x,y,c) = 0.5\left(x+y-\text{smooth\_abs}(x-y,c)\right) 
      
   :param x: First argument to min
   :param y: Second argument to min
   :param c: Cut-off point for the spline-approximated area (default: ``0.001``)
   :return: The smoothed min function, defined as above
   
.. leave this for now
   .. cpp:function:: T smooth_step(T x, T c = 0.001)
   
      Smoothed version of a step function. 
      The step function is ``0`` for ``x < 0`` and ``1`` for ``x > 0``. 
   
   
.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _higher-order:

Higher-Order Derivatives
========================

.. highlight:: cpp

As explained in :ref:`aad-higher`, higher order derivatives can be computed
by nesting first order algorithmic differentiation techniques.
For example, one can obtain second order by computing forward mode over adjoint mode.
With XAD, 
this technique can be used directly to compute higher order derivatives.

XAD's automatic differentiation interface structures (see :ref:`ref-interface`)
define second order mode data types for easy access.
Types for third or higher orders need to defined manually
from the basic first-order types.

We will demonstrate second-order derivatives using forward-over-adjoint mode
in the following.

Example Algorithm
-----------------

For demonstration purposes, we use the same algorithm from :ref:`tutorial`::

   template <class T>
   T f(T x0, T x1, T x2, T x3)
   {
     T a = sin(x0) * cos(x1);
     T b = x2 * x3 - tan(x1 - x2);
     T c = a + 2* b;
     return c*c;
   }
   
We are interested in derivatives at the point::

   double x0 = 1.0;
   double x1 = 1.5;
   double x2 = 1.3;
   double x3 = 1.2;
   
Forward Over Adjoint
--------------------

In this mode, we can compute all first-order derivatives (as a single output
function derived with adjoints gives all first order derivatives),
and the first row of the Hessian matrix of second order derivatives.
The full Hessian is defined as:

.. math::
   \Large
   \bm H = \left[ \begin{array}{cccc}
     \frac{\partial^2 f}{\partial x_0^2} & 
     \frac{\partial^2 f}{\partial x_0 \partial x_1} &
     \frac{\partial^2 f}{\partial x_0 \partial x_2} &
     \frac{\partial^2 f}{\partial x_0 \partial x_3} \\[6pt]
     \frac{\partial^2 f}{\partial x_1 \partial x_0} & 
     \frac{\partial^2 f}{\partial x_1^2} &
     \frac{\partial^2 f}{\partial x_1 \partial x_2} &
     \frac{\partial^2 f}{\partial x_1 \partial x_3} \\[6pt]
     \frac{\partial^2 f}{\partial x_2 \partial x_0} & 
     \frac{\partial^2 f}{\partial x_2 \partial x_1} &
     \frac{\partial^2 f}{\partial x_2^2} &
     \frac{\partial^2 f}{\partial x_2 \partial x_3} \\[6pt]
     \frac{\partial^2 f}{\partial x_3 \partial x_0} & 
     \frac{\partial^2 f}{\partial x_3 \partial x_1} &
     \frac{\partial^2 f}{\partial x_3 \partial x_2} &
     \frac{\partial^2 f}{\partial x_3^2} 
   \end{array}\right]

Note that the Hessian matrix is typically symmetric, 
which can be used to reduce the amount of computation needed for the full Hessian.

The first step is to set up the tape and active data types needed for this computation::

   typedef xad::fwd_adj<double> mode;
   typedef mode::tape_type tape_type;
   typedef mode::active_type AD;
   
   tape_type tape;

Note that the active type for this mode is actually ``AReal<FReal<double> >``.

Now we need to setup the independent variables and register them::

   AD x0_ad = x0;
   AD x1_ad = x1;
   AD x2_ad = x2;
   AD x3_ad = x3;

   tape.registerInput(x0_ad);
   tape.registerInput(x1_ad);
   tape.registerInput(x2_ad);
   tape.registerInput(x3_ad);

As we compute the second order using forward mode, 
we need to seed the initial derivative for the second order before running the algorithm::

   derivative(value(x0_ad)) = 1.0;
   
The inner call to :cpp:func:`value` takes the value of the outer type, 
i.e. it returns the value as the type ``FReal<double>``,
of which we set the derivative to ``1``.

Now we can start recording derivatives on the tape and run the algorithm::

   tape.newRecording();
   
   AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
   
For the inner adjoint mode, we need to register the output and seed the initial adjoint with 1::

   tape.registerOutput(y);
   value(derivative(y)) = 1.0;
   
Here, the inner call to :cpp:func:`derivative` gives the derivative of the outer 
type, i.e. the derivative of the adjoint-mode active type.
This is of type ``FReal<double>``, for which we set the value to ``1``.

Next we compute the adjoints, which computes both the first and second order 
derivatives::

   tape.computeAdjoints();
   
We can now output the result::

   std::cout << "y = " << value(value(y)) << "\n";
   
And the first order derivatives::

   std::cout << "dy/dx0 = " << value(derivative(x0_ad)) << "\n"
             << "dy/dx1 = " << value(derivative(x1_ad)) << "\n"
             << "dy/dx2 = " << value(derivative(x2_ad)) << "\n"
             << "dy/dx3 = " << value(derivative(x3_ad)) << "\n";

Note again that the inner call to :cpp:func:`derivative` obtains the derivative
of the outer active data type,
hence it gives a ``FReal<double>`` reference that represents the first order adjoint value.
We can get this value as a ``double`` using the :cpp:func:`value` call.

The second order derivatives w.r.t. ``x0`` can be obtained as::

   std::cout << "d2y/dx0dx0 = " << derivative(derivative(x0_ad)) << "\n"
             << "d2y/dx0dx1 = " << derivative(derivative(x1_ad)) << "\n"
             << "d2y/dx0dx2 = " << derivative(derivative(x2_ad)) << "\n"
             << "d2y/dx0dx3 = " << derivative(derivative(x3_ad)) << "\n";

which 'unwraps' the derivatives of the first and second order active types.

.. highlight:: text

The result of the running the application for the given inputs is::

   y      = 7.69565
   dy/dx0 = 0.21205
   dy/dx1 = -16.2093
   dy/dx2 = 24.8681
   dy/dx3 = 14.4253
   d2y/dx0dx0 = -0.327326
   d2y/dx0dx1 = -3.21352
   d2y/dx0dx2 = 0.342613
   d2y/dx0dx3 = 0.198741

Forward over adjoint is the recommended mode for second-order derivatives.

.. seealso::  This example is included with XAD (`fwd_adj_2nd <https://github.com/xcelerit/XAD/tree/main/samples/fwd_adj_2nd>`_).
   
Other Second-Order Modes
------------------------

.. highlight:: cpp

Other second-order modes work in a similar fashion.
They are briefly described in the following.

Forward Over Forward
^^^^^^^^^^^^^^^^^^^^

With forward-over-forward mode, 
there is no tape needed and the derivatives of both orders need to be seeded
before running the algorithm. 
One element of the Hessian and one first-order derivative can be computed 
with this method, if the function has one output.
The derivative initialization sequence in this mode is typically::

   value(derivative(x)) = 1.0;   // initialize the first-order derivative
   derivative(value(x)) = 1.0;   // initialize the second-order derivative
   
   
After the computation, the first order derivative can be retrieved as::

   std::cout << "dy/dx = " << derivative(value(y)) << "\n";

And the second order derivative as::

   std::cout << "d2y/dxdx = " << derivative(derivative(y)) << "\n";
   
With different initial seeding, different elements of the Hessian can be obtained.

Adjoint Over Forward
^^^^^^^^^^^^^^^^^^^^

Here the inner mode is forward, 
computing one derivative in a tape-less fashion,
and the outer mode is adjoint, requiring a tape.
With this mode, we need to initialize the forward-mode derivative with::

   value(derivative(x)) = 1.0;   // initialize the first-order derivative
   
As the derivative of the output corresponds to the first order result, 
we need to seed its derivative (i.e. the adjoint) after running the algorithm::

   derivative(derivative(y)) = 1.0;

After tape interpretation, we can now obtain the first-order derivative as::

   std::cout << "dy/dx = " << value(derivative(y)) << "\n";
   
Due to the symmetries in this mode of operation, the same first-order derivatives
can also be obtained as::

   std::cout << "dy/dx = " << derivative(derivative(x)) << "\n";
   
Which allows to get all first-order derivatives w.r.t. to all inputs in this mode,
similar to the forward-over-adjoint mode.
   
The second-order derivatives can be obtained as::

   std::cout << "d2y/dxdx = " << derivative(value(x))

Adjoint Over Adjoint
^^^^^^^^^^^^^^^^^^^^

As both nested modes are adjoint, 
this mode needs to two tapes for both orders.
Hence the types defined in the interface structure :cpp:class:`adj_adj` 
need an inner and an outer tape type::

   typedef xad::adj_adj<double> mode;
   typedef mode::inner_tape_type inner_tape_type;
   typedef mode::outer_tape_type outer_tape_type;
   typedef mode::active_type AD;
   
In this mode, no initial derivatives need to be set, 
but it is important that both tapes are initialized and a new recording is
started on both before running the algorithm.

After the execution, the outer derivative needs to be seeded as::

   value(derivative(y)) = 1.0;
   
And then the outer tape needs to compute the adjoints. 
This computes the ``value(derivative(x))`` as an output, 
and the derivative of this needs to be set before interpreting the inner tape::

   derivative(derivative(x)) = 1.0;
   
After calling ``computeAdjoints()`` on the inner tape, 
we can read the first-order derivatives as::

   std::cout << "dy/dx = " << value(derivative(x)) << "\n;
   
And the second-order derivatives as::

   std::cout << "d2y/dxdx" << derivative(value(x)) << "\n";
   

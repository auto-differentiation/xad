.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 

.. meta::
   :description: Basic usage of the XAD C++ automatic differentiation library, forward and adjoint modes.

.. _tutorial:

Basic Usage
===========

.. highlight:: cpp

In this section, we will illustrate how to use XAD to compute 
first order derivatives in both forward and adjoint mode. 

As an example, we choose a simple function with 4 inputs and 1 output variable,
defined as::

   double f(double x0, double x1, double x2, double x3)
   {
     double a = sin(x0) * cos(x1);
     double b = x2 * x3 - tan(x1 - x2);
     double c = a + 2* c;
     return c*c;
   }

We will compute derivatives of this function at the point::
  
   double x0 = 1.0;
   double x1 = 1.5;
   double x2 = 1.3;
   double x3 = 1.2;


Prerequisite: Replace Active Variables
--------------------------------------
  
In order to use XAD to differentiate this function, 
we first must replace all independent data types and all values that
depend on them with an active data type provided by XAD.
In the above function, 
all variables depend on the inputs and thus 
all occurrences of ``double`` must be replaced.

This can be done in one of two ways:

1. The variables can be replaced directly, given the desired mode of differentiation.
   For example, for forward mode ``double`` is replaced by the type 
   :cpp:class:`FReal` and for adjoint mode the type :cpp:class:`AReal`.
2. The function is made a template, so that it can be called with any data type,
   including the original ``double``.
   
We choose the second approach for this tutorial, thus the function becomes::

   template <class T>
   T f(T x0, T x1, T x2, T x3)
   {
     T a = sin(x0) * cos(x1);
     T b = x2 * x3 - tan(x1 - x2);
     T c = a + 2* b;
     return c*c;
   }
   
This means we can use the same definition with both forward and adjoint modes.

.. _tutorial-fwd:

Forward Mode
------------

As illustrated in :ref:`aad-fwd`, 
when applied to a function with a single output,
the forward mode of algorithmic differentiation can compute *one* derivative
at a time.
For illustration, we choose to derive the function with respect to the input
variable ``x0``.
  
To initiate the forward mode, we must first declare active variables with
the appropriate type.
XAD provides convenience typedefs to select the mode of differentiation,
illustrated in detail in :ref:`ref-interface`. 
For forward mode, we can declare the types needed as::

   typedef xad::fwd<double> mode;
   typedef mode::active_type AD;
   
We can then use the ``AD`` typedef for our variables.

The next step is to initialize the dependent variables, 
which is simply done by assigning the input values to new variables of type ``AD``::

   AD x0_ad = x0;
   AD x1_ad = x1;
   AD x2_ad = x2;
   AD x3_ad = x3;
   
For forward mode, we must now seed the initial derivative for the variable
we are interested in with the value 1 (as described in :ref:`aad-fwd`), as::

   derivative(x0_ad) = 1.0;

The global function :cpp:func:`derivative` is a convenience function that works
on any active data type. 
Alternatively, we could have used the member function :cpp:func:`FReal::setDerivative`.

At this point we are ready to call our function and it will compute the function
value as well as the derivative we are interested in::

   AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
   
We can now access the results using the :cpp:func:`value` and :cpp:func:`derivative`
functions on the output (or the member functions :cpp:func:`FReal::getDerivative`
and :cpp:func:`FReal::getValue`).
For example, the following code outputs them to the console::

   std::cout << "y = " << value(y) << "\n"
             << "dy/dx0 = " << derivative(y) << "\n";

.. seealso:: This example is included with XAD (`fwd_1st <https://github.com/xcelerit/XAD/tree/main/samples/fwd_1st>`_).

.. _tutorial-adj:

Adjoint Mode
------------

The adjoint mode of automatic differentiation
is the natural choice for the function at hand, 
as it has a single output and multiple inputs.
We can get all four derivatives in one execution.

Adjoint mode needs a tape to record the operations and their values
during the valuation.
After setting the adjoints of the outputs, 
this tape can then be rolled back to compute the adjoints of the inputs.

Both the active data type and the tape type can be obtained from the 
interface structure :cpp:class:`adj` (see :ref:`ref-interface` for details)::

   typedef xad::adj<double> mode;
   typedef mode::tape_type tape_type;
   typedef mode::active_type AD;
   
The first step for computing adjoints is to initialise the tape::

   tape_type tape;
   
This calls the default constructor :cpp:func:`Tape::Tape`, 
which creates the tape and activates it.
Before creating active variables (type ``AD``), 
the current threads needs to have an active tape.
To ensure thread-safety, 
every thread of the application can have its own active tape.

Next, we create the input variables and register them with the tape::

   AD x0_ad = x0;
   AD x1_ad = x1;
   AD x2_ad = x2;
   AD x3_ad = x3;
   tape.registerInput(x0);
   tape.registerInput(x1);
   tape.registerInput(x2);
   tape.registerInput(x3);

Note that only variables registered as inputs with the tape and all variables dependent on them
are recorded.

Once the independent variables are set, we can start recording derivatives
on tape and run the algorithm::

   tape.newRecording();
   
   AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
   
At this stage, we have all operations recorded and have the value computed.
We now need to register the outputs with the tape as well, 
before we can seed the initial adjoint of the output wit 1 
as explained in :ref:`add-adj`::

   tape.registerOutput(y);
   derivative(y) = 1.0;
   
This uses the free function :cpp:func:`derivative`, 
which returns a reference to the stored derivative (or adjoint) 
of the given parameter. 
Alternatively the member functions :cpp:func:`AReal::setAdjoint` or
:cpp:func:`AReal::setDerivative` can be used for the same purpose.

What is left is interpreting the tape to compute the adjoints of the 
independent variables::

   tape.computeAdjoints();
   
We can now access the adjoints of the inputs,
which are the derivatives we are interested in,
via the global :cpp:func:`derivative` function or the member function
:cpp:func:`AReal::getDerivative`::

   std::cout << "y     = " << value(y) << "\n"
             << "dy/dx0 = " << derivative(x0_ad) << "\n"
             << "dy/dx1 = " << derivative(x1_ad) << "\n"
             << "dy/dx2 = " << derivative(x2_ad) << "\n"
             << "dy/dx3 = " << derivative(x3_ad) << "\n";

.. seealso:: This example is included with XAD (`adj_1st <https://github.com/xcelerit/XAD/tree/main/samples/adj_1st>`_).

Best Practices
--------------

When the algorithm to be evaluated has less outputs than inputs,
adjoint mode should be preferred. 
However, when only a small number of derivatives are needed (e.g. less than 5), 
the memory for the tape can be avoided by using forward mode.
Experimentation is advised to find the optimal mode for the given algorithm.


.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions.

.. meta::
   :description: How to handle external functions in the XAD C++ automatic differentiation library.
    
.. _external:

External Functions
==================

.. highlight:: cpp

Often parts of the algorithm to be differentiated are not available 
in source code.
For example, a routine from an external math library may be called. 
Reimplementing it may not be desirable (for performance or development effort reasons),
in which case the derivatives of this function need to be implemented 
manually in some form. 

This can be achieved by either: 

* Applying finite differences to the library function (bumping),
* Implementing the adjoint code of the function by hand, or
* Computing the derivatives analytically, possibly using other library functions.

In these cases, 
the *external function interface* of XAD can be used to integrate 
the manual derivatives, which is described below.
With the same technique, 
performance- or memory-critical parts of the application may be hand-tuned. 


Example Algorithm
-----------------

We pick an example algorithm which computes the length of a multi dimensional
vector. 
This is defined as:

.. math::
   y = \sqrt{\sum_0^{N-1} x_i^2}
   
The goal is to compute the derivatives of ``y`` with respect to all 
input vector elements using adjoint mode.

The algorithm can be implemented in C++ code as::

   std::vector<double> xsqr(n);
   for (int i = 0; i < n; ++i)
     xsqr[i] = x[i] * x[i];
   double y = sqrt(sum_elements(x, n));
  
For this example, we assume that the ``sum_elements`` is an external function
implemented in a library that we do not have source code of.
It has the prototype::

   double sum_elements(const double* x, int n);
    

External Function For Adjoint Mode
----------------------------------

To use the external function, we follow this procedure:

1. At the point of the call, convert the values of the input active variables
   to the underlying plain data type (``double``)
2. Call the external function passively
3. Assign the result values to active output variables so the tape recording
   can continue
4. Store the tape slots of the inputs and outputs with a checkpoint callback object
   and register it with the tape. 
5. When computing adjoints, this callback needs to load the adjoint of the outputs, 
   propagate to them to the inputs manually, 
   and increment the input adjoints by these values.

We put all the functionality into a callback object.
We derive from the :cpp:class:`CheckpointCallback` base class
and implement at least the virtual method :cpp:func:`CheckpointCallback::computeAdjoint`.
This method gets called during tape rollback. 
We also place the forward computation within the same object 
(this could also be done outside of the callback class).
The declaration of our callback class looks like this::

   template <class Tape>
   class ExternalSumElementsCallback : public xad::CheckpointCallback<Tape>
   {
   public:
     typedef typename Tape::slot_type   slot_type;    // type for slot in the tape
     typedef typename Tape::value_type  value_type;   // double
     typedef typename Tape::active_type active_type;  // AReal<double>

     active_type computeExternal(const active_type* x, int n);  // forward computation
     void computeAdjoint(Tape* tape) override;                  // adjoint computation

   private:
     std::vector<slot_type> inputSlots_;              // slots of the inputs in tape
     slot_type outputSlot_;                           // slot of the output in tape
   };
   
We declare it as a template for arbitrary tape types,
which is good practice as it allows to reuse 
this implementation with higher order derivatives too.

``computeExternal`` Method
^^^^^^^^^^^^^^^^^^^^^^^^^^
   
Within the ``computeExternal`` method, 
we first store the slots in the tape for the input variables,
as we will need them during adjoint computation to increment the corresponding
adjoints. 
We use the ``inputSlots_`` member vector to keep this information::

   for (int i = 0; i < n; ++i)
     inputSlots_.push_back(x[i].getSlot());
     
Then we create a copy of the active inputs and store them in a vector of passive
values,
with which we can call the external function::

   std::vector<value_type> x_p(n);
   for (int i = 0; i < n; ++i)
     x_p[i] = value(x[i]);
     
   value_type y = sum_elements(&x_p[0], n);
   
We now need to store this result in an active variable,
register it as an output of the external function
(to allow the tape to continue recording dependent variables),
and keep its slot in the tape for the later adjoint computation::

   active_type ret = y;
   Tape::getActive()->registerOutput(ret);
   outputSlot_ = ret.getSlot();
   
Finally we need to insert the callback into the tape, 
hence requesting it to be called during adjoint rollback of the tape,
and return::

   Tape::getActive()->insertCallback(this);
   return ret;
   

``computeAdjoint`` Method
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``computeAdjoint`` method gets called by XAD during tape rollback. 
We need to override this method and implement the manual adjoint code.
For a simple sum operation, this is straightforward: 
all input adjoints are equal to the output adjoint since all 
partial derivatives are 1.
Thus we need to obtain the output adjoint and increment all input adjoints by 
this value::

   value_type output_adj = tape->getAndResetOutputAdjoint(outputSlot_);
   for (int i = 0; i < inputSlots_.size(); ++i)
     tape->incrementAdjoint(inputSlots_[i], output_adj); 

The function :cpp:func:`Tape::getAndResetOutputAdjoint` obtains the
adjoint value corresponding to the given slot and resets it to zero.
This reset is necessary in general as the output variable may
have been overwriting other values in the forward computation.
The :cpp:func:`Tape::incrementAdjoint` function simply 
increments the adjoint with the given slot by the given value.

Wrapper Function
^^^^^^^^^^^^^^^^

With the checkpointing callback class in place, 
we can implement a ``sum_elements`` overload for :cpp:class:`AReal` that 
wraps the creation of this callback::

   template <class T>
   xad::AReal<T> sum_elements(const xad::AReal<T>* x, int n)
   {
     typedef typename xad::AReal<T>::tape_type tape_type;
     tape_type* tape = tape_type::getActive();
     ExternalSumElementsCallback<tape_type>* ckp = new ExternalSumElementsCallback<tape_type>;
     tape->pushCallback(ckp);
   
     return ckp->computeExternal(x, n);
   }

This function dynamically allocates the checkpoint callback object
and lets the tape manage its destruction through the :cpp:func:`Tape::pushCallback`
function.
This call simply ensures that the callback object is destroyed 
when the tape is destroyed,
making sure that no memory is leaked.
If the callback object was managed elsewhere, this call would not be necessary.
It then redirects the computation to the ``computeExternal`` function
of the checkpoint callback class.
Using this wrapper class, the ``sum_elements`` function can be used for active types
in the same fashion as the original external function ``sum_elements`` for ``double``.
Defining it as a template allows us to re-use this function for higher-order derivatives, 
should we need them in future.

Call-Site
^^^^^^^^^

The call site then can be implemented as 
(assuming that ``x_ad`` is the vector holding the independent variables, already registered on tape)::

   tape.newRecording();
   
   std::vector<AD> xsqr(n);
   for (int i = 0; i < n; ++i)
     xsqr[i] = x_ad[i] * x_ad[i];
   AD y = sqrt(sum_elements(xsqr.data(), n)); // calls external function wrapper
   
   tape.registerOutput(y);
   derivative(y) = 1.0;
   tape.computeAdjoints();
   
   std::cout << "y = " << value(y) << "\n";
   for (int i = 0; i < n; ++i)
     std::cout << "dy/dx" << i << " = " << derivative(x[i]) << "\n";


This follows exactly the same procedure as given in :ref:`tutorial-adj`.

.. seealso:: This example is included with XAD (`external_function <https://github.com/xcelerit/XAD/tree/main/samples/external_function>`_).

External Function For Forward Mode
----------------------------------

Since forward mode involves no tape,
a manual implementation of the derivative computation needs to be implemented
together with computing the value.
The manual derivatives can be updated directly in the output values
using the :cpp:func:`derivative` function.

In our example, we can implement the external function in forward mode as::

   template <class T>
   xad::FReal<T> sum_elements(const xad::FReal<T>* x, int n)
   {
     typedef xad::FReal<T> active_type;
     
     std::vector<T> x_p(n);
     for (int i = 0; i < n; ++i)
       x_p[i] = value(x[i]);
     
     T y_p = sum_elements(&x_p[0], n);
     
     active_type y = y_p;
     
     for (int i = 0; i < n; ++i)
       derivative(y) += derivative(x[i]);

     return y;
   }

We first extract the passive values from the ``x`` vector and call the
external library function to get the passive output value ``y_p``.
This value is then assigned to the active output variable ``y``,
which also initializes its derivative to ``0``.

As we have a simple sum in this example, 
the derivative of the output
is a sum of the derivatives of the inputs, 
which is computed by the loop in the end.

.. seealso:: This example is included with XAD (``external_function``).
   
 


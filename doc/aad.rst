.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _aad:

Algorithmic Differentiation Background
======================================

As every computer program is made up of a series of simple arithmetic operations, 
i.e.

.. math::
   a \rightarrow b \rightarrow \ldots \rightarrow u \rightarrow v \rightarrow \ldots \rightarrow z

where the inputs :math:`a` are modified in stages 
in order to get the final output :math:`z`. 
When the individual derivatives of each operation are known, 
the final derivative can be computed by recursive application of the chain rule. 
This is method is called *Algorithmic Differentiation*,
with the modes *forward* (or tangent-linear), i.e., from inputs to outputs, 
and *adjoint* (or reverse) mode, i.e., from outputs to inputs. 

In this section, 
we introduce the underlying theory for computing derivatives of a computer program.
We start with a review of the traditional finite difference method,
often called *bumping*, 
before introducing forward and adjoint algorithmic differentiation.

.. _aad-bump:

Finite Differences
------------------

The traditional approach for computing these derivatives 
is by employing a finite difference approximation. 
That is, each of the input variables are *bumped* one by one 
and the change of the result is used to estimate the sensitivities:

.. math::

   \frac{\partial f(x,\bm y)}{\partial x} &= \lim_{h\rightarrow 0}\frac{f(x+h,\bm y) - f(x,\bm y)}{h}\\
   \frac{\partial f(x,\bm y)}{\partial x} &= \lim_{h\rightarrow 0} \frac{f(x+h,\bm y) - f(x-h,\bm y)}{2h}

where :math:`f(x, \bm y)` is the function of which we are interested in derivatives
with respect to the input parameter :math:`x`.
The vector-valued argument :math:`\bm y` denotes the remaining function parameters.
The first equation represents *forward finite differences* and requires two
evaluations of the function.
The second equation gives *central finite differences* 
with potentially higher accuracy
and requires two function evaluation for the derivative and another evaluation
for the function's value.
 
In practice, the value :math:`h` is chosen small enough to approximate the theoretical limit,
but large enough to cause a detectable change of the result beyond typical numerical error levels.
Clearly, this choice impacts the accuracy of the approximation.

Further, this method implies that the function needs to be evaluated 
once for the result and once for each derivative that we are interested in.
This results in a high overall computational complexity
as soon as more than a few derivatives are needed. 

Thus, the finite differences approach has accuracy and performance limitations.

.. _aad-fwd:

Forward Mode
------------

Theory
^^^^^^

The forward mode defines :math:`\dot{u}` 
as the derivative of :math:`u` with respect to :math:`a`, i.e.

.. math::
   \dot{u} = \frac{\partial u}{\partial a}

Applying the chain rule of differentiation 
and assuming that the intermediate variables are vectors, 
the elements of :math:`\dot{v}` can be calculated as

.. math::
   \dot{v}_{i} = \sum_j \frac{\partial v_{i}}{\partial u_{j}} \dot{u}_j

Applying this to each step in the chain of operations from inputs to outputs, 
the value of :math:`\dot{z}` can be calculated. 
This is the *forward mode* of algorithmic differentiation.

For a function :math:`f\,{:}\,\mathbb{R}^n\,{\rightarrow}\,\mathbb{R}^m`, 
one application of the forward mode 
gives the sensitivities for all :math:`m` outputs with respect to 
*one* input parameter. 
It needs to be re-evaluated :math:`n` times to obtain all sensitivities. 
The computational cost is constant in the number of output variables :math:`m` 
and linear in the number of input variables :math:`n`. 

Example
^^^^^^^

We illustrate the forward mode on the example function:

.. math::
   z = \sin x_1 + x_1 x_2

.. highlight:: cpp

Which can be implemented in a computer program as::

   a = sin(x1);
   b = x1 * x2;
   z = a + b;

We are interested of the derivative with respect to :math:`x1`
for the input values :math:`x_1 = \pi` and :math:`x_2 = 2`.
The following figure illustrates how the forward mode algorithm differentiation
is applied to this problem: 

.. image:: /images/forward_illustration.*
   :align: center
   :alt: Forward mode example


On the left we see the computational graph representing the equation,
and the table on the right illustrates the the steps performed.

In step 0, we initialize the input values and we seed the derivatives
of these inputs. 
As we are interested in the derivative w.r.t. :math:`x_1`, 
we set its derivative to 1 while setting the other to 0.

Next we compute :math:`a` by taking the sine function. 
The value of :math:`a` is zero, 
while :math:`\dot{a}` is computed by multiplying 
the partial derivative of the sine w.r.t. to :math:`x1`, 
i.e. the cosine, with :math:`\dot{x_1}`.
This gives a value of  -1.

In the next step, the value of :math:`b` is computed as usual,
and :math:`\dot{b}` is calculated similarly to :math:`\dot{a}`,
this time depending on both  :math:`\dot{x_1}` and :math:`\dot{x_2}`.
This results in a value of 2.

The final statement adds both :math:`a` and :math:`b`, 
which gives the result of :math:`2\pi`.
To calculate :math:`\dot{z}`, 
we see that the  :math:`\dot{a}` and :math:`\dot{b}` 
can simply be added 
since their partial derivatives are both 1. 
This gives a final derivative of 1.

Hence:

.. math::

  \left.\frac{\partial z}{\partial x_1}\right|_{(\pi,2)} = 1
  
which can be easily verified analytically.


   
.. _add-adj:
   
Adjoint Mode
------------

Theory
^^^^^^

The adjoint mode applies the chain rule backwards, 
from outputs to inputs. 
Using standard notation, we define

.. math::
   \bar{u}_i = \frac{\partial z}{\partial u_i}

where :math:`i` is the index in the vector :math:`\bm u`. 
Applying the chain rule yields

.. math::
   \frac{\partial z}{\partial u_i} = \sum_j \frac{\partial z}{\partial v_j} \frac{\partial v_j}{\partial u_i}

which leads to the *adjoint mode equation*

.. math::
   \bar{u}_{i}    =  \sum_j \frac{\partial v_j}{\partial u_i} \bar{v}_{j}

Seeding :math:`\bar{z} = 1`, 
the adjoint mode equation can be applied for each step, 
from output to input, 
to obtain :math:`\bar{\bm a}`, 
which is the derivative of the output :math:`z` 
with respect to each of the input variables :math:`\bm a`. 

For a function :math:`f\,{:}\,\mathbb{R}^n\,{\rightarrow}\,\mathbb{R}^m`, 
the adjoint mode gives the sensitivities of *one* output 
with respect to all :math:`n` input parameters. 
It needs to be re-evaluated :math:`m` times to obtain all sensitivities. 
The computational cost is constant in the number of input variables :math:`n` 
and linear in the number of output variables :math:`m`. 

Example
^^^^^^^

We illustrate the adjoint mode using the same example as above: 

.. math::
   z = \sin x_1 + x_1 x_2

.. highlight:: cpp

implemented::

   a = sin(x1);
   b = x1 * x2;
   z = a + b;

With adjoint mode, we can get both partial derivatives of the output
in a single execution,
for the input values :math:`x_1 = \pi` and :math:`x_2 = 2`.
This is illustrated in the figure below:

.. image:: images/adjoint_illustration.*
   :align: center
   :alt: Adjoint mode example


As the adjoint mode walks from outputs back to inputs,
we execute the full computation of the value as usual,
until we have an output for :math:`z` of :math:`2\pi`.

Then we seed the adjoint of :math:`z` to 1 in the final step,
and walk backwards to compute the adjoints of the inputs.

In step 2, we can compute the adjoint of :math:`b` by multiplying 
the adjoint of :math:`z` with the partial derivative of the equation
for :math:`z` w.r.t. :math:`b`, which is 1.

The same is performed in step 1 to compute the adjoint of :math:`a`,
which also yields 1.

The adjoint of :math:`x2` is then computed by multiplying the partial
derivative of :math:`b` w.r.t. :math:`x_2` with the adjoint of :math:`b`,
which gives the value :math:`\pi`. 

The same method is applied to compute the adjoint of :math:`x_1`,
giving the value 1.

Thus, the two derivatives we were interested in are:

.. math::

  \left.\frac{\partial z}{\partial x_1}\right|_{(\pi,2)} &= 1\\
  \left.\frac{\partial z}{\partial x_2}\right|_{(\pi,2)} &= \pi

Which can be easily verified analytically.


.. _aad-higher:

Higher Orders
-------------

Higher order derivatives can be obtained by nesting the principles described above.
For example, applying forward mode algorithmic differentiation over adjoint mode
gives second order derivatives.
This method can be extended to any order.




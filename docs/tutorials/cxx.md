---
title: "Tutorial: Computing Derivatives Simplified"
description: "Learn to use XAD for first and higher-order derivatives with our guide on forward and adjoint modes, external functions and checkpointing."
---

## Basic Usage

In this section, we will illustrate how to use XAD to compute
first order derivatives in both forward and adjoint mode.

As an example, we choose a simple function with 4 inputs and 1 output variable,
defined as:

```c++
double f(double x0, double x1, double x2, double x3)
{
    double a = sin(x0) * cos(x1);
    double b = x2 * x3 - tan(x1 - x2);
    double c = a + 2* c;
    return c*c;
}
```

We will compute derivatives of this function at the point:

```c++
double x0 = 1.0;
double x1 = 1.5;
double x2 = 1.3;
double x3 = 1.2;
```

### Prerequisite: Replace Active Variables

In order to use XAD to differentiate this function,
we first must replace all independent data types and all values that
depend on them with an active data type provided by XAD.
In the above function,
all variables depend on the inputs and thus
all occurrences of `#!c++ double` must be replaced.

This can be done in one of two ways:

1.  The variables can be replaced directly, given the desired mode of differentiation.
    For example, for forward mode `#!c++ double` is replaced by the type
    [`FReal`](../ref/freal.md) and for adjoint mode the type [`AReal`](../ref/areal.md).
2.  The function is made a template, so that it can be called with any data type,
    including the original `#!c++ double`.

We choose the second approach for this tutorial, thus the function becomes:

```c++
template <class T>
T f(T x0, T x1, T x2, T x3)
{
    T a = sin(x0) * cos(x1);
    T b = x2 * x3 - tan(x1 - x2);
    T c = a + 2* b;
    return c*c;
}
```

This means we can use the same definition with both forward and adjoint modes.

### Forward Mode

As illustrated in [Algorithmic Differentiation Background: Forward Mode](aad.md#forward-mode),
when applied to a function with a single output,
the forward mode of algorithmic differentiation can compute *one* derivative
at a time.
For illustration, we choose to derive the function with respect to the input
variable `x0`.

To initiate the forward mode, we must first declare active variables with
the appropriate type.
XAD provides convenience typedefs to select the mode of differentiation,
illustrated in detail in [AD Mode Interface](../ref/interface.md).
For forward mode, we can declare the types needed as:

```c++
typedef xad::fwd<double> mode;
typedef mode::active_type AD;
```

We can then use the `AD` typedef for our variables.

The next step is to initialize the dependent variables,
which is simply done by assigning the input values to new variables of type `AD`:

```c++
AD x0_ad = x0;
AD x1_ad = x1;
AD x2_ad = x2;
AD x3_ad = x3;
```

For forward mode, we must now seed the initial derivative for the variable
we are interested in with the value 1 (as described in [Algorithmic Differentiation Background: Forward Mode](aad.md#forward-mode)), as:

```c++
derivative(x0_ad) = 1.0;
```

The global function [`derivative`](../ref/global.md#derivative) is a convenience function that
works on any active data type.
Alternatively, we could have used the member function [`FReal::setDerivative`](../ref/freal.md#setderivative).

At this point we are ready to call our function and it will compute the function
value as well as the derivative we are interested in:

```c++
AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
```

We can now access the results using the [`value`](../ref/global.md#value) and [`derivative`](../ref/global.md#derivative)
functions on the output (or the member functions [`FReal::getDerivative`](../ref/freal.md#getderivative)
and [`FReal::getValue`](../ref/freal.md#getvalue)).
For example, the following code outputs them to the console:

```c++
std::cout << "y = " << value(y) << "\n"
          << "dy/dx0 = " << derivative(y) << "\n";
```

!!! note "See also"

    This example is included with XAD ([`fwd_1st`](https://github.com/auto-differentiation/xad/tree/main/samples/fwd_1st)).

### Adjoint Mode

The adjoint mode of automatic differentiation
is the natural choice for the function at hand,
as it has a single output and multiple inputs.
We can get all four derivatives in one execution.

Adjoint mode needs a tape to record the operations and their values
during the valuation.
After setting the adjoints of the outputs,
this tape can then be rolled back to compute the adjoints of the inputs.

Both the active data type and the tape type can be obtained from the
interface structure [`adj`](../ref/interface.md#adjt):

```c++
typedef xad::adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;
```

The first step for computing adjoints is to initialise the tape::

```c++
tape_type tape;
```

This calls the default constructor [`Tape::Tape`](../ref/tape.md#construct-destruct-and-assign),
which creates the tape and activates it.

Next, we create the input variables and register them with the tape:

```c++
AD x0_ad = x0;
AD x1_ad = x1;
AD x2_ad = x2;
AD x3_ad = x3;
tape.registerInput(x0);
tape.registerInput(x1);
tape.registerInput(x2);
tape.registerInput(x3);
```

Note that only variables registered as inputs with the tape and all variables dependent on them are recorded.
Also note that before registering active variables, the current threads needs to have an active tape. To ensure thread-safety,
every thread of the application can have its own active tape.

Once the independent variables are set, we can start recording derivatives
on tape and run the algorithm:

```c++
tape.newRecording();

AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
```

At this stage, we have all operations recorded and have the value computed.
We now need to register the outputs with the tape as well,
before we can seed the initial adjoint of the output wit 1
as explained in [Algorithmic Differentiation Background: Adjoint Mode](aad.md#adjoint-mode):

```c++
tape.registerOutput(y);
derivative(y) = 1.0;
```

This uses the global function [`derivative`](../ref/global.md#derivative),
which returns a reference to the stored derivative (or adjoint)
of the given parameter.
Alternatively the member functions [`AReal::setAdjoint`](../ref/areal.md#setadjoint) or
[`AReal::setDerivative`](../ref/areal.md#setderivative) can be used for the same purpose.

What is left is interpreting the tape to compute the adjoints of the
independent variables:

```c++
tape.computeAdjoints();
```

We can now access the adjoints of the inputs,
which are the derivatives we are interested in,
via the global [`derivative`](../ref/global.md#derivative) function or the member function
[`AReal::getDerivative`](../ref/areal.md#getderivative):

```c++
std::cout << "y     = " << value(y) << "\n"
          << "dy/dx0 = " << derivative(x0_ad) << "\n"
          << "dy/dx1 = " << derivative(x1_ad) << "\n"
          << "dy/dx2 = " << derivative(x2_ad) << "\n"
          << "dy/dx3 = " << derivative(x3_ad) << "\n";
```

!!! note "See also"

    This example is included with XAD ([`adj_1st`](https://github.com/auto-differentiation/xad/tree/main/samples/adj_1st)).

### Best Practices

When the algorithm to be evaluated has less outputs than inputs,
adjoint mode should be preferred.
However, when only a small number of derivatives are needed (e.g. less than 5),
the memory for the tape can be avoided by using forward mode.
Experimentation is advised to find the optimal mode for the given algorithm.

## External Functions

Often parts of the algorithm to be differentiated are not available
in source code.
For example, a routine from an external math library may be called.
Reimplementing it may not be desirable (for performance or development effort reasons),
in which case the derivatives of this function need to be implemented
manually in some form.

This can be achieved by either:

*   Applying finite differences to the library function (bumping),
*   Implementing the adjoint code of the function by hand, or
*   Computing the derivatives analytically, possibly using other library functions.

In these cases,
the *external function interface* of XAD can be used to integrate
the manual derivatives, which is described below.
With the same technique,
performance- or memory-critical parts of the application may be hand-tuned.

### Example Algorithm

We pick an example algorithm which computes the length of a multi dimensional
vector.
This is defined as:

$$
y = \sqrt{\sum_0^{N-1} x_i^2}
$$

The goal is to compute the derivatives of $y$ with respect to all
input vector elements using adjoint mode.

The algorithm can be implemented in C++ code as:

```c++
std::vector<double> xsqr(n);
for (int i = 0; i < n; ++i)
    xsqr[i] = x[i] * x[i];
double y = sqrt(sum_elements(x, n));
```

For this example, we assume that the `sum_elements` is an external function
implemented in a library that we do not have source code of.
It has the prototype:

```c++
double sum_elements(const double* x, int n);
```

### External Function For Adjoint Mode

To use the external function, we follow this procedure:

1.  At the point of the call, convert the values of the input active variables
    to the underlying plain data type (`#!c++ double`)
2.  Call the external function passively
3.  Assign the result values to active output variables so the tape recording
    can continue
4.  Store the tape slots of the inputs and outputs with a checkpoint callback object
    and register it with the tape.
5.  When computing adjoints, this callback needs to load the adjoint of the outputs,
    propagate to them to the inputs manually,
    and increment the input adjoints by these values.

We put all the functionality into a callback object.
We derive from the [`CheckpointCallback`](../ref/chkpt_cb.md#checkpointcallback) base class
and implement at least the virtual method [`CheckpointCallback::computeAdjoint`](../ref/chkpt_cb.md#computeadjoint).
This method gets called during tape rollback.
We also place the forward computation within the same object
(this could also be done outside of the callback class).
The declaration of our callback class looks like this:

```c++
template <class Tape>
class ExternalSumElementsCallback : public xad::CheckpointCallback<Tape>
{
public:
    typedef typename Tape::slot_type   slot_type;   // type for slot in the tape
    typedef typename Tape::value_type  value_type;  // double
    typedef typename Tape::active_type active_type; // AReal<double>

    active_type computeExternal(const active_type* x, int n); // forward compute
    void computeAdjoint(Tape* tape) override;                 // adjoint compute

private:
    std::vector<slot_type> inputSlots_;             // slots of inputs in tape
    slot_type outputSlot_;                          // slot of output in tape
   };
```

We declare it as a template for arbitrary tape types,
which is good practice as it allows to reuse
this implementation with higher order derivatives too.

#### `computeExternal` Method

Within the `computeExternal` method,
we first store the slots in the tape for the input variables,
as we will need them during adjoint computation to increment the corresponding
adjoints.
We use the `inputSlots_` member vector to keep this information:

```c++
for (int i = 0; i < n; ++i)
    inputSlots_.push_back(x[i].getSlot());
```

Then we create a copy of the active inputs and store them in a vector of passive
values,
with which we can call the external function:

```c++
std::vector<value_type> x_p(n);
for (int i = 0; i < n; ++i)
    x_p[i] = value(x[i]);
    
value_type y = sum_elements(&x_p[0], n);
```

We now need to store this result in an active variable,
register it as an output of the external function
(to allow the tape to continue recording dependent variables),
and keep its slot in the tape for the later adjoint computation:

```c++
active_type ret = y;
Tape::getActive()->registerOutput(ret);
outputSlot_ = ret.getSlot();
```

Finally we need to insert the callback into the tape,
hence requesting it to be called during adjoint rollback of the tape,
and return:

```c++
Tape::getActive()->insertCallback(this);
return ret;
```

#### `computeAdjoint` Method

The `computeAdjoint` method gets called by XAD during tape rollback.
We need to override this method and implement the manual adjoint code.
For a simple sum operation, this is straightforward:
all input adjoints are equal to the output adjoint since all
partial derivatives are 1.
Thus we need to obtain the output adjoint and increment all input adjoints by
this value:

```c++
value_type output_adj = tape->getAndResetOutputAdjoint(outputSlot_);
for (int i = 0; i < inputSlots_.size(); ++i)
    tape->incrementAdjoint(inputSlots_[i], output_adj); 
```

The function [`Tape::getAndResetOutputAdjoint`](../ref/tape.md#getandresetoutputadjoint) obtains the
adjoint value corresponding to the given slot and resets it to zero.
This reset is necessary in general as the output variable may
have been overwriting other values in the forward computation.
The [`Tape::incrementAdjoint`](../ref/tape.md#incrementadjoint) function simply
increments the adjoint with the given slot by the given value.

#### Wrapper Function

With the checkpointing callback class in place,
we can implement a `sum_elements` overload for [`AReal`](../ref/areal.md) that
wraps the creation of this callback::

```c++
template <class T>
xad::AReal<T> sum_elements(const xad::AReal<T>* x, int n)
{
    typedef typename xad::AReal<T>::tape_type tape_type;
    tape_type* tape = tape_type::getActive();
    ExternalSumElementsCallback<tape_type>* ckp = 
        new ExternalSumElementsCallback<tape_type>;
    tape->pushCallback(ckp);

    return ckp->computeExternal(x, n);
}
```

This function dynamically allocates the checkpoint callback object
and lets the tape manage its destruction via the [`Tape::pushCallback`](../ref/tape.md#pushcallback)
function.
This call simply ensures that the callback object is destroyed
when the tape is destroyed,
making sure that no memory is leaked.
If the callback object was managed elsewhere, this call would not be necessary.
It then redirects the computation to the `computeExternal` function
of the checkpoint callback class.
Using this wrapper class, the `sum_elements` function can be used for active types
in the same fashion as the original external function `sum_elements` for `#!c++ double`.
Defining it as a template allows us to re-use this function for higher-order derivatives,
should we need them in future.

#### Call-Site

The call site then can be implemented as
(assuming that `x_ad` is the vector holding the independent variables, already registered on tape):

```c++
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
```

This follows exactly the same procedure as given in [Basic Usage](#adjoint-mode).

!!! note "See also"

     This example is included with XAD ([`external_function`](https://github.com/auto-differentiation/xad/tree/main/samples/external_function)).

### External Function For Forward Mode

Since forward mode involves no tape,
a manual implementation of the derivative computation needs to be implemented
together with computing the value.
The manual derivatives can be updated directly in the output values
using the [`derivative`](../ref/global.md#derivative) function.

In our example, we can implement the external function in forward mode as:

```c++
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
```

We first extract the passive values from the `x` vector and call the
external library function to get the passive output value `y_p`.
This value is then assigned to the active output variable `y`,
which also initializes its derivative to `0`.

As we have a simple sum in this example,
the derivative of the output
is a sum of the derivatives of the inputs,
which is computed by the loop in the end.

!!! note "See also"

    This example is included with XAD ([`external_function`](https://github.com/auto-differentiation/xad/tree/main/samples/external_function)).

## Checkpointing

Checkpointing is a technique to reduce the memory footprint of the tape
in adjoint mode algorithmic differentiation.
Instead of recording the full algorithm on tape,
which can quickly result in gigabytes of memory in a large computation,
the tape is recorded for specific stages of the algorithm, one at a time.
This is illustrated in the following figure:

![Checkpointing](../images/checkpointing.svg)

The algorithm is divided into stages,
where the input data of each stage is stored in a checkpoint
and the outputs are computed passively (without recording on tape).
Once the final output of the algorithm is computed,
the adjoint of the output is initialized
and at each checkpoint during tape rollback:

1.  The inputs to the checkpoint are loaded,
2.  The operations of this stage only are recorded on tape,
3.  The output adjoints of this stage are initialized,
4.  The tape is rolled back for this stage, computing the adjoints of the stage inputs,
5.  The input adjoints are incremented by these values, and
6.  The tape is wiped before proceeding with the previous stage.

Using this method,
the tape memory is limited by the amount needed to record one algorithm stage
instead of the full algorithm.
However, each forward computation is computed twice,
hence checkpointing trades computation for memory.

In practice, as using less memory leads to higher cache-efficiency,
checkpointing may be faster overall than recording the full algorithm
even though more computations are performed.

### Example Algorithm

To demonstrate the checkpointing method,
we choose a simple repeated application of the sine function to a single input:

```c++
template <class T>
void repeated_sin(int n, T& x)
{
    for (int i = 0; i < n; ++i)
    x = sin(x);
}
```

We divide the for loop into equidistant stages and insert a checkpoint at each
of these.

### Checkpoint Callback

To create a checkpoint,
we need to store the inputs of the stage and the slots in the tape for the inputs
and outputs in a callback object inheriting from [`CheckpointCallback`](../ref/chkpt_cb.md#checkpointcallback).
The virtual method [`CheckpointCallback::computeAdjoint`](../ref/chkpt_cb.md#computeadjoint)
needs to be overridden to perform the per-stage adjoint computation.
As all stages are identical, we choose to implement the functionality of all
checkpoints within a single callback object
and store the required inputs in a stack data structure.
Alternatively we could have created a new checkpoint callback object at every
checkpoint.
The prototype for our callback is:

```c++
template <class Tape>
class SinCheckpointCallback : public xad::CheckpointCallback<Tape>
{
public:
    typedef typename Tape::slot_type   slot_type;   // type for slot in the tape
    typedef typename Tape::value_type  value_type;  // double
    typedef typename Tape::active_type active_type; // AReal<double>

    active_type computeStage(int n, active_type& x); // forward computation
    void computeAdjoint(Tape* tape) override;        // adjoint computation

private:
    std::stack<int> n_;                    // number of iterations in this stage
    std::stack<value_type> x_;             // input values for this stage
    std::stack<slot_type> slots_;          // tape slots for input and output
};
```

For convenience of implementation,
we added the forward computation for one stage within the same class
in the `computeStage` method,
which could also be performed outside of the object.

#### `computeStage` Method

Within the `computeStage` method,
we first store the input value, the number of iterations,
and the slots of the input in the checkpoint object:

```c++
n_.push(n);
slots_.push(x.getSlot());
value_type x_p = value(x);
x_.push(x_p);
```

We then compute the stage with the passive variable (not recording on the tape):

```c++
repeated_sin(n, x_p);
```

The value of the output active variable needs to be updated with the result
and we need to store the slot of the output variable in the checkpoint also:

```c++
value(x) = x_p;
slots_.push(x.getSlot());
```

Note that we did not need to register `x` as an output with the tape here,
as we had to do with the external functions example before,
since the variable is already registered on tape (it's both input and output).

What is left is to register this callback object with the tape so that its
`computeAdjoint` method is called at this point when the tape is rolled back:

```c++
Tape::getActive()->insertCallback(this);
```

#### `computeAdjoint` Method

The `computeAdjoint` method is called automatically by XAD
at the checkpoints in the tape.
We first need to load the inputs to this computation stage and
obtain the adjoint of the output:

```c++
slot_type outputidx = slots_.top();  slots_.pop();
slot_type inputidx = slots_.top();   slots_.pop();
int n = n_.top();                    n_.pop();
value_type outputadj = tape->getAndResetOutputAdjoint(outputslot);
```

The function [`Tape::getAndResetOutputAdjoint`](../ref/tape.md#getandresetoutputadjoint) reads the adjoint
corresponding to the slot given and resets it to 0.
This reset is generally required as the variable corresponding to the slot
may be re-used (overwritten) in the algorithm,
as is the case in the `repeated_sin` function.

We now want to use XAD to compute the adjoints just for this computation
stage.
This is performed by creating a nested recording within the global tape,
than can be rolled back individually:

```c++
active_type x = x_.top();               // local independent variable
x_.pop();
tape->registerInput(x);                 // need to register to record
   
xad::ScopedNestedRecording<Tape> nested(tape);  // nested recording
repeated_sin(n, x_ad);                  // run actively
tape->registerOutput(x);                // register x as an output
derivative(x) = output_adj;             // set output adjoint
nested.computeAdjoints();               // rollback nested tape

nested.incrementAdjoint(inputslot, derivative(x));  // incr. input adjoint
```

In a similar fashion to simple adjoint mode (see [Basic Usage](#adjoint-mode)),
we first initialize the local independent variables as active data types
and start a nested recording.
This is performed by creating a local object `nested` of type
[`ScopedNestedRecording`](../ref/tape.md#scopednestedrecording),
which wraps calls to [`Tape::newNestedRecording`](../ref/tape.md#newnestedrecording) in its constructor
and [`Tape::endNestedRecording`](../ref/tape.md#endnestedrecording) in its destructor.
It is recommended to use the [`ScopedNestedRecording`](../ref/tape.md#scopednestedrecording) for this
purpose to make sure the nested recording is always finished when the scope is left.

Next we record the operations for this stage by running the algorithm actively.
We then set the adjoint of the output and compute the adjoints of the inputs.
The adjoints of the inputs to this stage can then be incremented.

Note that when the `nested` object goes out of scope,
i.e. when its destructor is called,
the nested tape for this computation stage is wiped and the memory can be
reused for the previous stage.
This saves overall memory.

### Call-Site

The full algorithm with checkpointing can then be initiated as follows:

```c++
tape_type tape;

AD x_ad = x;                             // initialized indepedent variables
tape.registerInput(x_ad);                // register with the tape
tape.newRecording();                     // start recording derivatives

SinCheckpointCallback<tape_type> chkpt;  // setup checkpointing object

int checkpt_distance = 4;                // we checkpoint every 4 iterations
for (int i = 0; i < n; i += checkpt_distance)
{
    int m = min(checkpt_distance, n-i);
    chkpt.computeStage(m, x_ad);             // one computation stage
}

tape.registerOutput(x_ad);
derivative(x_ad) = 1.0;
tape.computeAdjoints();

std::cout << "xout       = " << value(x_ad) << "\n"
          << "dxout/dxin = " << derivative(x_ad) << "\n";
```

This follows largely the same procedure as given in [Basic Usage](#adjoint-mode),
but setting up the checkpoint object and calling its `computeStage` member
for every stage of the algorithm (4 iterations in this example).

!!! note

    It is important that the checkpoint callback object is valid when
    [`Tape::computeAdjoints`](../ref/tape.md#computeadjoints) is called. 
    It should not be destroyed before.

See [Checkpoint Callback Memory Management](../ref/tape.md#checkpoint-callback-memory-management) 
for how to use tape-based destruction with
dynamically allocated checkpoint callbacks.

!!! note "See also"

    This example is included with XAD ([`checkpointing`](https://github.com/auto-differentiation/xad/tree/main/samples/checkpointing)).

### Other Usage Patterns

Alternative methods may be used to update the adjoints
within a checkpoint's [`CheckpointCallback::computeAdjoint`](../ref/chkpt_cb.md#computeadjoint) method,
such as:

*   Forward mode algorithmic differentiation within an outer adjoint mode
*   Finite differences (bumping)
*   Analytic derivatives
*   External library functions (see [External Functions](#external-functions))

Checkpointing can also be used recursively,
i.e., new checkpoints are created within a nested tape in a checkpoint.

The benefits of each of these approaches are highly application-dependent.


## Higher-Order Derivatives

As explained in [Algorithmic Differentiation Background: Higher Orders](aad.md#higher-orders), higher order derivatives can be computed
by nesting first order algorithmic differentiation techniques.
For example, one can obtain second order by computing forward mode over adjoint mode.
With XAD,
this technique can be used directly to compute higher order derivatives.

XAD's automatic differentiation interface structures (see [AD Mode Interface](../ref/interface.md))
define second order mode data types for easy access.
Types for third or higher orders need to defined manually
from the basic first-order types.

We will demonstrate second-order derivatives using forward-over-adjoint mode
in the following.

### Example Algorithm

For demonstration purposes, we use the same algorithm from [Basic Usage](#basic-usage):

```c++
template <class T>
T f(T x0, T x1, T x2, T x3)
{
    T a = sin(x0) * cos(x1);
    T b = x2 * x3 - tan(x1 - x2);
    T c = a + 2* b;
    return c*c;
}
```

We are interested in derivatives at the point:

```c++
double x0 = 1.0;
double x1 = 1.5;
double x2 = 1.3;
double x3 = 1.2;
```

### Forward Over Adjoint

In this mode, we can compute all first-order derivatives (as a single output
function derived with adjoints gives all first order derivatives),
and the first row of the Hessian matrix of second order derivatives.
The full Hessian is defined as:

$$
\pmb{H} = \left[ \begin{array}{cccc}
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
$$

Note that the Hessian matrix is typically symmetric,
which can be used to reduce the amount of computation needed for the full Hessian.

The first step is to set up the tape and active data types needed for this computation:

```c++
typedef xad::fwd_adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;

tape_type tape;
```

Note that the active type for this mode is actually `#!c++ AReal<FReal<double>>`.

Now we need to setup the independent variables and register them:

```c++
AD x0_ad = x0;
AD x1_ad = x1;
AD x2_ad = x2;
AD x3_ad = x3;

tape.registerInput(x0_ad);
tape.registerInput(x1_ad);
tape.registerInput(x2_ad);
tape.registerInput(x3_ad);
```

As we compute the second order using forward mode,
we need to seed the initial derivative for the second order before running the algorithm:

```c++
derivative(value(x0_ad)) = 1.0;
```

The inner call to [`value`](../ref/global.md#value) takes the value of the outer type,
i.e. it returns the value as the type `#!c++ FReal<double>`,
of which we set the derivative to `1`.

Now we can start recording derivatives on the tape and run the algorithm:

```c++
tape.newRecording();

AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);
```

For the inner adjoint mode, we need to register the output and seed the initial adjoint with 1:

```c++
tape.registerOutput(y);
value(derivative(y)) = 1.0;
```

Here, the inner call to [`derivative`](../ref/global.md#derivative) gives the derivative of the outer
type, i.e. the derivative of the adjoint-mode active type.
This is of type `#!c++ FReal<double>`, for which we set the value to `1`.

Next we compute the adjoints, which computes both the first and second order
derivatives:

```c++
tape.computeAdjoints();
```

We can now output the result:

```c++
std::cout << "y = " << value(value(y)) << "\n";
```

And the first order derivatives:

```c++
std::cout << "dy/dx0 = " << value(derivative(x0_ad)) << "\n"
          << "dy/dx1 = " << value(derivative(x1_ad)) << "\n"
          << "dy/dx2 = " << value(derivative(x2_ad)) << "\n"
          << "dy/dx3 = " << value(derivative(x3_ad)) << "\n";
```

Note again that the inner call to [`derivative`](../ref/global.md#derivative) obtains the derivative
of the outer active data type,
hence it gives a `#!c++ FReal<double>` reference that represents the first order adjoint value.
We can get this value as a `#!c++ double` using the [`value`](../ref/global.md#value) call.

The second order derivatives w.r.t. `x0` can be obtained as:

```c++
std::cout << "d2y/dx0dx0 = " << derivative(derivative(x0_ad)) << "\n"
          << "d2y/dx0dx1 = " << derivative(derivative(x1_ad)) << "\n"
          << "d2y/dx0dx2 = " << derivative(derivative(x2_ad)) << "\n"
          << "d2y/dx0dx3 = " << derivative(derivative(x3_ad)) << "\n";
```

which 'unwraps' the derivatives of the first and second order active types.

The result of the running the application for the given inputs is:

```text
y      = 7.69565
dy/dx0 = 0.21205
dy/dx1 = -16.2093
dy/dx2 = 24.8681
dy/dx3 = 14.4253
d2y/dx0dx0 = -0.327326
d2y/dx0dx1 = -3.21352
d2y/dx0dx2 = 0.342613
d2y/dx0dx3 = 0.198741
```

Forward over adjoint is the recommended mode for second-order derivatives.

!!! note "See also"

    This example is included with XAD ([`fwd_adj_2nd`](https://github.com/auto-differentiation/xad/tree/main/samples/fwd_adj_2nd)).

### Other Second-Order Modes

Other second-order modes work in a similar fashion.
They are briefly described in the following.

#### Forward Over Forward

With forward-over-forward mode,
there is no tape needed and the derivatives of both orders need to be seeded
before running the algorithm.
One element of the Hessian and one first-order derivative can be computed
with this method, if the function has one output.
The derivative initialization sequence in this mode is typically:

```c++
value(derivative(x)) = 1.0;   // initialize the first-order derivative
derivative(value(x)) = 1.0;   // initialize the second-order derivative
```

After the computation, the first order derivative can be retrieved as:

```c++
std::cout << "dy/dx = " << derivative(value(y)) << "\n";
```

And the second order derivative as:

```c++
std::cout << "d2y/dxdx = " << derivative(derivative(y)) << "\n";
```

With different initial seeding, different elements of the Hessian can be obtained.

#### Adjoint Over Forward

Here the inner mode is forward,
computing one derivative in a tape-less fashion,
and the outer mode is adjoint, requiring a tape.
With this mode, we need to initialize the forward-mode derivative with:

```c++
value(derivative(x)) = 1.0;   // initialize the first-order derivative
```

As the derivative of the output corresponds to the first order result,
we need to seed its derivative (i.e. the adjoint) after running the algorithm:

```c++
derivative(derivative(y)) = 1.0;
```

After tape interpretation, we can now obtain the first-order derivative as:

```c++
std::cout << "dy/dx = " << value(derivative(y)) << "\n";
```

Due to the symmetries in this mode of operation, the same first-order derivatives
can also be obtained as:

```c++
std::cout << "dy/dx = " << derivative(derivative(x)) << "\n";
```

Which allows to get all first-order derivatives w.r.t. to all inputs in this mode,
similar to the forward-over-adjoint mode.

The second-order derivatives can be obtained as:

```c++
std::cout << "d2y/dxdx = " << derivative(value(x))
```

#### Adjoint Over Adjoint

As both nested modes are adjoint,
this mode needs to two tapes for both orders.
Hence the types defined in the interface structure [`adj_adj`](../ref/interface.md#adj_adjt)
need an inner and an outer tape type:

```c++
typedef xad::adj_adj<double> mode;
typedef mode::inner_tape_type inner_tape_type;
typedef mode::outer_tape_type outer_tape_type;
typedef mode::active_type AD;
```

In this mode, no initial derivatives need to be set,
but it is important that both tapes are initialized and a new recording is
started on both before running the algorithm.

After the execution, the outer derivative needs to be seeded as:

```c++
value(derivative(y)) = 1.0;
```

And then the outer tape needs to compute the adjoints.
This computes the `value(derivative(x))` as an output,
and the derivative of this needs to be set before interpreting the inner tape:

```c++
derivative(derivative(x)) = 1.0;
```

After calling `computeAdjoints()` on the inner tape,
we can read the first-order derivatives as:

```c++
std::cout << "dy/dx = " << value(derivative(x)) << "\n;
```

And the second-order derivatives as:

```c++
std::cout << "d2y/dxdx" << derivative(value(x)) << "\n";
```

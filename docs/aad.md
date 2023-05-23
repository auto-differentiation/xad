---
hide:
  - navigation
  - footer
description: Required mathematical background for the XAD C++ automatic differentiation library.
---

# Algorithmic Differentiation Background

As every computer program is made up of a series of simple arithmetic operations,
i.e.

$$
a \rightarrow b \rightarrow \ldots \rightarrow u \rightarrow v \rightarrow \ldots \rightarrow z
$$

where the inputs $a$ are modified in stages
in order to get the final output $z$.
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

## Finite Differences

The traditional approach for computing these derivatives
is by employing a finite difference approximation.
That is, each of the input variables are *bumped* one by one
and the change of the result is used to estimate the sensitivities:

$$
\begin{align}
\frac{\partial f(x, \pmb{y})}{\partial x} &= \lim_{h\rightarrow 0}\frac{f(x+h, \pmb{y}) - f(x,\pmb{y})}{h}  \\
\frac{\partial f(x, \pmb{y})}{\partial x} &= \lim_{h\rightarrow 0} \frac{f(x+h, \pmb{y}) - f(x-h, \pmb{y})}{2h}
\end{align}
$$

where $f(x, \pmb{y})$ is the function of which we are interested in derivatives
with respect to the input parameter $x$.
The vector-valued argument $\pmb{y}$ denotes the remaining function parameters.
The first equation represents *forward finite differences* and requires two
evaluations of the function.
The second equation gives *central finite differences*
with potentially higher accuracy
and requires two function evaluation for the derivative and another evaluation
for the function's value.

In practice, the value $h$ is chosen small enough to approximate the theoretical limit,
but large enough to cause a detectable change of the result beyond typical numerical error levels.
Clearly, this choice impacts the accuracy of the approximation.

Further, this method implies that the function needs to be evaluated
once for the result and once for each derivative that we are interested in.
This results in a high overall computational complexity
as soon as more than a few derivatives are needed.

Thus, the finite differences approach has accuracy and performance limitations.

## Forward Mode

### Theory

The forward mode defines $\dot{u}$
as the derivative of $u$ with respect to $a$, i.e.

$$
\dot{u} = \frac{\partial u}{\partial a}
$$

Applying the chain rule of differentiation
and assuming that the intermediate variables are vectors,
the elements of $\dot{v}$ can be calculated as

$$
\dot{v}_i = \sum_j \frac{\partial v_i}{\partial u_{j}} \dot{u}_j
$$

Applying this to each step in the chain of operations from inputs to outputs,
the value of $\dot{z}$ can be calculated.
This is the *forward mode* of algorithmic differentiation.

For a function $f,{:},\mathbb{R}^n,{\rightarrow},\mathbb{R}^m$,
one application of the forward mode
gives the sensitivities for all $m$ outputs with respect to
*one* input parameter.
It needs to be re-evaluated $n$ times to obtain all sensitivities.
The computational cost is constant in the number of output variables $m$
and linear in the number of input variables $n$.

### Example

We illustrate the forward mode on the example function:

$$
z = \sin x_1 + x_1 x_2
$$

Which can be implemented in a computer program as:

```c++
a = sin(x1);
b = x1 * x2;
z = a + b;
```

We are interested of the derivative with respect to $x_1$
for the input values $x_1 = \pi$ and $x_2 = 2$.
The following figure illustrates how the forward mode algorithm differentiation
is applied to this problem:

![Forward mode example](images/forward_illustration.svg)

On the left we see the computational graph representing the equation,
and the table on the right illustrates the the steps performed.

In step 0, we initialize the input values and we seed the derivatives
of these inputs.
As we are interested in the derivative w.r.t. $x_1$,
we set its derivative to 1 while setting the other to 0.

Next we compute $a$ by taking the sine function.
The value of $a$ is zero,
while $\dot{a}$ is computed by multiplying
the partial derivative of the sine w.r.t. to $x_1$,
i.e. the cosine, with $\dot{x_1}$.
This gives a value of -1.

In the next step, the value of $b$ is computed as usual,
and $\dot{b}$ is calculated similarly to $\dot{a}$,
this time depending on both  $\dot{x_1}$ and $\dot{x_2}$.
This results in a value of 2.

The final statement adds both $a$ and $b$,
which gives the result of $2\pi$.
To calculate $\dot{z}$,
we see that the  $\dot{a}$ and $\dot{b}$
can simply be added
since their partial derivatives are both 1.
This gives a final derivative of 1.

Hence:

$$
\left.\frac{\partial z}{\partial x_1}\right|_{(\pi,2)} = 1
$$

which can be easily verified analytically.

## Adjoint Mode

### Theory

The adjoint mode applies the chain rule backwards,
from outputs to inputs.
Using standard notation, we define

$$
\bar{u}_i = \frac{\partial z}{\partial u_i}
$$

where $i$ is the index in the vector $\pmb{u}$.
Applying the chain rule yields

$$
\frac{\partial z}{\partial u_i} = \sum_j \frac{\partial z}{\partial v_j} \frac{\partial v_j}{\partial u_i}
$$

which leads to the *adjoint mode equation*

$$
\bar{u}_i    =  \sum_j \frac{\partial v_j}{\partial u_i} \bar{v}_j
$$

Seeding $\bar{z} = 1$,
the adjoint mode equation can be applied for each step,
from output to input,
to obtain $\bar{\pmb{a}}$,
which is the derivative of the output $z$
with respect to each of the input variables $\pmb{a}$.

For a function $f,{:},\mathbb{R}^n,{\rightarrow},\mathbb{R}^m$,
the adjoint mode gives the sensitivities of *one* output
with respect to all $n$ input parameters.
It needs to be re-evaluated $m$ times to obtain all sensitivities.
The computational cost is constant in the number of input variables $n$
and linear in the number of output variables $m$.

### Example

We illustrate the adjoint mode using the same example as above:

$$
z = \sin x_1 + x_1 x_2
$$

implemented as:

```c++
a = sin(x1);
b = x1 * x2;
z = a + b;
```

With adjoint mode, we can get both partial derivatives of the output
in a single execution,
for the input values $x_1 = \pi$ and $x_2 = 2$.
This is illustrated in the figure below:

![Adjoint mode example](images/adjoint_illustration.svg)

As the adjoint mode walks from outputs back to inputs,
we execute the full computation of the value as usual,
until we have an output for $z$ of $2\pi$.

Then we seed the adjoint of $z$ to 1 in the final step,
and walk backwards to compute the adjoints of the inputs.

In step 2, we can compute the adjoint of $b$ by multiplying
the adjoint of $z$ with the partial derivative of the equation
for $z$ w.r.t. $b$, which is 1.

The same is performed in step 1 to compute the adjoint of $a$,
which also yields 1.

The adjoint of $x_2$ is then computed by multiplying the partial
derivative of $b$ w.r.t. $x_2$ with the adjoint of $b$,
which gives the value $\pi$.

The same method is applied to compute the adjoint of $x_1$,
giving the value 1.

Thus, the two derivatives we were interested in are:

$$
\begin{align}
\left.\frac{\partial z}{\partial x_1}\right|_{(\pi,2)} &= 1 &,&&
\left.\frac{\partial z}{\partial x_2}\right|_{(\pi,2)} &= \pi
\end{align}
$$

Which can be easily verified analytically.

## Higher Orders

Higher order derivatives can be obtained by nesting the principles described above.
For example, applying forward mode algorithmic differentiation over adjoint mode
gives second order derivatives.
This method can be extended to any order.

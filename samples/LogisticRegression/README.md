# Logistic regression with adjoint differentiation

This sample shows how to use XAD to differentiate the loss of a binary logistic
regression model. It generates its own small dataset, computes the gradient for
every model weight in one adjoint sweep, checks two gradients against finite
differences, and takes 20 gradient-descent steps.

The example is self-contained: it does not read a dataset or depend on a
machine-learning library. See [`main.cpp`](main.cpp) for the complete program.

## Dataset and model

The program generates 1,000 points with 10 normally distributed features per
point. It uses `std::mt19937` with seed `42` for reproducibility. A separate set
of randomly generated weights determines each binary label: the label is `1`
when the weighted feature sum plus the bias is positive, and `0` otherwise.
These generating weights are used only to make the labels; training starts with
all model weights set to zero.

There are 11 trainable values: one weight for each feature and a bias at index
`10`. For point `i`, the model computes a score:

```text
score_i = bias + sum_j(weight_j * feature_ij)
```

The objective is mean binary cross-entropy. The program evaluates it directly
from each score, without forming a probability or adding an epsilon:

```text
if score_i >= 0:
    loss_i = log1p(exp(-score_i)) + (1 - label_i) * score_i
else:
    loss_i = log1p(exp(score_i)) - label_i * score_i
loss = mean_i(loss_i)
```

The two branches express the same loss. Choosing by the sign of the score
keeps the exponential argument non-positive, avoiding overflow for large
scores. A single templated function computes this loss for both XAD values
and ordinary `double` values.

## Adjoint gradient

The sample registers all 11 weights as XAD inputs and records the loss
calculation. It seeds the loss derivative with `1.0` and calls
`computeAdjoints()` once. That reverse sweep provides the derivative of the loss
with respect to every registered weight, including the bias. The program prints
the loss and all 11 gradients at the initial zero weights.

## Finite-difference check

To check the adjoint result independently, the sample estimates the gradients
for weights `0` and `1` using central finite differences. For each weight, it
evaluates the same loss with that weight increased and decreased by `h = 1e-5`:

```text
finite_difference = (loss(weight + h) - loss(weight - h)) / (2 * h)
relative_difference = abs(adjoint - finite_difference) / (abs(adjoint) + 1e-15)
```

The output shows the adjoint gradient, finite-difference gradient, and relative
difference for both weights. The program exits with a nonzero status if either
gradient is non-finite or its relative difference exceeds `1e-5`. This also
makes the registered CTest sample fail when the comparison fails.

## Gradient descent

Starting again from zero weights, the sample records the loss and computes a
new adjoint gradient on each of 20 iterations. It reuses the initial tape:
after each iteration's active values leave scope, it calls `clearAll()`,
registers the current weights, and starts a new recording. It prints the loss
**before** each update, then subtracts `0.1 * gradient` from each weight. On
this seeded dataset, the printed loss falls across the iterations.

This is a small differentiation example, not a complete classifier training
pipeline: it does not split the dataset, report accuracy, or save a model. The
descent trend is printed for inspection; the automatic pass/fail check covers
the two finite-difference comparisons.

## Build and run

From the repository root, build the sample with tests enabled:

```bash
cmake -S . -B build -DXAD_ENABLE_TESTS=ON
cmake --build build --target LogisticRegression
```

Run its registered sample test from the build directory:

```bash
cd build
ctest -R '^sample_LogisticRegression$' --output-on-failure
```

Use `ctest -V -R '^sample_LogisticRegression$'` to see the printed gradients
and loss values. Exact values from the synthetic dataset can vary between C++
standard-library implementations, while runs with the same build and seed
are reproducible.

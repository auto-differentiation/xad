# LogisticRegression

Computes the binary cross-entropy loss and all 11 weight gradients for a logistic
regression model in one adjoint sweep. It checks two gradients against central
finite differences, then shows the loss falling over 20 gradient-descent steps.

The sample generates 1,000 points with 10 normally distributed features using
`std::mt19937` with seed `42`. Random weights produce binary labels; the model
being trained starts with all weights at zero. No dataset file or machine-learning
library is needed.

The loss is written once as a template for both XAD's active type and `double`:

    score_i = bias + sum_j(weight_j * feature_ij)

    if score_i >= 0:
        loss_i = log1p(exp(-score_i)) + (1 - label_i) * score_i
    else:
        loss_i = log1p(exp(score_i)) - label_i * score_i

    loss = mean_i(loss_i)

The two branches give the same binary cross-entropy. They keep the exponential
argument non-positive, avoiding overflow for large scores without adding an
epsilon to the loss.

## Why one adjoint sweep is enough

The 10 feature weights and the bias are registered as XAD inputs. After recording
the full-dataset loss, the sample seeds its derivative with `1.0` and calls
`computeAdjoints()` once. This gives the gradient for all 11 weights. The same
tape is cleared and reused for each gradient-descent iteration.

For weights `0` and `1`, the sample also estimates the gradient by evaluating the
same loss at `weight + h` and `weight - h`, with `h = 1e-5`:

    finite_difference = (loss(weight + h) - loss(weight - h)) / (2 * h)
    relative_difference = abs(adjoint - finite_difference) / (abs(adjoint) + 1e-15)

It prints both gradients and their relative difference. The sample returns a
non-zero exit code if either gradient is non-finite or the relative difference
exceeds `1e-5`. Gradient descent uses a learning rate of `0.1`; its loss is
printed before each of the 20 updates.

## Build and run

From the repository root:

    cmake -S . -B build
    cmake --build build --target LogisticRegression
    build/samples/LogisticRegression/LogisticRegression

Or, from the build directory, `ctest -R LogisticRegression`. The sample returns
a non-zero exit code if either finite-difference check fails.

## Expected output

The structure is fixed; exact gradients and loss values may vary between C++
standard-library implementations. A representative run prints:

    Dataset: 1000 points, 10 features

    adjoint sweep at w = 0
    loss = 0.693147
    gradients dL/dw:
      w[ 0] =     0.088978
      w[ 1] =    -0.054788
      ...

    finite-difference check
     index    adjoint grad         FD grad       rel. diff
         0        0.088978        0.088978       2.336e-10
         1       -0.054788       -0.054788       1.148e-09

    gradient descent (lr = 0.1, 20 iterations)
    iter  0  loss =   0.693147
    ...
    iter 19  loss =   0.497008

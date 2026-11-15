# Cross-Gamma Matrix of a Basket Option (Second-Order AD)

Computes the full matrix of second derivatives (gammas and cross-gammas, $\Gamma_{ij} = \frac{\partial^2 C}{\partial S_i \partial S_j}$) of a multi-asset European basket call option price with respect to underlying asset spot prices using XAD's forward-over-adjoint mode (`xad::fwd_adj<double>`).

## Problem Formulation

Consider a basket of $N = 15$ correlated assets with spot prices $S_i$, weights $w_i$ ($\sum_{i=1}^N w_i = 1$), volatilities $\sigma_i$, and correlation matrix $\rho_{ij} = \rho^{|i-j|}$ with $\rho = 0.5$.

Under risk-neutral dynamics with risk-free rate $r$ and maturity $T$, the basket value at maturity is:
$$B_T = \sum_{i=1}^N w_i S_i(T)$$

Because the sum of correlated lognormals does not have a closed-form density, we employ lognormal moment matching (Levy approximation). The first two moments are matched exactly:

1. **First Moment:**
   $$M_1 = \mathbb{E}[B_T] = e^{r T} \sum_{i=1}^N w_i S_i$$

2. **Second Moment:**
   $$M_2 = \mathbb{E}[B_T^2] = \sum_{i=1}^N \sum_{j=1}^N w_i w_j S_i S_j \exp\left( (2r + \rho_{ij} \sigma_i \sigma_j) T \right)$$

3. **Effective Volatility & Black-Scholes Pricing:**
   $$\nu^2 = \ln\left(\frac{M_2}{M_1^2}\right) = \ln(M_2) - 2 \ln(M_1), \quad \nu = \sqrt{\nu^2}$$
   $$d_1 = \frac{\ln(M_1 / K) + \frac{1}{2} \nu^2}{\nu}, \quad d_2 = d_1 - \nu$$
   $$C = e^{-r T} \left( M_1 N(d_1) - K N(d_2) \right)$$

This analytical approximation is completely smooth with respect to all spot prices $S_i$, allowing exact, noise-free computation of the entire second-order Hessian matrix.

## Second-Order AD (Forward-over-Adjoint)

XAD's forward-over-adjoint mode (`xad::fwd_adj<double>`) computes the second-order derivative matrix $\Gamma_{ij}$ efficiently:
- For each asset $i$, we seed the tangent forward derivative $\dot{S}_i = 1$.
- The pricing graph is recorded on the tape.
- Seeding the output adjoint $\bar{C} = 1$ and executing a single reverse sweep yields the entire row of cross-gammas $\Gamma_{ij} = \frac{\partial^2 C}{\partial S_i \partial S_j}$ for all $j \in \{1, \dots, N\}$.
- Repeating this for $N$ assets yields the full $N \times N$ Hessian matrix in only $N$ sweeps.

## Checks and Regression Testing

The sample carries out two automated checks:
1. **Symmetry:** Checks that the computed cross-gamma matrix is symmetric ($\max_{i < j} |\Gamma_{ij} - \Gamma_{ji}| < 10^{-12}$).
2. **Finite-Difference Validation:** Compares all elements $\Gamma_{ij}$ against central finite differences of first-order adjoint deltas with relative difference $< 10^{-4}$.

If any check fails, the binary exits with a non-zero exit code to alert CTest of regressions.

## Performance Reporting

The sample benchmarks and reports execution costs relative to single pricing baseline ($1.0\times$), comparing AD forward-over-adjoint against finite-difference bumping:
- **Baseline:** Single pricing evaluation.
- **AD Forward-over-Adjoint:** Full $N \times N$ matrix via $N$ passes.
- **Finite-Difference Bumping:** Bumping requires $2$ pricings per diagonal element and $4$ per off-diagonal element, totaling $2N^2$ pricing evaluations ($450$ for $N = 15$).

## Build and Run

From the build directory:

```bash
# Build the sample
cmake --build . --target CrossGamma

# Run the executable directly
./samples/CrossGamma/CrossGamma

# Run via CTest
ctest -R sample_CrossGamma --output-on-failure
```

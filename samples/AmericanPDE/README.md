# American Option PDE Sample

Prices an American put option using a finite-difference solver under a local
volatility surface $\sigma(S, t)$, and computes the sensitivities to every point
on the volatility grid in a single adjoint pass.

The sample implements:
- Crank-Nicolson discretization with Rannacher smoothing (implicit Euler initial half-steps).
- Early exercise constraint evaluated with `xad::max()`.
- Tridiagonal matrix solve (Thomas algorithm) wrapped as an external function with
  a hand-written adjoint (solving the transposed system $A^T \lambda = \bar{x}$).
- Comparison of memory and runtime between the external function callback and
  recording the solve operation-by-operation.
- Sensitivity validation against finite-difference bumping across grid points.
- Benchmark comparing the cost of computing all sensitivities in one adjoint pass
  against finite-difference bumping.

---

## Mathematical Formulation

### Black-Scholes PDE with Local Volatility

Under a local volatility model $\sigma(S, t)$, the option value $V(S, t)$ satisfies:

$$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2(S, t) S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

Transforming to time-to-maturity $\tau = T - t$ with $\tau \in [0, T]$ gives:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2} \sigma^2(S, T - \tau) S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

For an American put option with strike $K$:
- **Terminal condition** (at $\tau = 0$): $V(S, 0) = \max(K - S, 0)$.
- **Boundary conditions**: $V(0, \tau) = K$ and $V(S_{\max}, \tau) = 0$.
- **Early exercise**: At every time step, $V(S, \tau) \ge \max(K - S, 0)$.

### Spatial & Time Discretization

The spatial domain $[0, S_{\max}]$ is discretized with $N_S = 200$ points ($S_i = i \Delta S$).
At interior points $i \in [1, N_S - 2]$, standard central differences yield:

$$\mathcal{L} V_i = \alpha_i V_{i-1} + \beta_i V_i + \gamma_i V_{i+1}$$

where:

$$\alpha_i = \frac{\sigma_i^2 S_i^2}{2 \Delta S^2} - \frac{r S_i}{2 \Delta S}, \quad \beta_i = -\frac{\sigma_i^2 S_i^2}{\Delta S^2} - r, \quad \gamma_i = \frac{\sigma_i^2 S_i^2}{2 \Delta S^2} + \frac{r S_i}{2 \Delta S}$$

### Rannacher Smoothing & Crank-Nicolson

The initial payoff derivative is discontinuous at the strike $K$, which can trigger
high-frequency oscillations in standard Crank-Nicolson stepping. To suppress this,
the solver applies **Rannacher smoothing**:
- The first 2 time steps are divided into 4 half-steps of size $\Delta \tau / 2$
  solved with fully implicit Euler ($\theta = 1.0$):
  $$(I - \delta \tau \mathcal{L}^{m+1}) V^{m+1} = V^m$$
- The remaining $N_t - 2$ time steps use Crank-Nicolson ($\theta = 0.5$):
  $$(I - \frac{\Delta \tau}{2} \mathcal{L}^{m+1}) V^{m+1} = (I + \frac{\Delta \tau}{2} \mathcal{L}^m) V^m$$

After each solve, early exercise is applied using `xad::max()`:

$$V_i^{m+1} = \operatorname{xad::max}(V_i^{m+1}, K - S_i)$$

Both `std::max()` and `xad::max()` propagate derivatives through active arguments.
What `xad::max()` adds is a defined derivative at a tie ($V_i = K - S_i$), splitting
the derivative equally (weight 0.5) between both branches rather than arbitrarily
taking one.

---

## Tridiagonal Solver & Hand-Written Adjoint

Each time step requires solving a tridiagonal linear system $A x = d$.

### Matrix Differentiation

Differentiating $A x = d$ gives:

$$A \, \mathrm{d}x + (\mathrm{d}A) \, x = \mathrm{d}d \implies \mathrm{d}x = A^{-1}(\mathrm{d}d - (\mathrm{d}A) x)$$

For an output sensitivity vector $\bar{x} = \frac{\partial y}{\partial x}$, the adjoint relation is:

$$\bar{x}^T \mathrm{d}x = \bar{x}^T A^{-1}(\mathrm{d}d - (\mathrm{d}A) x) = (A^{-T} \bar{x})^T (\mathrm{d}d - (\mathrm{d}A) x)$$

Defining the adjoint state $\lambda$ as the solution to the transposed system:

$$A^T \lambda = \bar{x}$$

The contributions to the input adjoints are:
- $\bar{d}_i += \lambda_i$
- $\bar{b}_i += -\lambda_i x_i$ (main diagonal)
- $\bar{a}_i += -\lambda_{i+1} x_i$ (subdiagonal)
- $\bar{c}_i += -\lambda_i x_{i+1}$ (superdiagonal)

Since $A$ is tridiagonal, its transpose $A^T$ is also tridiagonal, with the subdiagonal
and superdiagonal swapped. Solving $A^T \lambda = \bar{x}$ is performed directly
with the Thomas algorithm in $O(M)$ operations without matrix inversion.

### External Function Implementation

Wrapping the tridiagonal solve in `TridiagonalSolveCallback<Tape>` (inheriting from
`xad::CheckpointCallback`):
1. In the forward pass, passive values are solved and recorded outputs are registered.
2. In the reverse pass, `computeAdjoint` solves $A^T \lambda = \bar{x}$ and propagates
   adjoints directly to inputs using `tape->incrementAdjoint(...)`.

This avoids recording about 80,000 intermediate arithmetic operations of the linear
solver on the tape across the 102 steps, saving ~2.7 MiB (~24%) of total memory
(tape + callback storage) while producing identical sensitivities to machine
precision ($\approx 10^{-14}$).

---

## Local Volatility Grid & Sensitivities

The local volatility surface is discretized on a $10 \times 10$ parameter grid
($S_k \in [0, S_{\max}]$, $t_l \in [0, T]$). Volatilities at arbitrary grid points $(S, t)$
are evaluated via bilinear interpolation across the 4 surrounding knots.

A single reverse AD pass computes $\frac{\partial V}{\partial \sigma_{k, l}}$ for all
100 parameters simultaneously.

---

## Validation & Verification Checks

1. **Tape Consistency**: Sensitivities from the external function callback match those
   from operation-by-operation recording to within $10^{-11}$ (observed diff $\approx 1.9 \times 10^{-14}$).
2. **Finite-Difference Bumping**: Sensitivities at representative grid knots agree with
   two-sided bumping ($h = 10^{-5}$) to within $10^{-4}$ (observed diff $\approx 5.8 \times 10^{-10}$).
3. **Memory Reduction**: The external function reduces total memory (tape plus callback
   storage) compared to operation-by-operation recording.
4. **Automated Testing**: Any failure in the above checks returns a non-zero exit code.

---

## Build and Run

From the repository root:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target AmericanPDE
build/samples/AmericanPDE/AmericanPDE
```

Or via `ctest`:

```bash
ctest --test-dir build -R sample_AmericanPDE --output-on-failure
```

---

## Sample Output

```text
========================================================================
 American Option PDE Solver: Local Volatility Sensitivities via Adjoint
========================================================================

Grid: 200 price points, 100 time steps (2 Rannacher smoothing steps)
Local volatility grid: 10 x 10 = 100 parameters
Parameters: S0 = 100, K = 100, T = 1, r = 0.05

Option Price (passive PDE): 6.086968
Passive pricing time:       1.763690 ms

--- Solver Comparison: External Function vs. Operation-by-Operation ---
Metric                        External Function     Operation-by-Operation
--------------------------------------------------------------------------
Tape Memory (KiB)             7785.972656           11553.281250          
Callbacks Storage (KiB)       1045.500000           0.000000              
Total Memory (KiB)            8831.472656           11553.281250          
Record + adjoint time (ms)    16.593292             11.213576             
Total Memory Ratio            1.00x                 1.308194              

Max sensitivity difference: 1.91e-14

--- Finite Difference Bumping Validation (step h = 1e-5) ---
Knot (k,l)  S_knot        t_knot        Adjoint dV/dSigma FD dV/dSigma      Abs. Error    
------------------------------------------------------------------------------------------
(1,1)      33.3          0.1           0.000000          0.000000          2.16e-21      
(2,5)      66.7          0.6           0.318012          0.318012          2.06e-11      
(3,2)      100.0         0.2           3.982293          3.982293          5.84e-10      
(3,5)      100.0         0.6           3.192619          3.192619          3.06e-10      
(4,3)      133.3         0.3           0.538035          0.538035          6.24e-11      
(5,5)      166.7         0.6           0.000938          0.000938          7.96e-11      
(7,2)      233.3         0.2           0.000000          0.000000          3.17e-15      
(8,7)      266.7         0.8           0.000000          0.000000          8.77e-21      

Max finite difference error: 5.84e-10

--- Performance Cost Analysis ---
1 Passive PDE pricing:              1.76 ms  (1.00x)
All 100 sensitivities via Adjoint:  16.59 ms  (9.41x pricing)
All 100 sensitivities via Bumping:  116.84 ms  (66.25x pricing)
Adjoint speedup vs. full bumping:   7.04x faster
Note: Timings and exact ratios depend on the machine, compiler, and CPU load.

AmericanPDE sample validation PASSED successfully.
```

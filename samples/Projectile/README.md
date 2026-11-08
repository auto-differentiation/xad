# Projectile Aiming Sample

Simulates 2D projectile motion under gravity and quadratic air drag using a
fixed-step integrator, and optimizes the initial launch angle and speed using
XAD's adjoint (reverse) mode to hit a target point.

The sample validates the adjoint gradients against central finite differences,
runs gradient descent to iteratively steer the projectile toward the target,
and exports the trajectories across optimization iterations to an SVG file
(`trajectory.svg`).

## Problem Formulation

The projectile has position $(x, y)$ and velocity $(v_x, v_y)$. Starting from
the origin $(0, 0)$ with speed $v_0$ and launch angle $\theta$:

    v_x(0) = v_0 * cos(theta)
    v_y(0) = v_0 * sin(theta)

At each step, motion is subject to gravitational acceleration $g = 9.81\,\text{m/s}^2$
and quadratic aerodynamic drag with coefficient $k$:

    speed = sqrt(v_x^2 + v_y^2)
    a_x   = -k * speed * v_x
    a_y   = -g - k * speed * v_y

The simulation integrates these equations using a fixed-step integrator over
$N = 1,000$ steps with time step $\Delta t = 0.004\,\text{s}$ (total duration $4.0\,\text{s}$).

The objective is to reach target $(x_{\text{target}}, y_{\text{target}})$. The loss is the
squared Euclidean distance at the final step:

    loss = (x_N - x_target)^2 + (y_N - y_target)^2

## Differentiating Through the Simulation

The simulation is templated so it evaluates both with passive `double` and with
XAD's active adjoint type `xad::adj<double>`.

During the forward simulation, XAD records all 1,000 integration steps onto the
tape. Seeding the output derivative `derivative(loss) = 1.0` and calling
`tape.computeAdjoints()` propagates sensitivities backward through all 1,000
steps in **a single reverse pass**, yielding:

    dL/dv0     = derivative(v0)
    dL/dtheta  = derivative(theta)

## Finite-Difference Check

At the initial configuration ($v_0 = 45\,\text{m/s}$, $\theta = 40^\circ$), the sample
estimates both gradients using central finite differences with perturbation $h = 10^{-6}$:

    fd_grad = (loss(param + h) - loss(param - h)) / (2 * h)
    rel_diff = abs(adj_grad - fd_grad) / (abs(adj_grad) + 1e-15)

The sample exits with a non-zero exit code if either gradient is non-finite or if
the relative difference exceeds $10^{-4}$, allowing `ctest` to catch regressions.

## Trajectory Visualization

During gradient descent, trajectories from selected iterations ($0, 5, 10, 15, 20, 24$)
are recorded and exported to `trajectory.svg`. The SVG visually displays the target,
ground line, and how each iteration adjusts launch velocity and angle to home in on
the target.

## Build and Run

From the repository root:

    cmake -S . -B build
    cmake --build build --target Projectile
    build/samples/Projectile/Projectile

Or run via `ctest`:

    ctest --test-dir build -R sample_Projectile --output-on-failure

## Expected Output

    === XAD Projectile Aiming Optimization ===
    Simulation: fixed-step integrator, 1000 steps, dt = 0.004s
    Target: (80, 0)
    Initial guess: v0 = 45 m/s, theta = 40 deg

    Initial loss = 96.960958
    Adjoint gradients:
      dL/dv0     =    28.067373
      dL/dtheta  =  1275.624377

    Finite-difference check:
         param    adjoint grad         FD grad       rel. diff
            v0       28.067373       28.067373       3.234e-09
         theta     1275.624377     1275.624377       2.776e-12

    Finite-difference validation passed successfully.

    Running gradient descent optimization (25 iterations):
    Iter  0: loss =  96.9610, final pos = (82.6458, 9.4848), v0 = 45.0000 m/s, theta = 40.0000 deg
    Iter  5: loss =   2.0343, final pos = (81.2703, 0.6485), v0 = 41.3691 m/s, theta = 36.7664 deg
    Iter 10: loss =   0.0692, final pos = (80.2345, 0.1192), v0 = 40.6890 m/s, theta = 36.9475 deg
    Iter 15: loss =   0.0022, final pos = (80.0422, 0.0214), v0 = 40.5638 m/s, theta = 36.9818 deg
    Iter 20: loss =   0.0001, final pos = (80.0076, 0.0038), v0 = 40.5413 m/s, theta = 36.9880 deg
    Iter 24: loss =   0.0000, final pos = (80.0019, 0.0010), v0 = 40.5376 m/s, theta = 36.9890 deg

    Exported trajectory visualization to trajectory.svg

    Sample completed successfully.

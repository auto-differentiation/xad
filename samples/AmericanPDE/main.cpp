/*******************************************************************************

   Prices an American put option using a Crank-Nicolson finite-difference solver
   under a local volatility surface, and computes sensitivities to all grid points
   in a single adjoint pass.

   Features:
   - Crank-Nicolson discretization with Rannacher smoothing (implicit Euler half-steps).
   - Early exercise constraint handled via xad::max().
   - Tridiagonal solver wrapped as an external function with a hand-written adjoint
     (transposed matrix solve A^T lambda = x_bar).
   - Comparison of tape memory and runtime: external function vs. operation-by-operation.
   - Validation against two-sided finite-difference bumping.
   - Reporting of adjoint speedup relative to bumping every parameter.


   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include <XAD/XAD.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

typedef xad::adj<double> mode;
typedef mode::tape_type tape_type;
typedef mode::active_type AD;

// American put problem parameters
const double SPOT = 100.0;
const double STRIKE = 100.0;
const double MATURITY = 1.0;
const double RATE = 0.05;
const double S_MAX = 300.0;

// Finite difference grid dimensions
const int N_S = 200;        // spatial price grid points
const int N_T = 100;        // time steps
const int N_RANNACHER = 2;  // first 2 time steps split into 4 implicit half-steps

// Local volatility surface grid dimensions (10 x 10 = 100 parameters)
const int VOL_GRID_S = 10;
const int VOL_GRID_T = 10;
const double BASE_VOL = 0.20;

enum SolverMode
{
    ModeExternalFunction,
    ModeOperationByOperation
};

// -----------------------------------------------------------------------------
// Local Volatility Surface Grid with Bilinear Interpolation
// -----------------------------------------------------------------------------
template <typename T>
struct LocalVolGrid
{
    int nS;
    int nT;
    double S_max;
    double T_max;
    std::vector<T> vol;  // size nS * nT, row-major: index = k * nT + l

    T getVol(double S, double t) const
    {
        double s_clamped = std::min(std::max(0.0, S), S_max);
        double t_clamped = std::min(std::max(0.0, t), T_max);

        double dS = S_max / (nS - 1);
        double dt = T_max / (nT - 1);

        int k = static_cast<int>(s_clamped / dS);
        if (k >= nS - 1)
            k = nS - 2;
        int l = static_cast<int>(t_clamped / dt);
        if (l >= nT - 1)
            l = nT - 2;

        double u = (s_clamped - k * dS) / dS;
        double v = (t_clamped - l * dt) / dt;

        const T& v00 = vol[k * nT + l];
        const T& v10 = vol[(k + 1) * nT + l];
        const T& v01 = vol[k * nT + (l + 1)];
        const T& v11 = vol[(k + 1) * nT + (l + 1)];

        return (1.0 - u) * (1.0 - v) * v00 + u * (1.0 - v) * v10 + (1.0 - u) * v * v01 +
               u * v * v11;
    }
};

// -----------------------------------------------------------------------------
// Passive Tridiagonal Solver (Thomas Algorithm)
// Solves A * x = d where:
// - sub has size M - 1: sub[i] is element (i+1, i)
// - diag has size M: diag[i] is element (i, i)
// - sup has size M - 1: sup[i] is element (i, i+1)
// - rhs has size M
// -----------------------------------------------------------------------------
inline void solveTridiagonalPassive(const std::vector<double>& sub, const std::vector<double>& diag,
                                    const std::vector<double>& sup, const std::vector<double>& rhs,
                                    std::vector<double>& sol)
{
    const std::size_t M = diag.size();
    sol.resize(M);
    if (M == 0)
        return;
    if (M == 1)
    {
        sol[0] = rhs[0] / diag[0];
        return;
    }

    std::vector<double> c_prime(M - 1);
    std::vector<double> d_prime(M);

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (std::size_t i = 1; i < M; ++i)
    {
        double denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if (i < M - 1)
            c_prime[i] = sup[i] / denom;
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    sol[M - 1] = d_prime[M - 1];
    for (std::size_t i = M - 1; i-- > 0;)
    {
        sol[i] = d_prime[i] - c_prime[i] * sol[i + 1];
    }
}

// -----------------------------------------------------------------------------
// Active Tridiagonal Solver: Operation-by-Operation
// Records every arithmetic step of the Thomas algorithm onto the tape.
// -----------------------------------------------------------------------------
inline void solveTridiagonalActive(const std::vector<AD>& sub, const std::vector<AD>& diag,
                                   const std::vector<AD>& sup, const std::vector<AD>& rhs,
                                   std::vector<AD>& sol)
{
    const std::size_t M = diag.size();
    sol.resize(M);
    if (M == 0)
        return;
    if (M == 1)
    {
        sol[0] = rhs[0] / diag[0];
        return;
    }

    std::vector<AD> c_prime(M - 1);
    std::vector<AD> d_prime(M);

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (std::size_t i = 1; i < M; ++i)
    {
        AD denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if (i < M - 1)
            c_prime[i] = sup[i] / denom;
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    sol[M - 1] = d_prime[M - 1];
    for (std::size_t i = M - 1; i-- > 0;)
    {
        sol[i] = d_prime[i] - c_prime[i] * sol[i + 1];
    }
}

// -----------------------------------------------------------------------------
// External Function Callback with Hand-Written Adjoint
// In the reverse pass, solves transposed system A^T * lambda = sol_adj using
// the Thomas algorithm (swapping sub and sup diagonals).
// -----------------------------------------------------------------------------
template <class Tape>
class TridiagonalSolveCallback : public xad::CheckpointCallback<Tape>
{
  public:
    typedef typename Tape::slot_type slot_type;
    typedef typename Tape::value_type value_type;
    typedef typename Tape::active_type active_type;

    void compute(const std::vector<active_type>& sub, const std::vector<active_type>& diag,
                 const std::vector<active_type>& sup, const std::vector<active_type>& rhs,
                 std::vector<active_type>& sol)
    {
        const std::size_t M = diag.size();
        sub_slots_.resize(sub.size());
        for (std::size_t i = 0; i < sub.size(); ++i) sub_slots_[i] = sub[i].getSlot();
        diag_slots_.resize(M);
        for (std::size_t i = 0; i < M; ++i) diag_slots_[i] = diag[i].getSlot();
        sup_slots_.resize(sup.size());
        for (std::size_t i = 0; i < sup.size(); ++i) sup_slots_[i] = sup[i].getSlot();
        rhs_slots_.resize(M);
        for (std::size_t i = 0; i < M; ++i) rhs_slots_[i] = rhs[i].getSlot();

        sub_val_.resize(sub.size());
        for (std::size_t i = 0; i < sub.size(); ++i) sub_val_[i] = value(sub[i]);
        diag_val_.resize(M);
        for (std::size_t i = 0; i < M; ++i) diag_val_[i] = value(diag[i]);
        sup_val_.resize(sup.size());
        for (std::size_t i = 0; i < sup.size(); ++i) sup_val_[i] = value(sup[i]);
        rhs_val_.resize(M);
        for (std::size_t i = 0; i < M; ++i) rhs_val_[i] = value(rhs[i]);

        sol_val_.resize(M);
        solveTridiagonalPassive(sub_val_, diag_val_, sup_val_, rhs_val_, sol_val_);

        sol.resize(M);
        sol_slots_.resize(M);
        Tape* tape = Tape::getActive();
        for (std::size_t i = 0; i < M; ++i)
        {
            sol[i] = sol_val_[i];
            tape->registerOutput(sol[i]);
            sol_slots_[i] = sol[i].getSlot();
        }

        tape->insertCallback(this);
    }

    void computeAdjoint(Tape* tape) override
    {
        const std::size_t M = diag_slots_.size();
        std::vector<value_type> sol_adj(M);
        for (std::size_t i = 0; i < M; ++i)
            sol_adj[i] = tape->getAndResetOutputAdjoint(sol_slots_[i]);

        // Solve A^T * lambda = sol_adj
        // Subdiagonal of A^T is sup_val_, superdiagonal is sub_val_
        std::vector<value_type> lambda(M);
        solveTridiagonalPassive(sup_val_, diag_val_, sub_val_, sol_adj, lambda);

        // Accumulate adjoint contributions to inputs:
        // rhs: lambda_i
        // diag: -lambda_i * sol_i
        for (std::size_t i = 0; i < M; ++i)
        {
            if (rhs_slots_[i] != Tape::INVALID_SLOT)
                tape->incrementAdjoint(rhs_slots_[i], lambda[i]);
            if (diag_slots_[i] != Tape::INVALID_SLOT)
                tape->incrementAdjoint(diag_slots_[i], -lambda[i] * sol_val_[i]);
        }
        // sub (row i+1, col i): -lambda_{i+1} * sol_i
        for (std::size_t i = 0; i < sub_slots_.size(); ++i)
        {
            if (sub_slots_[i] != Tape::INVALID_SLOT)
                tape->incrementAdjoint(sub_slots_[i], -lambda[i + 1] * sol_val_[i]);
        }
        // sup (row i, col i+1): -lambda_i * sol_{i+1}
        for (std::size_t i = 0; i < sup_slots_.size(); ++i)
        {
            if (sup_slots_[i] != Tape::INVALID_SLOT)
                tape->incrementAdjoint(sup_slots_[i], -lambda[i] * sol_val_[i + 1]);
        }
    }

  private:
    std::vector<slot_type> sub_slots_, diag_slots_, sup_slots_, rhs_slots_, sol_slots_;
    std::vector<value_type> sub_val_, diag_val_, sup_val_, rhs_val_, sol_val_;
};

inline void solveTridiagonalExternal(const std::vector<AD>& sub, const std::vector<AD>& diag,
                                     const std::vector<AD>& sup, const std::vector<AD>& rhs,
                                     std::vector<AD>& sol)
{
    tape_type* tape = tape_type::getActive();
    TridiagonalSolveCallback<tape_type>* cb = new TridiagonalSolveCallback<tape_type>;
    tape->pushCallback(cb);
    cb->compute(sub, diag, sup, rhs, sol);
}

// Solver dispatch helpers
inline void solveStep(const std::vector<double>& sub, const std::vector<double>& diag,
                      const std::vector<double>& sup, const std::vector<double>& rhs,
                      std::vector<double>& sol, SolverMode /*mode*/)
{
    solveTridiagonalPassive(sub, diag, sup, rhs, sol);
}

inline void solveStep(const std::vector<AD>& sub, const std::vector<AD>& diag,
                      const std::vector<AD>& sup, const std::vector<AD>& rhs, std::vector<AD>& sol,
                      SolverMode mode)
{
    if (mode == ModeExternalFunction)
        solveTridiagonalExternal(sub, diag, sup, rhs, sol);
    else
        solveTridiagonalActive(sub, diag, sup, rhs, sol);
}

// -----------------------------------------------------------------------------
// American Option PDE Pricer
// -----------------------------------------------------------------------------
template <typename T>
T priceAmericanPut(const LocalVolGrid<T>& volGrid, SolverMode mode = ModeExternalFunction)
{
    const double dS = S_MAX / (N_S - 1);
    const double dt = MATURITY / N_T;

    std::vector<double> S(N_S);
    for (int i = 0; i < N_S; ++i) S[i] = i * dS;

    // Terminal payoff at maturity (tau = 0): V(S, 0) = max(K - S, 0)
    std::vector<T> V(N_S);
    for (int i = 0; i < N_S; ++i)
    {
        V[i] = (STRIKE > S[i]) ? T(STRIKE - S[i]) : T(0.0);
    }

    const std::size_t M = N_S - 2;  // number of interior points
    std::vector<T> sub(M - 1), diag(M), sup(M - 1), rhs(M), sol(M);

    double tau = 0.0;
    const int total_substeps = N_RANNACHER * 2 + (N_T - N_RANNACHER);

    for (int step = 0; step < total_substeps; ++step)
    {
        const double dtau = (step < N_RANNACHER * 2) ? (dt / 2.0) : dt;
        const double theta = (step < N_RANNACHER * 2) ? 1.0 : 0.5;

        const double tau_old = tau;
        const double tau_new = tau + dtau;
        tau = tau_new;

        const double t_old = MATURITY - tau_old;
        const double t_new = MATURITY - tau_new;

        // Build LHS tridiagonal matrix (I - theta * dtau * L_new)
        for (std::size_t j = 0; j < M; ++j)
        {
            int i = static_cast<int>(j + 1);
            T vol_new = volGrid.getVol(S[i], t_new);
            T vol2_S2 = vol_new * vol_new * (S[i] * S[i]);
            T beta_new = -vol2_S2 / (dS * dS) - RATE;
            diag[j] = 1.0 - theta * dtau * beta_new;

            if (j < M - 1)
            {
                T gamma_new = 0.5 * vol2_S2 / (dS * dS) + 0.5 * RATE * S[i] / dS;
                sup[j] = -theta * dtau * gamma_new;

                int i_next = static_cast<int>(j + 2);
                T vol_next = volGrid.getVol(S[i_next], t_new);
                T vol2_S2_next = vol_next * vol_next * (S[i_next] * S[i_next]);
                T alpha_next = 0.5 * vol2_S2_next / (dS * dS) - 0.5 * RATE * S[i_next] / dS;
                sub[j] = -theta * dtau * alpha_next;
            }
        }

        // Build RHS vector (I + (1 - theta) * dtau * L_old) * V_old
        for (std::size_t j = 0; j < M; ++j)
        {
            int i = static_cast<int>(j + 1);
            if (theta == 1.0)
            {
                rhs[j] = V[i];
            }
            else
            {
                T vol_old = volGrid.getVol(S[i], t_old);
                T vol2_S2 = vol_old * vol_old * (S[i] * S[i]);
                T alpha_old = 0.5 * vol2_S2 / (dS * dS) - 0.5 * RATE * S[i] / dS;
                T beta_old = -vol2_S2 / (dS * dS) - RATE;
                T gamma_old = 0.5 * vol2_S2 / (dS * dS) + 0.5 * RATE * S[i] / dS;

                rhs[j] = (1.0 - theta) * dtau * alpha_old * V[i - 1] +
                         (1.0 + (1.0 - theta) * dtau * beta_old) * V[i] +
                         (1.0 - theta) * dtau * gamma_old * V[i + 1];
            }
        }

        // Boundary condition at S = 0: V(0, tau) = STRIKE
        T vol_1 = volGrid.getVol(S[1], t_new);
        T vol2_S2_1 = vol_1 * vol_1 * (S[1] * S[1]);
        T alpha_1 = 0.5 * vol2_S2_1 / (dS * dS) - 0.5 * RATE * S[1] / dS;
        rhs[0] += theta * dtau * alpha_1 * T(STRIKE);

        // Solve tridiagonal system
        solveStep(sub, diag, sup, rhs, sol, mode);

        // Apply early exercise constraint using xad::max()
        using std::max;
        using xad::max;
        for (std::size_t j = 0; j < M; ++j)
        {
            int i = static_cast<int>(j + 1);
            T exercise = (STRIKE > S[i]) ? T(STRIKE - S[i]) : T(0.0);
            V[i] = max(sol[j], exercise);
        }
        V[0] = T(STRIKE);
        V[N_S - 1] = T(0.0);
    }

    // Linearly interpolate V at SPOT
    int idx = static_cast<int>(SPOT / dS);
    if (idx >= N_S - 1)
        idx = N_S - 2;
    double w = (SPOT - S[idx]) / dS;
    return (1.0 - w) * V[idx] + w * V[idx + 1];
}

// -----------------------------------------------------------------------------
// Main Function: Comparisons, Validation, and Benchmarks
// -----------------------------------------------------------------------------
int main()
{
    std::cout << "========================================================================\n"
              << " American Option PDE Solver: Local Volatility Sensitivities via Adjoint\n"
              << "========================================================================\n\n";

    std::cout << "Grid: " << N_S << " price points, " << N_T << " time steps (" << N_RANNACHER
              << " Rannacher smoothing steps)\n"
              << "Local volatility grid: " << VOL_GRID_S << " x " << VOL_GRID_T << " = "
              << (VOL_GRID_S * VOL_GRID_T) << " parameters\n"
              << "Parameters: S0 = " << SPOT << ", K = " << STRIKE << ", T = " << MATURITY
              << ", r = " << RATE << "\n\n";

    // Setup base volatility surface
    const int num_params = VOL_GRID_S * VOL_GRID_T;
    std::vector<double> base_vols(num_params, BASE_VOL);

    LocalVolGrid<double> passive_grid{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, base_vols};

    // 1. Passive pricing baseline
    auto t_start = std::chrono::high_resolution_clock::now();
    double price_passive = priceAmericanPut(passive_grid);
    auto t_end = std::chrono::high_resolution_clock::now();
    double t_price_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Option Price (passive PDE): " << price_passive << "\n";
    std::cout << "Passive pricing time:       " << t_price_ms << " ms\n\n";

    // 2. Adjoint pass using External Function
    std::vector<double> grad_external(num_params);
    std::size_t mem_external = 0;
    double t_ad_external_ms = 0.0;
    double price_external = 0.0;
    {
        tape_type tape;
        std::vector<AD> vols_ad(num_params);
        for (int i = 0; i < num_params; ++i) vols_ad[i] = base_vols[i];
        tape.registerInputs(vols_ad);
        tape.newRecording();

        t_start = std::chrono::high_resolution_clock::now();
        LocalVolGrid<AD> ad_grid{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, vols_ad};
        AD price_ad = priceAmericanPut(ad_grid, ModeExternalFunction);

        tape.registerOutput(price_ad);
        derivative(price_ad) = 1.0;
        mem_external = tape.getMemory();
        tape.computeAdjoints();
        t_end = std::chrono::high_resolution_clock::now();

        t_ad_external_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        price_external = value(price_ad);
        for (int i = 0; i < num_params; ++i) grad_external[i] = derivative(vols_ad[i]);
    }

    // 3. Adjoint pass using Operation-by-Operation
    std::vector<double> grad_opbyop(num_params);
    std::size_t mem_opbyop = 0;
    double t_ad_opbyop_ms = 0.0;
    double price_opbyop = 0.0;
    {
        tape_type tape;
        std::vector<AD> vols_ad(num_params);
        for (int i = 0; i < num_params; ++i) vols_ad[i] = base_vols[i];
        tape.registerInputs(vols_ad);
        tape.newRecording();

        t_start = std::chrono::high_resolution_clock::now();
        LocalVolGrid<AD> ad_grid{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, vols_ad};
        AD price_ad = priceAmericanPut(ad_grid, ModeOperationByOperation);

        tape.registerOutput(price_ad);
        derivative(price_ad) = 1.0;
        mem_opbyop = tape.getMemory();
        tape.computeAdjoints();
        t_end = std::chrono::high_resolution_clock::now();

        t_ad_opbyop_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        price_opbyop = value(price_ad);
        for (int i = 0; i < num_params; ++i) grad_opbyop[i] = derivative(vols_ad[i]);
    }

    // Compare external function vs. op-by-op
    double max_diff_ext_op = 0.0;
    for (int i = 0; i < num_params; ++i)
    {
        max_diff_ext_op = std::max(max_diff_ext_op, std::abs(grad_external[i] - grad_opbyop[i]));
    }

    std::cout << "--- Solver Comparison: External Function vs. Operation-by-Operation ---\n";
    std::cout << std::left << std::setw(28) << "Metric" << std::setw(20) << "External Function"
              << std::setw(20) << "Operation-by-Operation" << "\n";
    std::cout << std::string(68, '-') << "\n";
    std::cout << std::setw(28) << "Tape Memory (KiB)" << std::setw(20) << (mem_external / 1024.0)
              << std::setw(20) << (mem_opbyop / 1024.0) << "\n";
    std::cout << std::setw(28) << "Adjoint Pass Time (ms)" << std::setw(20) << t_ad_external_ms
              << std::setw(20) << t_ad_opbyop_ms << "\n";
    std::cout << std::setw(28) << "Tape Memory Ratio" << std::setw(20) << "1.00x" << std::setw(20)
              << (double(mem_opbyop) / mem_external) << "\n\n";

    std::cout << "Max sensitivity difference: " << std::scientific << std::setprecision(2)
              << max_diff_ext_op << "\n\n";

    // 4. Finite-difference bumping check for selected grid knots
    const double FD_H = 1e-5;
    const double FD_TOL = 1e-4;
    const std::vector<std::pair<int, int>> check_knots = {{1, 1}, {2, 5}, {3, 2}, {3, 5},
                                                          {4, 3}, {5, 5}, {7, 2}, {8, 7}};

    std::cout << "--- Finite Difference Bumping Validation (step h = 1e-5) ---\n";
    std::cout << std::left << std::setw(12) << "Knot (k,l)" << std::setw(14) << "S_knot"
              << std::setw(14) << "t_knot" << std::setw(18) << "Adjoint dV/dSigma" << std::setw(18)
              << "FD dV/dSigma" << std::setw(14) << "Abs. Error" << "\n";
    std::cout << std::string(90, '-') << "\n";

    double max_fd_error = 0.0;
    for (const auto& knot : check_knots)
    {
        int k = knot.first;
        int l = knot.second;
        int p = k * VOL_GRID_T + l;
        double s_val = k * S_MAX / (VOL_GRID_S - 1);
        double t_val = l * MATURITY / (VOL_GRID_T - 1);

        std::vector<double> bumped_up = base_vols;
        std::vector<double> bumped_dn = base_vols;
        bumped_up[p] += FD_H;
        bumped_dn[p] -= FD_H;

        LocalVolGrid<double> g_up{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, bumped_up};
        LocalVolGrid<double> g_dn{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, bumped_dn};

        double p_up = priceAmericanPut(g_up);
        double p_dn = priceAmericanPut(g_dn);
        double fd_sens = (p_up - p_dn) / (2.0 * FD_H);
        double err = std::abs(grad_external[p] - fd_sens);
        max_fd_error = std::max(max_fd_error, err);

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "(" << k << "," << l << ")" << std::string(6, ' ') << std::setw(14) << s_val
                  << std::setw(14) << t_val << std::setprecision(6) << std::setw(18)
                  << grad_external[p] << std::setw(18) << fd_sens << std::scientific
                  << std::setprecision(2) << std::setw(14) << err << "\n";
    }
    std::cout << "\nMax finite difference error: " << std::scientific << std::setprecision(2)
              << max_fd_error << "\n\n";

    // 5. Cost comparison: Adjoint pass vs. bumping all 100 parameters
    t_start = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < num_params; ++p)
    {
        std::vector<double> bumped_up = base_vols;
        std::vector<double> bumped_dn = base_vols;
        bumped_up[p] += FD_H;
        bumped_dn[p] -= FD_H;
        LocalVolGrid<double> g_up{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, bumped_up};
        LocalVolGrid<double> g_dn{VOL_GRID_S, VOL_GRID_T, S_MAX, MATURITY, bumped_dn};
        priceAmericanPut(g_up);
        priceAmericanPut(g_dn);
    }
    t_end = std::chrono::high_resolution_clock::now();
    double t_bump_all_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "--- Performance Cost Analysis ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "1 Passive PDE pricing:              " << t_price_ms << " ms  (1.00x)\n";
    std::cout << "All 100 sensitivities via Adjoint:  " << t_ad_external_ms << " ms  ("
              << (t_ad_external_ms / t_price_ms) << "x pricing)\n";
    std::cout << "All 100 sensitivities via Bumping:  " << t_bump_all_ms << " ms  ("
              << (t_bump_all_ms / t_price_ms) << "x pricing)\n";
    std::cout << "Adjoint speedup vs. full bumping:   " << (t_bump_all_ms / t_ad_external_ms)
              << "x faster\n\n";

    // Validation checks
    bool success = true;
    if (std::abs(price_external - price_passive) > 1e-12)
    {
        std::cerr << "FAIL: AD price does not match passive price!\n";
        success = false;
    }
    if (max_diff_ext_op > 1e-11)
    {
        std::cerr << "FAIL: External function sensitivities differ from op-by-op!\n";
        success = false;
    }
    if (max_fd_error > FD_TOL)
    {
        std::cerr << "FAIL: Sensitivities do not match finite difference bumping!\n";
        success = false;
    }
    if (mem_external >= mem_opbyop)
    {
        std::cerr << "FAIL: External function did not reduce tape memory!\n";
        success = false;
    }

    if (!success)
    {
        std::cerr << "AmericanPDE sample validation failed.\n";
        return 1;
    }

    std::cout << "AmericanPDE sample validation PASSED successfully.\n";
    return 0;
}

/*******************************************************************************

   Explicit finite-difference solver for the 1D heat equation, and a
   checkpoint callback that records it in stages so that the tape only holds
   one stage at a time during the adjoint pass.

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

#pragma once

#include <XAD/XAD.hpp>

#include <algorithm>
#include <cstddef>
#include <stack>
#include <vector>

// Advances the temperature profile u by nSteps explicit time steps of
//   u_i <- u_i + lambda * (u_{i+1} - 2 u_i + u_{i-1}),   lambda = alpha * dt / dx^2,
// keeping both end points fixed (Dirichlet boundary conditions).
// The scheme is stable for lambda <= 0.5.
template <class T>
void heatSteps(std::vector<T>& u, double lambda, int nSteps)
{
    std::vector<T> next(u);
    for (int step = 0; step < nSteps; ++step)
    {
        for (std::size_t i = 1; i + 1 < u.size(); ++i)
            next[i] = u[i] + lambda * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        u.swap(next);
    }
}

/**
 * Checkpoint callback for a stage of heatSteps.
 *
 * In the forward pass, computeStage runs a stage passively, so nothing of it is recorded,
 * and only stores the profile at its start. In the adjoint pass, XAD calls computeAdjoint
 * for the stages in reverse order; it re-runs the stage from the checkpoint actively in a
 * nested recording and propagates the adjoints of the stage outputs to its inputs.
 * The tape memory is then bounded by one stage, instead of growing with all time steps.
 */
template <class Tape>
class HeatCheckpointCallback : public xad::CheckpointCallback<Tape>
{
  public:
    typedef typename Tape::slot_type slot_type;      // type for slot in the tape
    typedef typename Tape::value_type value_type;    // double
    typedef typename Tape::active_type active_type;  // AReal<double>

    explicit HeatCheckpointCallback(double lambda) : lambda_(lambda), peakMemory_(0) {}

    // Runs nSteps time steps passively on u and stores a checkpoint for the reverse pass.
    void computeStage(std::vector<active_type>& u, int nSteps)
    {
        Checkpoint chk;
        chk.nSteps = nSteps;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            chk.slots.push_back(u[i].getSlot());
            chk.values.push_back(value(u[i]));
        }

        std::vector<value_type> passive = chk.values;
        heatSteps(passive, lambda_, nSteps);

        // the outputs are written into the same variables, so they keep the input slots
        for (std::size_t i = 0; i < u.size(); ++i) value(u[i]) = passive[i];

        checkpoints_.push(chk);
        Tape::getActive()->insertCallback(this);
    }

    void computeAdjoint(Tape* tape)
    {
        const Checkpoint& chk = checkpoints_.top();
        const std::size_t n = chk.values.size();

        // retrieve the output adjoints before any active operation, which may overwrite them
        std::vector<value_type> outputAdjoints(n);
        for (std::size_t i = 0; i < n; ++i)
            outputAdjoints[i] = tape->getAndResetOutputAdjoint(chk.slots[i]);

        std::vector<active_type> inputs(chk.values.begin(), chk.values.end());
        for (std::size_t i = 0; i < n; ++i) tape->registerInput(inputs[i]);

        // re-run the stage actively in a nested recording
        xad::ScopedNestedRecording<Tape> nested(tape);
        std::vector<active_type> u(inputs);
        heatSteps(u, lambda_, chk.nSteps);

        for (std::size_t i = 0; i < n; ++i)
        {
            tape->registerOutput(u[i]);
            derivative(u[i]) = outputAdjoints[i];
        }
        peakMemory_ = std::max(peakMemory_, tape->getMemory());
        nested.computeAdjoints();

        for (std::size_t i = 0; i < n; ++i)
            nested.incrementAdjoint(chk.slots[i], derivative(inputs[i]));

        checkpoints_.pop();
    }

    // largest tape memory seen while a stage was re-recorded in computeAdjoint
    std::size_t getPeakMemory() const { return peakMemory_; }

    // memory used by the checkpoints currently stored
    std::size_t getCheckpointMemory() const
    {
        if (checkpoints_.empty())
            return 0;
        return checkpoints_.size() * checkpoints_.top().values.size() *
               (sizeof(value_type) + sizeof(slot_type));
    }

  private:
    struct Checkpoint
    {
        int nSteps;                      // number of time steps in this stage
        std::vector<value_type> values;  // profile at the start of the stage
        std::vector<slot_type> slots;    // slots of the profile, for inputs and outputs
    };

    double lambda_;
    std::size_t peakMemory_;
    std::stack<Checkpoint> checkpoints_;
};

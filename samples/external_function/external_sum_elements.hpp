/*******************************************************************************

   Defines differentiated external functions for the sum_elements function,
   which is assumed to be defined in an optimized external library.


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

#include "functions.hpp"

#include <vector>

#include <XAD/XAD.hpp>

/** Callback object for computing derivatives of the external function.
 *
 * The only virtual method to be overridden is computeAdjoint.
 *
 * This is defined as a template so that it can also be used for higher order
 * derivatives.
 */
template <class Tape>
class ExternalSumElementsCallback : public xad::CheckpointCallback<Tape>
{
  public:
    typedef typename Tape::slot_type slot_type;      // type for slot in tape
    typedef typename Tape::value_type value_type;    // double
    typedef typename Tape::active_type active_type;  // AReal<double>

    /** Computes the value of the external function during the forward run
     * and stores the result in the active data type.
     *
     * This method is not part of the callback interface and could also be
     * implemented outside of this class.
     */
    active_type computeExternal(const active_type* x, unsigned n)
    {
        // store the slots of the input variables
        inputSlots_.resize(n);
        for (unsigned i = 0; i < n; ++i) inputSlots_[i] = x[i].getSlot();

        // create a copy of the data with passive data type
        std::vector<value_type> x_p(n);
        for (unsigned i = 0; i < n; ++i) x_p[i] = value(x[i]);

        // run the external function to compute the sum
        value_type y = sum_elements(&x_p[0], int(n));

        // set the output variable
        active_type ret = y;
        Tape::getActive()->registerOutput(ret);

        // store the slot of the active output
        outputSlot_ = ret.getSlot();

        // register this object as a callback, to call the manually AD'd adjoint
        // part
        Tape::getActive()->insertCallback(this);

        return ret;
    }

    /// Updates the adjoints for the external function, using manual AD
    void computeAdjoint(Tape* tape)
    {
        // we have a simple sum, so the adjoints of the inputs are simply the
        // output adjoint multiplied by 1, i.e., the output adjoint itself
        // -> hence we can increment the input adjoints directly by output_adj.

        value_type output_adj = tape->getAndResetOutputAdjoint(outputSlot_);
        for (std::size_t i = 0, e = inputSlots_.size(); i != e; ++i)
            tape->incrementAdjoint(inputSlots_[i], output_adj);
    }

  private:
    std::vector<slot_type> inputSlots_;  // slots of the inputs in the tape
    slot_type outputSlot_;               // slot of the output in the tape
};

// wrap this in a function for active types
template <class T>
xad::AReal<T> sum_elements(const xad::AReal<T>* x, int n)
{
    typedef typename xad::AReal<T>::tape_type tape_type;
    tape_type* tape = tape_type::getActive();
    ExternalSumElementsCallback<tape_type>* ckp = new ExternalSumElementsCallback<tape_type>;
    // register callback with tape - for automatic memory deallocation when tape
    // is cleared
    tape->pushCallback(ckp);

    return ckp->computeExternal(x, unsigned(n));
}

/// external function in forward mode AD
template <class T>
xad::FReal<T> sum_elements(const xad::FReal<T>* x, int n)
{
    typedef xad::FReal<T> active_type;

    // extract passive values
    std::vector<T> x_p(n);
    for (int i = 0; i < n; ++i) x_p[i] = value(x[i]);

    // call external function
    T y_p = sum_elements(&x_p[0], n);

    // assign to active output (also sets its derivative to 0)
    active_type y = y_p;

    // derivative of sum is sum of derivatives, so accumulate input derivatives
    for (int i = 0; i < n; ++i) derivative(y) += derivative(x[i]);

    return y;
}

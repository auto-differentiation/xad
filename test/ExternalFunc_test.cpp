/*******************************************************************************

   Unit tests for external functions

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
#include <gtest/gtest.h>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void g(std::vector<T> x, T& y)
{
    y = 0.0;
    for (unsigned i = 0; i < x.size(); i++) y += x[i];
}

template <typename T>
void g_adjoint(std::vector<T>& xa1, const T& ya1)
{
    typename std::vector<T>::iterator i;
    for (i = xa1.begin(); i != xa1.end(); ++i) *i += ya1;
}

template <class Tape>
class ExtChkCallback : public xad::CheckpointCallback<Tape>
{
  public:
    void computeAdjoint(Tape* tape) override
    {
        vector<double> xa1(inputs_.size());
        double ya1 = tape->getAndResetOutputAdjoint(output_);
        g_adjoint<double>(xa1, ya1);
        for (unsigned i = 0, n = unsigned(inputs_.size()); i < n; ++i)
            tape->incrementAdjoint(inputs_[i], xa1[i]);
    }

    std::vector<typename Tape::slot_type> inputs_;
    typename Tape::slot_type output_;
};

template <typename T>
void g_insert_ext(std::vector<T>& x, T& y)
{
    auto n = x.size();

    auto cb = new ExtChkCallback<typename T::tape_type>;
    x.front().getTape()->pushCallback(cb);  // memory management - deleted if tape is cleared

    cb->inputs_.resize(n);

    // get double inputs and register their slot with the callback
    vector<double> xv(n);
    for (unsigned i = 0; i < n; ++i)
    {
        xv[i] = value(x[i]);
        cb->inputs_[i] = x[i].getSlot();
    }

    double yv;
    g<double>(xv, yv);
    // set the value of y to yv and register its slot with the callback
    value(y) = yv;
    x.front().getTape()->registerOutput(y);
    cb->output_ = y.getSlot();

    // insert the callback
    y.getTape()->insertCallback(cb);
}

template <typename T>
void f(std::vector<T>& x, T& y)
{
    for (unsigned i = 0; i < x.size(); i++) x[i] *= x[i];
    g(x, y);

    y *= y;
}

template <typename T>
void f_ext(std::vector<T>& x, T& y)
{
    for (unsigned i = 0; i < x.size(); i++) x[i] *= x[i];
    g_insert_ext(x, y);

    y *= y;
}

// returns memory used
template <class F>
size_t driver_adj(const vector<double>& xv, vector<double>& xa1, double& yv, double& ya1, F func)
{
    typedef xad::AReal<double> ad_type;
    typedef ad_type::tape_type tape_type;
    tape_type t;

    unsigned n = unsigned(xv.size());

    vector<ad_type> x(n);

    for (unsigned i = 0; i < n; i++)
    {
        t.registerInput(x[i]);
        x[i] = xv[i];
        std::cout << xv[i] << "\n";
        // derivative(x[i]) = xa1[i];
    }

    t.newRecording();  // start recording derivatives - computeAdjoints reverses
                       // to here
    for (unsigned i = 0; i < n; i++) derivative(x[i]) = xa1[i];
    ad_type y;
    func(x, y);

    size_t mem = t.getMemory();
    t.registerOutput(y);
    yv = value(y);
    derivative(y) = ya1;
    t.computeAdjoints();

    for (unsigned i = 0; i < n; i++)
    {
        xa1[i] = derivative(x[i]);
    }

    return mem;
}

TEST(ExternalFunctions, manual)
{
    unsigned n = 5;
    vector<double> x(n), xa1(n);
    double y = 0., ya1;
    for (unsigned i = 0; i < n; i++)
    {
        x[i] = std::cos(double(i));
        xa1[i] = 0.;
    }
    ya1 = 1.0;

    vector<double> x_ref(x), xa1_ref(xa1);
    double y_ref = 0.0, ya1_ref = ya1;
    // without ext. func
    auto mem_ref = driver_adj(x_ref, xa1_ref, y_ref, ya1_ref, &f<xad::AReal<double> >);
    // with ext. func
    auto mem_ext = driver_adj(x, xa1, y, ya1, &f_ext<xad::AReal<double> >);

    EXPECT_DOUBLE_EQ(y_ref, y);
    for (unsigned i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(xa1_ref[i], xa1[i]) << i;
    EXPECT_LT(mem_ext, mem_ref);
}

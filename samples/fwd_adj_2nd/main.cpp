/*******************************************************************************

   Computes
      y = f(x0, x1, x2, x3),
   its first order derivatives using adjoints
      dy/dx0, dy/dx1, dy/dx2, dy/dx3
   and the second order derivatives w.r.t. x0 using forward AD
      d2y/dx0dx0, d2y/dx1dx0, d2y/dx2dx0, d2y/dx3dx0


   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

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

#include "functions.hpp"

#include <iostream>

#include <XAD/XAD.hpp>

int main()
{
    // input values
    double x0 = 1.0;
    double x1 = 1.5;
    double x2 = 1.3;
    double x3 = 1.2;

    // tape and active data type for 1st order adjoint computation
    typedef xad::fwd_adj<double> mode;
    // Uncomment the following to disable expression templates for debugging
    // typedef xad::fwdd_adjd<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    // initialize tape
    tape_type tape;

    // set independent variables
    AD x0_ad = x0;
    AD x1_ad = x1;
    AD x2_ad = x2;
    AD x3_ad = x3;

    // register them with the tape
    tape.registerInput(x0_ad);
    tape.registerInput(x1_ad);
    tape.registerInput(x2_ad);
    tape.registerInput(x3_ad);

    // set variable of which we are interest in the second order derivatives
    derivative(value(x0_ad)) = 1.0;

    // start recording derivatives
    tape.newRecording();

    AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);

    // seed 1st order adjoint of output
    tape.registerOutput(y);
    value(derivative(y)) = 1.0;

    // compute all other adjoints and the second order tangent linear
    tape.computeAdjoints();

    // output results
    std::cout << "y      = " << value(value(y)) << "\n"
              << "\nfirst order derivatives:\n"
              << "dy/dx0 = " << value(derivative(x0_ad)) << "\n"
              << "dy/dx1 = " << value(derivative(x1_ad)) << "\n"
              << "dy/dx2 = " << value(derivative(x2_ad)) << "\n"
              << "dy/dx3 = " << value(derivative(x3_ad)) << "\n"
              << "\nsecond order derivatives w.r.t. x0:\n"
              << "d2y/dx0dx0 = " << derivative(derivative(x0_ad)) << "\n"
              << "d2y/dx0dx1 = " << derivative(derivative(x1_ad)) << "\n"
              << "d2y/dx0dx2 = " << derivative(derivative(x2_ad)) << "\n"
              << "d2y/dx0dx3 = " << derivative(derivative(x3_ad)) << "\n";
}

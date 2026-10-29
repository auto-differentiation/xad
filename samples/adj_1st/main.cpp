/*******************************************************************************

   Computes
     y = f(x0, x1, x2, x3)
   and its first order derivatives
     dy/dx0, dy/dx1, dy/dx2, dy/dx3
   using adjoint mode.


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
    typedef xad::adj<double> mode;

    // Uncomment the following to disable expression templates for debugging
    // typedef xad::adjd<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    // initialize tape
    tape_type tape;

    // set independent variables
    AD x0_ad = x0;
    AD x1_ad = x1;
    AD x2_ad = x2;
    AD x3_ad = x3;

    // and register them
    tape.registerInput(x0_ad);
    tape.registerInput(x1_ad);
    tape.registerInput(x2_ad);
    tape.registerInput(x3_ad);

    // start recording derivatives
    tape.newRecording();

    AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);

    // register and seed adjoint of output
    tape.registerOutput(y);
    derivative(y) = 1.0;

    // compute all other adjoints
    tape.computeAdjoints();

    // output results
    std::cout << "y = " << value(y) << "\n"
              << "\nfirst order derivatives:\n"
              << "dy/dx0 = " << derivative(x0_ad) << "\n"
              << "dy/dx1 = " << derivative(x1_ad) << "\n"
              << "dy/dx2 = " << derivative(x2_ad) << "\n"
              << "dy/dx3 = " << derivative(x3_ad) << "\n";
}

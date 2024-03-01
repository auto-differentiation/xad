/*******************************************************************************

   Computes
     y = f(x0, x1, x2, x3)
   and it's first order derivative w.r.t. x0 using forward mode:
     dy/dx0

   Further derivatives can be obtained by repeating the process and seeding the
   initial derivatives differently.

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

    // tape and active data type for 1st order forward (tangent-linear)
    // computation
    typedef xad::fwd<double> mode;
    typedef mode::active_type AD;

    // set independent variables
    AD x0_ad = x0;
    AD x1_ad = x1;
    AD x2_ad = x2;
    AD x3_ad = x3;

    // compute derivative w.r.t. x0
    // (if other derivatives are needed, the initial derivatives have to be reset
    // and the function run again)
    derivative(x0_ad) = 1.0;

    // run the algorithm with active variables
    AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);

    // output results
    std::cout << "y = " << value(y) << "\n"
              << "\nfirst order derivative:\n"
              << "dy/dx0 = " << derivative(y) << "\n";
}

/*******************************************************************************

   Computes the length of a vector as
     y = sqrt(sum(x^2))
   where the square operation is element-wise.

   The derivatives w.r.t. to all inputs are computed using adjoint mode AD,
   where the summing of the elements is implemented as an external function
   and a manual adjoint is implemented for the reverse path.


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

#include "external_sum_elements.hpp"

#include <XAD/XAD.hpp>
#include <iostream>
#include <vector>

int main()
{
    // setup input variables
    unsigned n = 5;
    std::vector<double> x(n);
    for (unsigned i = 0; i < n; ++i) x[i] = double(i + 1) + std::sin(double(i));

    // typedefs for adjoint mode AD
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    // initialize tape
    tape_type tape;

    // setup independent variables
    std::vector<AD> x_ad(x.begin(), x.end());
    tape.registerInputs(x_ad);

    // start recording derivatives
    tape.newRecording();

    // compute sqrt(sum(x^2)) - see external_sum_elements.hpp for sum_elements
    // implementation using the external functions interface
    std::vector<AD> xsqr(n);
    for (unsigned i = 0; i < n; ++i) xsqr[i] = x_ad[i] * x_ad[i];
    AD y = sqrt(sum_elements(xsqr.data(), int(n)));

    // set output adjoint and compute the other adjoints
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();

    // output results
    std::cout << "y = " << value(y) << "\n";
    std::cout << "\nfirst order derivatives:\n";
    for (unsigned i = 0; i < n; ++i)
        std::cout << "dy/dx" << i << " = " << derivative(x_ad[i]) << "\n";
}

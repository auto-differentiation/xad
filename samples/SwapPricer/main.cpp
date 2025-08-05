/*******************************************************************************

   Computes the discount rate sensitivities of a simple swap pricer
   using adjoint mode.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <XAD/XAD.hpp>
#include "SwapPricer.hpp"

int main()
{
    // initialize dummy input data
    int nRates = 30;
    double faceValue = 10000000.;
    double fixedRate = 0.03;
    bool isFixedPay = true;
    std::vector<double> floatRates, discRates;
    std::vector<double> maturities;
    std::srand(unsigned(std::time(NULL)));
    for (int i = 0; i < nRates; ++i)
    {
        floatRates.push_back(0.01 + double(rand()) / double(RAND_MAX) * 0.1);
        discRates.push_back(0.01 + double(rand()) / double(RAND_MAX) * 0.06);
        maturities.push_back(double(i + 1));
    }

    // tape and active data type for 1st order adjoint computation
    typedef xad::adj<double> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    // initialize tape
    tape_type tape;

    // set independent variables
    std::vector<AD> discRates_ad(discRates.begin(), discRates.end());
    tape.registerInputs(discRates_ad);

    // start recording derivatives
    tape.newRecording();

    AD v = priceSwap(&discRates_ad[0], isFixedPay, nRates, &maturities[0], &floatRates[0],
                     fixedRate, faceValue);

    // seed adjoint of output
    tape.registerOutput(v);
    derivative(v) = 1.0;

    // compute all other adjoints
    tape.computeAdjoints();

    // output results
    std::cout << "v = " << value(v) << "\n";
    std::cout << "Discount rate sensitivities for 1 basispoint shift:\n";
    for (unsigned i = 0; i < unsigned(nRates); ++i)
        std::cout << "dv/dr" << i << " = " << derivative(discRates_ad[i]) * 0.0001 << "\n";
}

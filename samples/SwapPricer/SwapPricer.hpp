/*******************************************************************************

   Functions to price a simple IR Swap

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2023 Xcelerit Computing Ltd.

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

/// prices a simple swap, given the discount and floating rates at a range
/// of cashflow dates (maturities)
template <class T>
T priceSwap(const T* discRates, bool isFixedPay, int n, const double* mat, const double* floatRates,
            double fixedRate, double faceValue)
{
    using std::pow;

    T Bfix = 0.0;
    for (int t = 0; t < n; ++t) Bfix += fixedRate / pow(1.0 + discRates[t], mat[t]);
    Bfix += faceValue / pow(1.0 + discRates[n - 1], mat[n - 1]);

    T Bflt = 0.0;
    for (int t = 0; t < n; ++t) Bflt += floatRates[t] / pow(1.0 + discRates[t], mat[t]);
    Bflt += faceValue / pow(1.0 + discRates[n - 1], mat[n - 1]);

    return isFixedPay ? Bflt - Bfix : Bfix - Bflt;
}

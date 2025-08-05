/*******************************************************************************

   Data structures used to price a portfolio of LIBOR swaptions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   The code is an adapted version of Prof. Mike Giles, available here:
   https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp

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

#pragma once

#include <vector>


/// Represents portfolio of swaptions, with maturities and corresponding
/// swap rates
struct SwaptionPortfolio
{
    std::vector<double> swaprates;
    std::vector<int> maturities;
};

/// Market parameters required for pricing
struct MarketParameters
{
    double delta = 0.;
    std::vector<double> lambda;
    std::vector<double> L0;
};

/// Holds portfolio price and derivatives
struct Results
{
    double price = 0.;
    double d_delta = 0.;
    std::vector<double> d_L0;
    std::vector<double> d_lambda;
};

/*******************************************************************************

   Pricing functions for a portfolio of LIBOR swaptions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   The code is an adapted version of Prof. Mike Giles, available here:
   https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp

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

#pragma once
#include "LiborData.hpp"

/// Pure pricing without sensitivities
Results pricePortfolio(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                       int numPaths, unsigned long long seed = 12354);

/// Price with first-order sensitivities, using AAD
Results pricePortfolioAD(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                         int numPaths, unsigned long long seed = 12354);

/// Price with first-order sensitivities, estimated using finite differences
Results pricePortfolioFD(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                         int numPaths, unsigned long long seed = 12354);

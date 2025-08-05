/*******************************************************************************

   Functions used to price a portfolio of LIBOR swaptions.

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

//////////////////// Pricing functions ////////////////
// Defined as templates, so that they can be used   ///
// with active and passive types
///////////////////////////////////////////////////////

/// Path generation - calculates LIBOR rates at the given times,
/// Based on Gaussian randdom numbers
template <class ADT>
void path_gen(const ADT& delta, std::vector<ADT>& L, const std::vector<ADT>& lambda,
              const std::vector<double>& z)
{
    using std::exp;
    using std::sqrt;

    for (size_t n = 0; n < z.size(); n++)
    {
        ADT sqez = sqrt(delta) * z[n];

        ADT v = 0.0;
        for (size_t i = n + 1; i < L.size(); i++)
        {
            ADT lam = lambda[i - n - 1];
            ADT con1 = delta * lam;
            v += (con1 * L[i]) / (1.0 + delta * L[i]);
            L[i] *= exp(con1 * v + lam * (sqez - 0.5 * con1));
        }
    }
}

/// Value the swap portfolio for the given LIBOR rates
template <class ADT>
ADT value_portfolio(const ADT delta, const std::vector<int>& maturities,
                    const std::vector<double>& swaprates, const std::vector<ADT>& L,
                    std::vector<ADT>& Btmp, std::vector<ADT>& Stmp)
{
    const size_t NN = L.size();
    const size_t N = NN / 2;
    const size_t Nopt = swaprates.size();

    // temporaries, passed as parameters to avoid re-allocating
    Btmp.resize(NN);
    Stmp.resize(NN);

    ADT b = 1.0;
    ADT s = 0.0;

    for (size_t n = N; n < NN; ++n)
    {
        b = b / (1.0 + delta * L[n]);
        s = s + delta * b;
        Btmp[n] = b;
        Stmp[n] = s;
    }

    ADT v = 0.0;

    for (size_t i = 0; i < Nopt; i++)
    {
        int m = maturities[i] + N - 1;
        ADT swapval = Btmp[m] + swaprates[i] * Stmp[m] - 1.0;
        if (swapval < 0.)
            v += -100.0 * swapval;
    }

    // apply discount

    for (size_t n = 0; n < N; n++) v = v / (1.0 + delta * L[n]);

    return v;
}

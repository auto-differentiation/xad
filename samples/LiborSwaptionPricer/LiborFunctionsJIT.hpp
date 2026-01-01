/*******************************************************************************

   JIT-compatible functions used to price a portfolio of LIBOR swaptions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   This is a JIT-compatible version of LiborFunctions.hpp that uses ABool::If
   for branching, allowing the JIT compiler to record both branches and select
   at runtime.

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

#include <XAD/XAD.hpp>
#include <vector>

//////////////////// JIT-Compatible Pricing functions /////////
// These functions use ABool::If for branching, which allows ///
// the JIT compiler to record both branches and select at    ///
// runtime based on the actual values.                       ///
////////////////////////////////////////////////////////////////

/// Path generation - calculates LIBOR rates at the given times,
/// Based on Gaussian random numbers (passed as AD types for JIT)
///
/// Note: This function has no branching, so it works with JIT as-is.
/// We just need the random samples to be AD types registered as JIT inputs.
template <class ADT>
void path_gen_jit(const ADT& delta, std::vector<ADT>& L, const std::vector<ADT>& lambda,
                  const std::vector<ADT>& z)
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

/// Value the swap portfolio for the given LIBOR rates (JIT-compatible version)
///
/// This version uses ABool::If for the swaption payoff, which allows the JIT
/// compiler to record both branches and select at runtime.
inline xad::AD value_portfolio_jit(const xad::AD& delta, const std::vector<int>& maturities,
                                   const std::vector<double>& swaprates, const std::vector<xad::AD>& L,
                                   std::vector<xad::AD>& Btmp, std::vector<xad::AD>& Stmp)
{
    const size_t NN = L.size();
    const size_t N = NN / 2;
    const size_t Nopt = swaprates.size();

    // temporaries, passed as parameters to avoid re-allocating
    Btmp.resize(NN);
    Stmp.resize(NN);

    xad::AD b = 1.0;
    xad::AD s = 0.0;

    for (size_t n = N; n < NN; ++n)
    {
        b = b / (1.0 + delta * L[n]);
        s = s + delta * b;
        Btmp[n] = b;
        Stmp[n] = s;
    }

    xad::AD v = 0.0;

    for (size_t i = 0; i < Nopt; i++)
    {
        int m = maturities[i] + static_cast<int>(N) - 1;
        xad::AD swapval = Btmp[m] + swaprates[i] * Stmp[m] - 1.0;

        // JIT-compatible branching: use ABool::If to record both branches
        // Original: if (swapval < 0.) v += -100.0 * swapval;
        // This computes: v += (swapval < 0) ? -100.0 * swapval : 0.0
        v += xad::less(swapval, xad::AD(0.0)).If(-100.0 * swapval, xad::AD(0.0));
    }

    // apply discount
    for (size_t n = 0; n < N; n++)
        v = v / (1.0 + delta * L[n]);

    return v;
}

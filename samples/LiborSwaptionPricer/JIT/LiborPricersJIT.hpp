/*******************************************************************************

   JIT-accelerated pricing functions for a portfolio of LIBOR swaptions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   These functions use Forge JIT compilation for accelerated AAD computation.

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
#include "../LiborData.hpp"

/// Statistics from JIT compilation
struct JITStats
{
    double compileTimeMs = 0.0;  ///< Time spent compiling the JIT kernel
};

/// Detailed timing breakdown for performance decomposition
struct TimingDecomposition
{
    double totalMs = 0.0;           ///< Total execution time
    double compileMs = 0.0;         ///< JIT compilation time (one-time)
    double setInputsMs = 0.0;       ///< Time setting input values
    double forwardMs = 0.0;         ///< Forward pass execution time
    double backwardMs = 0.0;        ///< Backward pass (adjoint) execution time
    double getGradientsMs = 0.0;    ///< Time retrieving gradients
    double accumulateMs = 0.0;      ///< Time accumulating results
    int numPaths = 0;               ///< Number of paths executed
};

/// Run performance decomposition for JIT scalar backend
/// Returns detailed timing breakdown of each phase
TimingDecomposition runDecompositionJIT(const SwaptionPortfolio& portfolio,
                                         const MarketParameters& market,
                                         int numPaths, unsigned long long seed = 12354);

/// Run performance decomposition for JIT AVX backend
/// Returns detailed timing breakdown of each phase
TimingDecomposition runDecompositionJIT_AVX(const SwaptionPortfolio& portfolio,
                                             const MarketParameters& market,
                                             int numPaths, unsigned long long seed = 12354);

/// Price with first-order sensitivities, using AAD with Forge JIT compilation
/// The computation graph is compiled once on the first path and reused.
/// @param stats Optional pointer to receive JIT compilation statistics
Results pricePortfolioJIT(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                          int numPaths, unsigned long long seed = 12354, JITStats* stats = nullptr);

/// Price with first-order sensitivities, using Forge JIT with AVX2 SIMD
/// Processes 4 Monte Carlo paths per kernel execution using AVX2 instructions.
/// @param stats Optional pointer to receive JIT compilation statistics
Results pricePortfolioJIT_AVX(const SwaptionPortfolio& portfolio, const MarketParameters& market,
                              int numPaths, unsigned long long seed = 12354, JITStats* stats = nullptr);

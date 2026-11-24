/*******************************************************************************

   Unit tests for graph recording of math function derivatives.

   This file tests graph-specific behavior for math functions. Only functions
   that work correctly with graph recording are included here.

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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include "TestHelpers.hpp"

#ifdef XAD_ENABLE_GRAPH

// Helper macros for graph test functors
#define GRAPH_TEST_FUNCTOR1(name, val)                                                               \
    struct graphTestFunctor_##name                                                                   \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x) const                                                             \
        {                                                                                          \
            using namespace std;                                                                   \
            return val;                                                                            \
        }                                                                                          \
    } name;

#define GRAPH_TEST_FUNCTOR2(name, val)                                                               \
    struct graphTestFunctor_##name                                                                   \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x1, const T& x2) const                                               \
        {                                                                                          \
            using namespace std;                                                                   \
            return val;                                                                            \
        }                                                                                          \
    } name;

// =============================================================================
// Basic arithmetic and unary operations
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphCosAD, cos(x))
TEST(GraphExpressionMath, cosAD)
{
    mathTest_graph(1.0, std::cos(1.0), -std::sin(1.0), graphCosAD);
}

GRAPH_TEST_FUNCTOR1(graphCosExpr, cos(2.3 * x))
TEST(GraphExpressionMath, cosExpr)
{
    mathTest_graph(1.0, std::cos(2.3), -2.3 * std::sin(2.3), graphCosExpr);
}

GRAPH_TEST_FUNCTOR1(graphSinAD, sin(x))
TEST(GraphExpressionMath, sinAD)
{
    mathTest_graph(1.0, std::sin(1.0), std::cos(1.0), graphSinAD);
}

GRAPH_TEST_FUNCTOR1(graphSinExpr, sin(2.3 * x))
TEST(GraphExpressionMath, sinExpr)
{
    mathTest_graph(1.0, std::sin(2.3), 2.3 * std::cos(2.3), graphSinExpr);
}

GRAPH_TEST_FUNCTOR1(graphExpAD, exp(x))
TEST(GraphExpressionMath, expAD)
{
    mathTest_graph(1.0, std::exp(1.0), std::exp(1.0), graphExpAD);
}

GRAPH_TEST_FUNCTOR1(graphExpExpr, exp(2.3 * x))
TEST(GraphExpressionMath, expExpr)
{
    mathTest_graph(1.0, std::exp(2.3), 2.3 * std::exp(2.3), graphExpExpr);
}

GRAPH_TEST_FUNCTOR1(graphLogAD, log(x))
TEST(GraphExpressionMath, logAD)
{
    mathTest_graph(1.3, std::log(1.3), 1.0 / 1.3, graphLogAD);
}

GRAPH_TEST_FUNCTOR1(graphLogExpr, log(2.3 * x))
TEST(GraphExpressionMath, logExpr)
{
    mathTest_graph(1.0, std::log(2.3), 1.0, graphLogExpr);
}

GRAPH_TEST_FUNCTOR1(graphLog10AD, log10(x))
TEST(GraphExpressionMath, log10AD)
{
    mathTest_graph(1.3, std::log10(1.3), 1.0 / std::log(10.0) / 1.3, graphLog10AD);
}

GRAPH_TEST_FUNCTOR1(graphLog2AD, log2(x))
TEST(GraphExpressionMath, log2AD)
{
    using xad::log2;
    mathTest_graph(1.3, log2(1.3), 1.0 / log(2.0) / 1.3, graphLog2AD);
}

GRAPH_TEST_FUNCTOR1(graphSqrtAD, sqrt(x))
TEST(GraphExpressionMath, sqrtAD)
{
    mathTest_graph(1.3, std::sqrt(1.3), 0.5 / std::sqrt(1.3), graphSqrtAD);
}

GRAPH_TEST_FUNCTOR1(graphSqrtExpr, sqrt(2.3 * x))
TEST(GraphExpressionMath, sqrtExpr)
{
    mathTest_graph(1.3, std::sqrt(2.3 * 1.3), 2.3 * 0.5 / std::sqrt(2.3 * 1.3), graphSqrtExpr);
}

GRAPH_TEST_FUNCTOR1(graphCbrtAD, cbrt(x))
TEST(GraphExpressionMath, cbrtAD)
{
    mathTest_graph(1.3, std::cbrt(1.3), 1.0 / 3.0 / std::cbrt(1.3) / std::cbrt(1.3), graphCbrtAD);
}

// =============================================================================
// Trigonometric functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphTanAD, tan(x))
TEST(GraphExpressionMath, tanAD)
{
    mathTest_graph(0.3, std::tan(0.3), 1.0 / std::cos(0.3) / std::cos(0.3), graphTanAD);
}

GRAPH_TEST_FUNCTOR1(graphAsinAD, asin(x))
TEST(GraphExpressionMath, asinAD)
{
    mathTest_graph(0.3, std::asin(0.3), 1.0 / std::sqrt(1.0 - 0.3 * 0.3), graphAsinAD);
}

GRAPH_TEST_FUNCTOR1(graphAcosAD, acos(x))
TEST(GraphExpressionMath, acosAD)
{
    mathTest_graph(0.3, std::acos(0.3), -1.0 / std::sqrt(1.0 - 0.3 * 0.3), graphAcosAD);
}

GRAPH_TEST_FUNCTOR1(graphAtanAD, atan(x))
TEST(GraphExpressionMath, atanAD)
{
    mathTest_graph(0.3, std::atan(0.3), 1.0 / (1.0 + 0.3 * 0.3), graphAtanAD);
}

// =============================================================================
// Hyperbolic functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphSinhAD, sinh(x))
TEST(GraphExpressionMath, sinhAD)
{
    mathTest_graph(0.3, std::sinh(0.3), std::cosh(0.3), graphSinhAD);
}

GRAPH_TEST_FUNCTOR1(graphCoshAD, cosh(x))
TEST(GraphExpressionMath, coshAD)
{
    mathTest_graph(0.3, std::cosh(0.3), std::sinh(0.3), graphCoshAD);
}

GRAPH_TEST_FUNCTOR1(graphTanhAD, tanh(x))
TEST(GraphExpressionMath, tanhAD)
{
    mathTest_graph(0.3, std::tanh(0.3), 1.0 - std::tanh(0.3) * std::tanh(0.3), graphTanhAD);
}

GRAPH_TEST_FUNCTOR1(graphAsinhAD, asinh(x))
TEST(GraphExpressionMath, asinhAD)
{
    using xad::asinh;
    mathTest_graph(0.3, asinh(0.3), 1.0 / std::sqrt(1.0 + 0.3 * 0.3), graphAsinhAD);
}

GRAPH_TEST_FUNCTOR1(graphAcoshAD, acosh(x))
TEST(GraphExpressionMath, acoshAD)
{
    using xad::acosh;
    mathTest_graph(1.3, acosh(1.3), 1.0 / std::sqrt(1.3 * 1.3 - 1.0), graphAcoshAD);
}

GRAPH_TEST_FUNCTOR1(graphAtanhAD, atanh(x))
TEST(GraphExpressionMath, atanhAD)
{
    using xad::atanh;
    mathTest_graph(0.3, atanh(0.3), 1.0 / (1.0 - 0.3 * 0.3), graphAtanhAD);
}

// =============================================================================
// Special functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphErfAD, erf(x))
TEST(GraphExpressionMath, erfAD)
{
    mathTest_graph(0.3, std::erf(0.3), 2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3), graphErfAD);
}

GRAPH_TEST_FUNCTOR1(graphErfcAD, erfc(x))
TEST(GraphExpressionMath, erfcAD)
{
    mathTest_graph(0.3, std::erfc(0.3), -2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3), graphErfcAD);
}

GRAPH_TEST_FUNCTOR1(graphTgammaAD, tgamma(x))
TEST(GraphExpressionMath, tgammaAD)
{
    mathTest_graph(2.7, std::tgamma(2.7), 1.2307796833525972, graphTgammaAD);
}

GRAPH_TEST_FUNCTOR1(graphLgammaAD, lgamma(x))
TEST(GraphExpressionMath, lgammaAD)
{
    mathTest_graph(-2.5, std::lgamma(-2.5), 1.1031566406452431, graphLgammaAD);
}

GRAPH_TEST_FUNCTOR1(graphDigammaAD, xad::digamma(x))
TEST(GraphExpressionMath, digammaAD)
{
    mathTest_graph(0.3, xad::digamma(0.3), 12.24536454610773, graphDigammaAD);
}

GRAPH_TEST_FUNCTOR1(graphExpm1AD, expm1(x))
TEST(GraphExpressionMath, expm1AD)
{
    using xad::expm1;
    mathTest_graph(0.3, expm1(0.3), std::exp(0.3), graphExpm1AD);
}

GRAPH_TEST_FUNCTOR1(graphLog1pAD, log1p(x))
TEST(GraphExpressionMath, log1pAD)
{
    using xad::log1p;
    mathTest_graph(0.3, log1p(0.3), 1.0 / (1.0 + 0.3), graphLog1pAD);
}

GRAPH_TEST_FUNCTOR1(graphExp2AD, exp2(x))
TEST(GraphExpressionMath, exp2AD)
{
    using xad::exp2;
    mathTest_graph(0.3, exp2(0.3), std::log(2.0) * exp2(0.3), graphExp2AD);
}

// =============================================================================
// Rounding functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphFloorAD, floor(x))
TEST(GraphExpressionMath, floorAD)
{
    mathTest_graph(1.7, 1.0, 0.0, graphFloorAD);
}

GRAPH_TEST_FUNCTOR1(graphCeilAD, ceil(x))
TEST(GraphExpressionMath, ceilAD)
{
    mathTest_graph(1.3, 2.0, 0.0, graphCeilAD);
}

GRAPH_TEST_FUNCTOR1(graphTruncAD, trunc(x))
TEST(GraphExpressionMath, truncAD)
{
    using xad::trunc;
    mathTest_graph(1.7, 1.0, 0.0, graphTruncAD);
}

GRAPH_TEST_FUNCTOR1(graphRoundAD, round(x))
TEST(GraphExpressionMath, roundAD)
{
    using xad::round;
    mathTest_graph(1.7, 2.0, 0.0, graphRoundAD);
}

// =============================================================================
// Absolute value (with special handling at x=0)
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphAbsAD, abs(x))
TEST(GraphExpressionMath, absAD)
{
    mathTest_graph(1.3, 1.3, 1.0, graphAbsAD);
    mathTest_graph(-1.3, 1.3, -1.0, graphAbsAD);
    mathTest_graph(0.0, 0.0, 0.0, graphAbsAD);  // derivative at 0 is 0
}

GRAPH_TEST_FUNCTOR1(graphFabsAD, fabs(x))
TEST(GraphExpressionMath, fabsAD)
{
    mathTest_graph(1.3, 1.3, 1.0, graphFabsAD);
    mathTest_graph(-1.3, 1.3, -1.0, graphFabsAD);
    mathTest_graph(0.0, 0.0, 0.0, graphFabsAD);  // derivative at 0 is 0
}

// =============================================================================
// Power functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphPowScalarExpAD, pow(x, 2.1))
TEST(GraphExpressionMath, powScalarExpAD)
{
    mathTest_graph(0.3, std::pow(0.3, 2.1), 2.1 * std::pow(0.3, 1.1), graphPowScalarExpAD);
}

GRAPH_TEST_FUNCTOR1(graphPowScalarBaseAD, pow(2.1, x))
TEST(GraphExpressionMath, powScalarBaseAD)
{
    mathTest_graph(0.3, std::pow(2.1, 0.3), std::log(2.1) * std::pow(2.1, 0.3), graphPowScalarBaseAD);
}

GRAPH_TEST_FUNCTOR2(graphPowADAD, pow(x1, x2))
TEST(GraphExpressionMath, powADAD)
{
    mathTest2_graph(0.3, 2.1, std::pow(0.3, 2.1),
                  2.1 * std::pow(0.3, 1.1),              // d1
                  std::log(0.3) * std::pow(0.3, 2.1),    // d2
                  graphPowADAD);
}

// =============================================================================
// Two-variable functions
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphAddADAD, x1 + x2)
TEST(GraphExpressionMath, addADAD)
{
    mathTest2_graph(1.3, 0.7, 2.0, 1.0, 1.0, graphAddADAD);
}

GRAPH_TEST_FUNCTOR2(graphSubADAD, x1 - x2)
TEST(GraphExpressionMath, subADAD)
{
    mathTest2_graph(1.3, 0.7, 0.6, 1.0, -1.0, graphSubADAD);
}

GRAPH_TEST_FUNCTOR2(graphMulADAD, x1 * x2)
TEST(GraphExpressionMath, mulADAD)
{
    mathTest2_graph(1.3, 0.7, 1.3 * 0.7, 0.7, 1.3, graphMulADAD);
}

GRAPH_TEST_FUNCTOR2(graphDivADAD, x1 / x2)
TEST(GraphExpressionMath, divADAD)
{
    mathTest2_graph(1.3, 0.7, 1.3 / 0.7, 1.0 / 0.7, -1.3 / (0.7 * 0.7), graphDivADAD);
}

GRAPH_TEST_FUNCTOR2(graphAtan2AD, xad::atan2(x1, x2))
TEST(GraphExpressionMath, atan2AD)
{
    mathTest2_graph(0.3, 0.5, std::atan2(0.3, 0.5),
                  0.5 / (0.3 * 0.3 + 0.5 * 0.5),   // d1
                  -0.3 / (0.3 * 0.3 + 0.5 * 0.5),  // d2
                  graphAtan2AD);
}

GRAPH_TEST_FUNCTOR2(graphHypotAD, hypot(x1, x2))
TEST(GraphExpressionMath, hypotAD)
{
    mathTest2_graph(0.3, 0.5, std::hypot(0.3, 0.5),
                  0.3 / std::hypot(0.3, 0.5),  // d1
                  0.5 / std::hypot(0.3, 0.5),  // d2
                  graphHypotAD);
}

GRAPH_TEST_FUNCTOR2(graphFmodAD, fmod(x1, x2))
TEST(GraphExpressionMath, fmodAD)
{
    int n = static_cast<int>(1.3 / 0.5);
    mathTest2_graph(1.3, 0.5, std::fmod(1.3, 0.5),
                  1.0,             // d1
                  -double(n),      // d2
                  graphFmodAD);
}

GRAPH_TEST_FUNCTOR2(graphRemainderAD, remainder(x1, x2))
TEST(GraphExpressionMath, remainderAD)
{
    int n = static_cast<int>(std::round(1.3 / 0.5));
    mathTest2_graph(1.3, 0.5, std::remainder(1.3, 0.5),
                  1.0,             // d1
                  -double(n),      // d2
                  graphRemainderAD);
}

GRAPH_TEST_FUNCTOR2(graphNextafterAD, nextafter(x1, x2))
TEST(GraphExpressionMath, nextafterAD)
{
    mathTest2_graph(0.3, 0.5, std::nextafter(0.3, 0.5),
                  1.0,   // d1
                  0.0,   // d2
                  graphNextafterAD);
}

// =============================================================================
// Max/Min functions (AD vs AD - equal values case with 0.5/0.5 derivative split)
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphMaxADAD, max(x1, x2))
TEST(GraphExpressionMath, maxADAD)
{
    // x1 > x2: derivative flows to x1
    mathTest2_graph(1.7, 0.7, 1.7, 1.0, 0.0, graphMaxADAD);
    // x1 < x2: derivative flows to x2
    mathTest2_graph(0.3, 0.7, 0.7, 0.0, 1.0, graphMaxADAD);
    // x1 == x2: derivative splits 0.5/0.5
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphMaxADAD);
}

GRAPH_TEST_FUNCTOR2(graphMinADAD, min(x1, x2))
TEST(GraphExpressionMath, minADAD)
{
    // x1 < x2: derivative flows to x1
    mathTest2_graph(0.3, 0.7, 0.3, 1.0, 0.0, graphMinADAD);
    // x1 > x2: derivative flows to x2
    mathTest2_graph(1.7, 0.7, 0.7, 0.0, 1.0, graphMinADAD);
    // x1 == x2: derivative splits 0.5/0.5
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphMinADAD);
}

GRAPH_TEST_FUNCTOR2(graphFmaxADAD, fmax(x1, x2))
TEST(GraphExpressionMath, fmaxADAD)
{
    mathTest2_graph(0.3, 0.7, 0.7, 0.0, 1.0, graphFmaxADAD);
    mathTest2_graph(1.7, 0.7, 1.7, 1.0, 0.0, graphFmaxADAD);
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphFmaxADAD);
}

GRAPH_TEST_FUNCTOR2(graphFminADAD, fmin(x1, x2))
TEST(GraphExpressionMath, fminADAD)
{
    mathTest2_graph(0.3, 0.7, 0.3, 1.0, 0.0, graphFminADAD);
    mathTest2_graph(1.7, 0.7, 0.7, 0.0, 1.0, graphFminADAD);
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphFminADAD);
}

// =============================================================================
// Ldexp (x * 2^exp) - works because exp is a compile-time integer
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphLdexpAD, ldexp(x, 3))
TEST(GraphExpressionMath, ldexpAD)
{
    mathTest_graph(1.1, 1.1 * 8, 8.0, graphLdexpAD);
}

GRAPH_TEST_FUNCTOR1(graphLdexpExpr, ldexp(2.0 * x, 3))
TEST(GraphExpressionMath, ldexpExpr)
{
    mathTest_graph(1.1, 2.2 * 8, 16.0, graphLdexpExpr);
}

// =============================================================================
// Scalbn (similar to ldexp)
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphScalbnAD, scalbn(x, 3))
TEST(GraphExpressionMath, scalbnAD)
{
    mathTest_graph(1.1, std::scalbn(1.1, 3), std::scalbn(1.0, 3), graphScalbnAD);
}

// =============================================================================
// Degrees and Radians
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphDegreesAD, degrees(x))
TEST(GraphExpressionMath, degreesAD)
{
    mathTest_graph(3.141592653589793238462643, 180.0,
                 57.2957795130823208767981548141051703324054725, graphDegreesAD);
}

GRAPH_TEST_FUNCTOR1(graphDegreesExpr, degrees(0.5 * x))
TEST(GraphExpressionMath, degreesExpr)
{
    mathTest_graph(3.141592653589793238462643, 90.0,
                 0.5 * 57.2957795130823208767981548141051703324054725, graphDegreesExpr);
}

GRAPH_TEST_FUNCTOR1(graphRadiansAD, radians(x))
TEST(GraphExpressionMath, radiansAD)
{
    mathTest_graph(180.0, 3.141592653589793238462643,
                 0.0174532925199432957692369076848861271344287, graphRadiansAD);
}

GRAPH_TEST_FUNCTOR1(graphRadiansExpr, radians(2.0 * x))
TEST(GraphExpressionMath, radiansExpr)
{
    mathTest_graph(180.0, 2.0 * 3.141592653589793238462643,
                 2.0 * 0.0174532925199432957692369076848861271344287, graphRadiansExpr);
}

// =============================================================================
// Expression variants for trig functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphTanExpr, tan(2.3 * x))
TEST(GraphExpressionMath, tanExpr)
{
    mathTest_graph(0.3, std::tan(2.3 * 0.3), 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3),
                 graphTanExpr);
}

GRAPH_TEST_FUNCTOR1(graphAsinExpr, asin(2.3 * x))
TEST(GraphExpressionMath, asinExpr)
{
    mathTest_graph(0.1, std::asin(0.1 * 2.3), 2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 graphAsinExpr);
}

GRAPH_TEST_FUNCTOR1(graphAcosExpr, acos(2.3 * x))
TEST(GraphExpressionMath, acosExpr)
{
    mathTest_graph(0.1, std::acos(0.1 * 2.3), -2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 graphAcosExpr);
}

GRAPH_TEST_FUNCTOR1(graphAtanExpr, atan(2.3 * x))
TEST(GraphExpressionMath, atanExpr)
{
    mathTest_graph(0.1, std::atan(0.1 * 2.3), 2.3 / (1.0 + 2.3 * 2.3 * 0.1 * 0.1), graphAtanExpr);
}

// =============================================================================
// Expression variants for hyperbolic functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphSinhExpr, sinh(2.3 * x))
TEST(GraphExpressionMath, sinhExpr)
{
    mathTest_graph(0.1, std::sinh(0.1 * 2.3), 2.3 * std::cosh(2.3 * 0.1), graphSinhExpr);
}

GRAPH_TEST_FUNCTOR1(graphCoshExpr, cosh(2.3 * x))
TEST(GraphExpressionMath, coshExpr)
{
    mathTest_graph(0.3, std::cosh(2.3 * 0.3), 2.3 * std::sinh(2.3 * 0.3), graphCoshExpr);
}

GRAPH_TEST_FUNCTOR1(graphTanhExpr, tanh(2.3 * x))
TEST(GraphExpressionMath, tanhExpr)
{
    mathTest_graph(0.3, std::tanh(2.3 * 0.3),
                 2.3 * (1.0 - std::tanh(2.3 * 0.3) * std::tanh(2.3 * 0.3)), graphTanhExpr);
}

GRAPH_TEST_FUNCTOR1(graphAsinhExpr, asinh(2.3 * x))
TEST(GraphExpressionMath, asinhExpr)
{
    using xad::asinh;
    mathTest_graph(0.3, asinh(2.3 * 0.3), 2.3 / std::sqrt(1.0 + 2.3 * 2.3 * 0.3 * 0.3), graphAsinhExpr);
}

GRAPH_TEST_FUNCTOR1(graphAcoshExpr, acosh(2.3 * x))
TEST(GraphExpressionMath, acoshExpr)
{
    using xad::acosh;
    mathTest_graph(1.3, acosh(2.3 * 1.3), 2.3 / std::sqrt(2.3 * 2.3 * 1.3 * 1.3 - 1.0), graphAcoshExpr);
}

GRAPH_TEST_FUNCTOR1(graphAtanhExpr, atanh(2.3 * x))
TEST(GraphExpressionMath, atanhExpr)
{
    using xad::atanh;
    mathTest_graph(0.3, atanh(2.3 * 0.3), 2.3 / (1.0 - 2.3 * 2.3 * 0.3 * 0.3), graphAtanhExpr);
}

// =============================================================================
// Expression variants for special functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphErfExpr, erf(2.3 * x))
TEST(GraphExpressionMath, erfExpr)
{
    mathTest_graph(0.3, std::erf(2.3 * 0.3),
                 2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3), graphErfExpr);
}

GRAPH_TEST_FUNCTOR1(graphErfcExpr, erfc(2.3 * x))
TEST(GraphExpressionMath, erfcExpr)
{
    mathTest_graph(0.3, std::erfc(2.3 * 0.3),
                 -2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3), graphErfcExpr);
}

GRAPH_TEST_FUNCTOR1(graphExpm1Expr, expm1(2.3 * x))
TEST(GraphExpressionMath, expm1Expr)
{
    using xad::expm1;
    mathTest_graph(0.3, expm1(2.3 * 0.3), 2.3 * std::exp(2.3 * 0.3), graphExpm1Expr);
}

GRAPH_TEST_FUNCTOR1(graphLog1pExpr, log1p(2.3 * x))
TEST(GraphExpressionMath, log1pExpr)
{
    using xad::log1p;
    mathTest_graph(0.3, log1p(2.3 * 0.3), 2.3 / (1.0 + 2.3 * 0.3), graphLog1pExpr);
}

GRAPH_TEST_FUNCTOR1(graphExp2Expr, exp2(2.3 * x))
TEST(GraphExpressionMath, exp2Expr)
{
    using xad::exp2;
    mathTest_graph(0.3, exp2(2.3 * 0.3), 2.3 * std::log(2.0) * exp2(2.3 * 0.3), graphExp2Expr);
}

GRAPH_TEST_FUNCTOR1(graphLog10Expr, log10(2.3 * x))
TEST(GraphExpressionMath, log10Expr)
{
    mathTest_graph(1.0, std::log10(2.3), 2.3 / std::log(10.0) / 2.3, graphLog10Expr);
}

GRAPH_TEST_FUNCTOR1(graphLog2Expr, log2(2.3 * x))
TEST(GraphExpressionMath, log2Expr)
{
    using xad::log2;
    mathTest_graph(1.3, log2(2.3 * 1.3), 2.3 / log(2.0) / 1.3 / 2.3, graphLog2Expr);
}

// =============================================================================
// Expression variants for rounding functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphFloorExpr, floor(2.3 * x))
TEST(GraphExpressionMath, floorExpr)
{
    mathTest_graph(1.3, std::floor(2.3 * 1.3), 0.0, graphFloorExpr);
}

GRAPH_TEST_FUNCTOR1(graphCeilExpr, ceil(2.3 * x))
TEST(GraphExpressionMath, ceilExpr)
{
    mathTest_graph(1.3, std::ceil(2.3 * 1.3), 0.0, graphCeilExpr);
}

GRAPH_TEST_FUNCTOR1(graphTruncExpr, trunc(2.3 * x))
TEST(GraphExpressionMath, truncExpr)
{
    using xad::trunc;
    mathTest_graph(1.3, trunc(2.3 * 1.3), 0.0, graphTruncExpr);
}

GRAPH_TEST_FUNCTOR1(graphRoundExpr, round(2.3 * x))
TEST(GraphExpressionMath, roundExpr)
{
    using xad::round;
    mathTest_graph(1.3, round(2.3 * 1.3), 0.0, graphRoundExpr);
}

// =============================================================================
// Absolute value expression variants
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphAbsExpr, abs(2.3 * x))
TEST(GraphExpressionMath, absExpr)
{
    mathTest_graph(0.3, std::abs(2.3 * 0.3), 2.3, graphAbsExpr);
    mathTest_graph(-0.3, std::abs(2.3 * -0.3), -2.3, graphAbsExpr);
    mathTest_graph(0.0, std::abs(2.3 * 0.0), 0.0, graphAbsExpr);
}

GRAPH_TEST_FUNCTOR1(graphFabsExpr, fabs(2.3 * x))
TEST(GraphExpressionMath, fabsExpr)
{
    mathTest_graph(0.3, std::fabs(2.3 * 0.3), 2.3, graphFabsExpr);
    mathTest_graph(-0.3, std::fabs(2.3 * -0.3), -2.3, graphFabsExpr);
    mathTest_graph(0.0, std::fabs(2.3 * 0.0), 0.0, graphFabsExpr);
}

// =============================================================================
// Smooth absolute value functions
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphSmoothAbsAD, smooth_abs(x))
TEST(GraphExpressionMath, smoothAbsAD)
{
    mathTest_graph(0.3, std::abs(0.3), 1.0, graphSmoothAbsAD);
    mathTest_graph(-0.3, std::abs(-0.3), -1.0, graphSmoothAbsAD);
    mathTest_graph(0.0, std::abs(0.0), 0.0, graphSmoothAbsAD);
}

GRAPH_TEST_FUNCTOR1(graphSmoothAbsExpr, smooth_abs(2.3 * x))
TEST(GraphExpressionMath, smoothAbsExpr)
{
    mathTest_graph(0.3, std::abs(2.3 * 0.3), 2.3, graphSmoothAbsExpr);
    mathTest_graph(-0.3, std::abs(2.3 * -0.3), -2.3, graphSmoothAbsExpr);
    mathTest_graph(0.0, std::abs(2.3 * 0.0), 0.0, graphSmoothAbsExpr);
}

GRAPH_TEST_FUNCTOR2(graphSmoothAbsADAD, smooth_abs(x1, x2))
TEST(GraphExpressionMath, smoothAbsADAD)
{
    mathTest2_graph(0.3, 0.001, 0.3, 1.0, 0.0, graphSmoothAbsADAD);
    mathTest2_graph(-0.3, 0.001, 0.3, -1.0, 0.0, graphSmoothAbsADAD);
    mathTest2_graph(0.0, 0.001, 0.0, 0.0, 0.0, graphSmoothAbsADAD);
}

// =============================================================================
// Power function expression variants
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphPowExprExpr, pow(1.2 * x1, 2.3 * x2))
TEST(GraphExpressionMath, powExprExpr)
{
    mathTest2_graph(0.3, 2.1, std::pow(1.2 * 0.3, 2.3 * 2.1),
                  1.2 * 2.3 * 2.1 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0),
                  2.3 * std::log(1.2 * 0.3) * std::pow(1.2 * 0.3, 2.3 * 2.1), graphPowExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphPownAD, pown(x, 2))
TEST(GraphExpressionMath, pownAD)
{
    mathTest_graph(0.3, std::pow(0.3, 2), 2. * 0.3, graphPownAD);
}

GRAPH_TEST_FUNCTOR1(graphPownExpr, pown(2.3 * x, 2))
TEST(GraphExpressionMath, pownExpr)
{
    mathTest_graph(0.3, std::pow(2.3 * 0.3, 2), 2.3 * 2 * 2.3 * 0.3, graphPownExpr);
}

GRAPH_TEST_FUNCTOR1(graphPown1AD, pow(x, 2))
TEST(GraphExpressionMath, pown1AD)
{
    mathTest_graph(0.3, std::pow(0.3, 2), 2 * std::pow(0.3, 1), graphPown1AD);
}

// =============================================================================
// Cbrt expression variant
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphCbrtExpr, cbrt(2.1 * x))
TEST(GraphExpressionMath, cbrtExpr)
{
    using xad::cbrt;
    mathTest_graph(1.3, cbrt(2.1 * 1.3), 2.1 / 3.0 / std::pow(2.1 * 1.3, 2.0 / 3.0), graphCbrtExpr);
}

// =============================================================================
// Two-variable function expression variants
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphAtan2ExprExpr, xad::atan2(1.3 * x1, 1.3 * x2))
TEST(GraphExpressionMath, atan2ExprExpr)
{
    mathTest2_graph(0.3, 0.5, std::atan2(1.3 * 0.3, 1.3 * 0.5),
                  1.3 * 0.5 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),   // d1
                  -1.3 * 0.3 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),  // d2
                  graphAtan2ExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphAtan2ADScalar, xad::atan2(x, 0.5))
TEST(GraphExpressionMath, atan2ADScalar)
{
    mathTest_graph(0.3, std::atan2(0.3, 0.5), 0.5 / (0.3 * 0.3 + 0.5 * 0.5), graphAtan2ADScalar);
}

GRAPH_TEST_FUNCTOR2(graphHypotExprExpr, xad::hypot(1.3 * x1, 1.3 * x2))
TEST(GraphExpressionMath, hypotExprExpr)
{
    mathTest2_graph(0.3, 0.5, std::hypot(1.3 * 0.3, 1.3 * 0.5),
                  1.3 * 1.3 * 0.3 / std::hypot(1.3 * 0.3, 1.3 * 0.5),  // d1
                  1.3 * 1.3 * 0.5 / std::hypot(1.3 * 0.3, 1.3 * 0.5),  // d2
                  graphHypotExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphHypotADScalar, xad::hypot(x, 0.5))
TEST(GraphExpressionMath, hypotADScalar)
{
    mathTest_graph(0.3, std::hypot(0.3, 0.5), 0.3 / std::hypot(0.3, 0.5), graphHypotADScalar);
}

GRAPH_TEST_FUNCTOR2(graphFmodExprExpr, xad::fmod(2.3 * x1, 2.3 * x2))
TEST(GraphExpressionMath, fmodExprExpr)
{
    mathTest2_graph(1.3, 0.5, std::fmod(2.3 * 1.3, 2.3 * 0.5),
                  2.3,                                // d1
                  -2.3 * std::floor(1.3 / 0.5),  // d2
                  graphFmodExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphFmodADScalar, xad::fmod(x, 0.5))
TEST(GraphExpressionMath, fmodADScalar)
{
    mathTest_graph(1.3, std::fmod(1.3, 0.5), 1.0, graphFmodADScalar);
}

GRAPH_TEST_FUNCTOR2(graphRemainderExprExpr, remainder(2.3 * x1, 2.3 * x2))
TEST(GraphExpressionMath, remainderExprExpr)
{
    int n1;
    auto r1 = std::remquo(2.3 * 1.3, 2.3 * 0.5, &n1);
    mathTest2_graph(1.3, 0.5, r1, 2.3, -2.3 * double(n1), graphRemainderExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphRemainderADScalar, remainder(x, 0.5))
TEST(GraphExpressionMath, remainderADScalar)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_graph(1.3, res, 1.0, graphRemainderADScalar);
}

// =============================================================================
// Nextafter expression variants
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphNextafterExprExpr, nextafter(2.3 * x1, 2.3 * x2))
TEST(GraphExpressionMath, nextafterExprExpr)
{
    mathTest2_graph(0.1, 0.2, std::nextafter(2.3 * 0.1, 2.3 * 0.2), 2.3, 0.0, graphNextafterExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphNextafterADScalar, nextafter(x, 0.2))
TEST(GraphExpressionMath, nextafterADScalar)
{
    mathTest_graph(0.1, std::nextafter(0.1, 0.2), 1.0, graphNextafterADScalar);
}

// =============================================================================
// Scalbn expression variant
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphScalbnExpr, scalbn(x * 2.3, 2))
TEST(GraphExpressionMath, scalbnExpr)
{
    mathTest_graph(0.1, std::scalbn(0.1 * 2.3, 2), std::pow(double(FLT_RADIX), 2.0) * 2.3,
                 graphScalbnExpr);
}

// =============================================================================
// Max/Min with expressions
// =============================================================================

// DISABLED: max/min with expressions have issues with operand slot mapping in the graph
GRAPH_TEST_FUNCTOR2(graphMaxADExpr, max(x1, 2.3 * x2))
TEST(GraphExpressionMath, DISABLED_maxADExpr)
{
    mathTest2_graph(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, graphMaxADExpr);
    mathTest2_graph(1.7, -0.7, 1.7, 1.0, 0.0, graphMaxADExpr);
}

GRAPH_TEST_FUNCTOR2(graphMaxExprAD, max(2.3 * x1, x2))
TEST(GraphExpressionMath, DISABLED_maxExprAD)
{
    mathTest2_graph(0.3, 0.7, 0.7, 0.0, 1.0, graphMaxExprAD);
    mathTest2_graph(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, graphMaxExprAD);
}

GRAPH_TEST_FUNCTOR2(graphMaxExprExpr, max(2.3 * x1, 2.3 * x2))
TEST(GraphExpressionMath, DISABLED_maxExprExpr)
{
    mathTest2_graph(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, graphMaxExprExpr);
    mathTest2_graph(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, graphMaxExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphMaxADScalar, max(x, 0.7))
TEST(GraphExpressionMath, maxADScalar)
{
    mathTest_graph(1.1, 1.1, 1.0, graphMaxADScalar);
    mathTest_graph(0.6, 0.7, 0.0, graphMaxADScalar);
}

GRAPH_TEST_FUNCTOR1(graphMaxExprScalar, max(2.0 * x, 0.7))
TEST(GraphExpressionMath, maxExprScalar)
{
    mathTest_graph(1.1, 2.0 * 1.1, 2.0, graphMaxExprScalar);
    mathTest_graph(0.3, 0.7, 0.0, graphMaxExprScalar);
}

GRAPH_TEST_FUNCTOR2(graphMinADExpr, min(x1, 2.3 * x2))
TEST(GraphExpressionMath, DISABLED_minADExpr)
{
    mathTest2_graph(0.3, 0.7, 0.3, 1.0, 0.0, graphMinADExpr);
    mathTest2_graph(1.7, -0.7, -0.7 * 2.3, 0.0, 2.3, graphMinADExpr);
}

GRAPH_TEST_FUNCTOR2(graphMinExprAD, min(2.3 * x1, x2))
TEST(GraphExpressionMath, DISABLED_minExprAD)
{
    mathTest2_graph(0.5, 0.7, 0.7, 0.0, 1.0, graphMinExprAD);
    mathTest2_graph(1.7, -0.7, -0.7, 0.0, 1.0, graphMinExprAD);
}

GRAPH_TEST_FUNCTOR2(graphMinExprExpr, min(2.3 * x1, 2.3 * x2))
TEST(GraphExpressionMath, DISABLED_minExprExpr)
{
    mathTest2_graph(0.3, 0.7, 2.3 * 0.3, 2.3, 0.0, graphMinExprExpr);
    mathTest2_graph(1.7, -0.7, 2.3 * -0.7, 0.0, 2.3, graphMinExprExpr);
}

GRAPH_TEST_FUNCTOR1(graphMinADScalar, min(x, 0.7))
TEST(GraphExpressionMath, minADScalar)
{
    mathTest_graph(1.1, 0.7, 0.0, graphMinADScalar);
    mathTest_graph(0.6, 0.6, 1.0, graphMinADScalar);
}

GRAPH_TEST_FUNCTOR1(graphMinExprScalar, min(2.0 * x, 0.7))
TEST(GraphExpressionMath, minExprScalar)
{
    mathTest_graph(1.1, 0.7, 0.0, graphMinExprScalar);
    mathTest_graph(0.3, 2.0 * 0.3, 2.0, graphMinExprScalar);
}

// =============================================================================
// Smooth max/min functions
// =============================================================================

GRAPH_TEST_FUNCTOR2(graphSmoothMaxADAD, smooth_max(x1, x2))
TEST(GraphExpressionMath, smoothMaxADAD)
{
    mathTest2_graph(0.3, 0.7, 0.7, 0.0, 1.0, graphSmoothMaxADAD);
    mathTest2_graph(1.7, -0.7, 1.7, 1.0, 0.0, graphSmoothMaxADAD);
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphSmoothMaxADAD);
}

GRAPH_TEST_FUNCTOR1(graphSmoothMaxADScalar, smooth_max(x, 0.7))
TEST(GraphExpressionMath, smoothMaxADScalar)
{
    mathTest_graph(1.1, 1.1, 1.0, graphSmoothMaxADScalar);
    mathTest_graph(0.6, 0.7, 0.0, graphSmoothMaxADScalar);
}

GRAPH_TEST_FUNCTOR2(graphSmoothMinADAD, smooth_min(x1, x2))
TEST(GraphExpressionMath, smoothMinADAD)
{
    mathTest2_graph(0.3, 0.7, 0.3, 1.0, 0.0, graphSmoothMinADAD);
    mathTest2_graph(1.7, -0.7, -0.7, 0.0, 1.0, graphSmoothMinADAD);
    mathTest2_graph(1.7, 1.7, 1.7, 0.5, 0.5, graphSmoothMinADAD);
}

GRAPH_TEST_FUNCTOR1(graphSmoothMinADScalar, smooth_min(x, 0.7))
TEST(GraphExpressionMath, smoothMinADScalar)
{
    mathTest_graph(1.1, 0.7, 0.0, graphSmoothMinADScalar);
    mathTest_graph(0.6, 0.6, 1.0, graphSmoothMinADScalar);
}

// =============================================================================
// Copysign - TODO: needs ABool.If support for graph recording
// The copysign function has conditional logic that is evaluated at recording
// time, not at replay time. This needs to be fixed using ABool.If.
// =============================================================================

struct graphTestFunctor_graphCopysignScalarAD
{
    explicit graphTestFunctor_graphCopysignScalarAD(double op1) : op1_(op1) {}
    double op1_ = 0.0;
    template <class T>
    double operator()(const T& x) const
    {
        return copysign(op1_, x);
    }
};

// DISABLED: copysign(scalar, AD) returns double, causing output count mismatch
TEST(GraphExpressionMath, DISABLED_copysignScalarAD)
{
    mathTest_graph(1.2, 42.2, 0.0, graphTestFunctor_graphCopysignScalarAD(42.2));
    mathTest_graph(-1.2, -42.2, 0.0, graphTestFunctor_graphCopysignScalarAD(42.2));
}

struct graphTestFunctor_graphCopysignADScalar
{
    explicit graphTestFunctor_graphCopysignADScalar(double op2) : op2_(op2) {}
    double op2_ = 0.0;
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, op2_);
    }
};

TEST(GraphExpressionMath, copysignADScalar)
{
    mathTest_graph(1.2, 1.2, 1.0, graphTestFunctor_graphCopysignADScalar(5.9));
    mathTest_graph(1.2, 1.2, 1.0, graphTestFunctor_graphCopysignADScalar(0.0));
    mathTest_graph(1.2, -1.2, -1.0, graphTestFunctor_graphCopysignADScalar(-5.9));
    mathTest_graph(1.2, -1.2, -1.0, graphTestFunctor_graphCopysignADScalar(-0.0000001));
}

struct graphTestFunctor_graphCopysignADAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, x);
    }
} graphCopysignADAD;

TEST(GraphExpressionMath, copysignADAD)
{
    mathTest_graph(1.2, 1.2, 1.0, graphCopysignADAD);
    mathTest_graph(-1.2, -1.2, 1.0, graphCopysignADAD);
}

struct graphTestFunctor_graphCopysignExprAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, x);
    }
} graphCopysignExprAD;

// DISABLED: copysign conditional evaluated at recording time, not replay time
TEST(GraphExpressionMath, DISABLED_copysignExprAD)
{
    mathTest_graph(1.2, 1.2, 1.0, graphCopysignExprAD);
}

struct graphTestFunctor_graphCopysignExprExpr
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, -x);
    }
} graphCopysignExprExpr;

// DISABLED: copysign conditional evaluated at recording time, not replay time
TEST(GraphExpressionMath, DISABLED_copysignExprExpr)
{
    mathTest_graph(1.2, -1.2, -1.0, graphCopysignExprExpr);
}

// =============================================================================
// frexp - pointer output function (writes exponent to pointer at recording time)
// Note: frexp writes to a pointer at recording time, so the replay won't re-execute that write.
// However, the value and derivative of the mantissa should still be correct.
// =============================================================================

struct graphTestFunctor_graphFrexpAD
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(x, &exponent);
    }
} graphFrexpAD;
int graphTestFunctor_graphFrexpAD::exponent = 0;

// DISABLED: frexp writes exponent to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_frexpAD)
{
    mathTest_graph(1024.0, 0.5, 1.0 / (1 << 11), graphFrexpAD);
    EXPECT_EQ(graphTestFunctor_graphFrexpAD::exponent, 11);
}

struct graphTestFunctor_graphFrexpExpr
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(1.0 * x, &exponent);
    }
} graphFrexpExpr;
int graphTestFunctor_graphFrexpExpr::exponent = 0;

// DISABLED: frexp writes exponent to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_frexpExpr)
{
    mathTest_graph(1024.0, 0.5, 1.0 / (1 << 11), graphFrexpExpr);
    EXPECT_EQ(graphTestFunctor_graphFrexpExpr::exponent, 11);
}

// =============================================================================
// modf - pointer output function (writes integer part to pointer at recording time)
// Note: modf writes to a pointer at recording time, so the replay won't re-execute that write.
// However, the value and derivative of the fractional part should still be correct.
// =============================================================================

struct graphTestFunctor_graphModfADScalar
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        return modf(x, &ipart);
    }
} graphModfADScalar;
double graphTestFunctor_graphModfADScalar::ipart = 0.0;

// DISABLED: modf writes integer part to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_modfADScalar)
{
    mathTest_graph(1.2, 0.2, 1.0, graphModfADScalar);
    EXPECT_NEAR(graphTestFunctor_graphModfADScalar::ipart, 1.0, 1e-9);
}

struct graphTestFunctor_graphModfADAD
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        T ipt;
        T ret = modf(x, &ipt);
        ipart = xad::value(xad::value(ipt));
        return ret;
    }
} graphModfADAD;
double graphTestFunctor_graphModfADAD::ipart = 0.0;

// DISABLED: modf writes integer part to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_modfADAD)
{
    mathTest_graph(1.2, 0.2, 1.0, graphModfADAD);
    EXPECT_NEAR(graphTestFunctor_graphModfADAD::ipart, 1.0, 1e-9);
}

// =============================================================================
// remquo - pointer output function (writes quotient to pointer at recording time)
// Note: remquo writes to a pointer at recording time, so the replay won't re-execute that write.
// However, the value and derivative of the remainder should still be correct.
// =============================================================================

int graphRmqn_ = 0;

struct graphTestFunctor_graphRemquoAD
{
    template <class T>
    T operator()(const T& x1, const T& x2) const
    {
        return xad::remquo(x1, x2, &graphRmqn_);
    }
} graphRemquoAD;

// DISABLED: remquo writes quotient to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_remquoAD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest2_graph(1.3, 0.5, res, 1.0, -double(n), graphRemquoAD);
    EXPECT_EQ(n, graphRmqn_);
    graphRmqn_ = 0;
}

struct graphTestFunctor_graphRemquoADScalar
{
    template <class T>
    T operator()(const T& x) const
    {
        return xad::remquo(x, 0.5, &graphRmqn_);
    }
} graphRemquoADScalar;

// DISABLED: remquo writes quotient to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_remquoADScalar)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_graph(1.3, res, 1.0, graphRemquoADScalar);
    EXPECT_EQ(n, graphRmqn_);
    graphRmqn_ = 0;
}

struct graphTestFunctor_graphRemquoScalarAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return xad::remquo(1.3, x, &graphRmqn_);
    }
} graphRemquoScalarAD;

// DISABLED: remquo writes quotient to pointer at recording time, not re-evaluated during replay
TEST(GraphExpressionMath, DISABLED_remquoScalarAD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_graph(0.5, res, -double(n), graphRemquoScalarAD);
    EXPECT_EQ(n, graphRmqn_);
    graphRmqn_ = 0;
}

// =============================================================================
// Scalar operations (scalar + AD, scalar * AD, etc.)
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphScalarAddAD, 2.3 + x)
TEST(GraphExpressionMath, scalarAddAD)
{
    mathTest_graph(1.0, 3.3, 1.0, graphScalarAddAD);
}

GRAPH_TEST_FUNCTOR1(graphScalarSubAD, 2.3 - x)
TEST(GraphExpressionMath, scalarSubAD)
{
    mathTest_graph(1.0, 1.3, -1.0, graphScalarSubAD);
}

GRAPH_TEST_FUNCTOR1(graphScalarMulAD, 2.3 * x)
TEST(GraphExpressionMath, scalarMulAD)
{
    mathTest_graph(1.0, 2.3, 2.3, graphScalarMulAD);
}

GRAPH_TEST_FUNCTOR1(graphScalarDivAD, 2.3 / x)
TEST(GraphExpressionMath, scalarDivAD)
{
    mathTest_graph(1.0, 2.3, -2.3, graphScalarDivAD);
}

GRAPH_TEST_FUNCTOR1(graphADAddScalar, x + 2.3)
TEST(GraphExpressionMath, adAddScalar)
{
    mathTest_graph(1.0, 3.3, 1.0, graphADAddScalar);
}

GRAPH_TEST_FUNCTOR1(graphADSubScalar, x - 2.3)
TEST(GraphExpressionMath, adSubScalar)
{
    mathTest_graph(1.0, -1.3, 1.0, graphADSubScalar);
}

GRAPH_TEST_FUNCTOR1(graphADMulScalar, x * 2.3)
TEST(GraphExpressionMath, adMulScalar)
{
    mathTest_graph(1.0, 2.3, 2.3, graphADMulScalar);
}

GRAPH_TEST_FUNCTOR1(graphADDivScalar, x / 2.3)
TEST(GraphExpressionMath, adDivScalar)
{
    mathTest_graph(1.0, 1.0 / 2.3, 1.0 / 2.3, graphADDivScalar);
}

// =============================================================================
// Negation
// =============================================================================

GRAPH_TEST_FUNCTOR1(graphNegAD, -x)
TEST(GraphExpressionMath, negAD)
{
    mathTest_graph(1.3, -1.3, -1.0, graphNegAD);
}

// =============================================================================
// Fma (fused multiply-add)
// =============================================================================

struct graphTestFunctor_graphFmaADADAD
{
    template <class T>
    T operator()(const T& x1, const T& x2) const
    {
        // fma(a, b, c) = a * b + c, but we only have 2 inputs
        // So test fma(x1, x2, x1) = x1 * x2 + x1
        return fma(x1, x2, x1);
    }
} graphFmaADADAD;

TEST(GraphExpressionMath, fmaADADAD)
{
    // fma(x1, x2, x1) = x1 * x2 + x1
    // d/dx1 = x2 + 1
    // d/dx2 = x1
    mathTest2_graph(1.3, 0.7, 1.3 * 0.7 + 1.3, 0.7 + 1.0, 1.3, graphFmaADADAD);
}

#undef GRAPH_TEST_FUNCTOR1
#undef GRAPH_TEST_FUNCTOR2

#endif  // XAD_ENABLE_GRAPH

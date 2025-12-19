/*******************************************************************************

   Unit tests for JIT compilation of math function derivatives.

   This file tests JIT-specific behavior for math functions. Only functions
   that work correctly with JIT compilation are included here.

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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include "TestHelpers.hpp"

#ifdef XAD_ENABLE_JIT

// Helper macros for JIT test functors
#define JIT_TEST_FUNCTOR1(name, val)                                                               \
    struct jitTestFunctor_##name                                                                   \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x) const                                                             \
        {                                                                                          \
            using namespace std;                                                                   \
            return val;                                                                            \
        }                                                                                          \
    } name;

#define JIT_TEST_FUNCTOR2(name, val)                                                               \
    struct jitTestFunctor_##name                                                                   \
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

JIT_TEST_FUNCTOR1(jitCosAD, cos(x))
TEST(ExpressionsMathJit, cosAD)
{
    mathTest_jit(1.0, std::cos(1.0), -std::sin(1.0), jitCosAD);
}

JIT_TEST_FUNCTOR1(jitCosExpr, cos(2.3 * x))
TEST(ExpressionsMathJit, cosExpr)
{
    mathTest_jit(1.0, std::cos(2.3), -2.3 * std::sin(2.3), jitCosExpr);
}

JIT_TEST_FUNCTOR1(jitSinAD, sin(x))
TEST(ExpressionsMathJit, sinAD)
{
    mathTest_jit(1.0, std::sin(1.0), std::cos(1.0), jitSinAD);
}

JIT_TEST_FUNCTOR1(jitSinExpr, sin(2.3 * x))
TEST(ExpressionsMathJit, sinExpr)
{
    mathTest_jit(1.0, std::sin(2.3), 2.3 * std::cos(2.3), jitSinExpr);
}

JIT_TEST_FUNCTOR1(jitExpAD, exp(x))
TEST(ExpressionsMathJit, expAD)
{
    mathTest_jit(1.0, std::exp(1.0), std::exp(1.0), jitExpAD);
}

JIT_TEST_FUNCTOR1(jitExpExpr, exp(2.3 * x))
TEST(ExpressionsMathJit, expExpr)
{
    mathTest_jit(1.0, std::exp(2.3), 2.3 * std::exp(2.3), jitExpExpr);
}

JIT_TEST_FUNCTOR1(jitLogAD, log(x))
TEST(ExpressionsMathJit, logAD)
{
    mathTest_jit(1.3, std::log(1.3), 1.0 / 1.3, jitLogAD);
}

JIT_TEST_FUNCTOR1(jitLogExpr, log(2.3 * x))
TEST(ExpressionsMathJit, logExpr)
{
    mathTest_jit(1.0, std::log(2.3), 1.0, jitLogExpr);
}

JIT_TEST_FUNCTOR1(jitLog10AD, log10(x))
TEST(ExpressionsMathJit, log10AD)
{
    mathTest_jit(1.3, std::log10(1.3), 1.0 / std::log(10.0) / 1.3, jitLog10AD);
}

JIT_TEST_FUNCTOR1(jitLog2AD, log2(x))
TEST(ExpressionsMathJit, log2AD)
{
    using xad::log2;
    mathTest_jit(1.3, log2(1.3), 1.0 / log(2.0) / 1.3, jitLog2AD);
}

JIT_TEST_FUNCTOR1(jitSqrtAD, sqrt(x))
TEST(ExpressionsMathJit, sqrtAD)
{
    mathTest_jit(1.3, std::sqrt(1.3), 0.5 / std::sqrt(1.3), jitSqrtAD);
}

JIT_TEST_FUNCTOR1(jitSqrtExpr, sqrt(2.3 * x))
TEST(ExpressionsMathJit, sqrtExpr)
{
    mathTest_jit(1.3, std::sqrt(2.3 * 1.3), 2.3 * 0.5 / std::sqrt(2.3 * 1.3), jitSqrtExpr);
}

JIT_TEST_FUNCTOR1(jitCbrtAD, cbrt(x))
TEST(ExpressionsMathJit, cbrtAD)
{
    mathTest_jit(1.3, std::cbrt(1.3), 1.0 / 3.0 / std::cbrt(1.3) / std::cbrt(1.3), jitCbrtAD);
}

// =============================================================================
// Trigonometric functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitTanAD, tan(x))
TEST(ExpressionsMathJit, tanAD)
{
    mathTest_jit(0.3, std::tan(0.3), 1.0 / std::cos(0.3) / std::cos(0.3), jitTanAD);
}

JIT_TEST_FUNCTOR1(jitAsinAD, asin(x))
TEST(ExpressionsMathJit, asinAD)
{
    mathTest_jit(0.3, std::asin(0.3), 1.0 / std::sqrt(1.0 - 0.3 * 0.3), jitAsinAD);
}

JIT_TEST_FUNCTOR1(jitAcosAD, acos(x))
TEST(ExpressionsMathJit, acosAD)
{
    mathTest_jit(0.3, std::acos(0.3), -1.0 / std::sqrt(1.0 - 0.3 * 0.3), jitAcosAD);
}

JIT_TEST_FUNCTOR1(jitAtanAD, atan(x))
TEST(ExpressionsMathJit, atanAD)
{
    mathTest_jit(0.3, std::atan(0.3), 1.0 / (1.0 + 0.3 * 0.3), jitAtanAD);
}

// =============================================================================
// Hyperbolic functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitSinhAD, sinh(x))
TEST(ExpressionsMathJit, sinhAD)
{
    mathTest_jit(0.3, std::sinh(0.3), std::cosh(0.3), jitSinhAD);
}

JIT_TEST_FUNCTOR1(jitCoshAD, cosh(x))
TEST(ExpressionsMathJit, coshAD)
{
    mathTest_jit(0.3, std::cosh(0.3), std::sinh(0.3), jitCoshAD);
}

JIT_TEST_FUNCTOR1(jitTanhAD, tanh(x))
TEST(ExpressionsMathJit, tanhAD)
{
    mathTest_jit(0.3, std::tanh(0.3), 1.0 - std::tanh(0.3) * std::tanh(0.3), jitTanhAD);
}

JIT_TEST_FUNCTOR1(jitAsinhAD, asinh(x))
TEST(ExpressionsMathJit, asinhAD)
{
    using xad::asinh;
    mathTest_jit(0.3, asinh(0.3), 1.0 / std::sqrt(1.0 + 0.3 * 0.3), jitAsinhAD);
}

JIT_TEST_FUNCTOR1(jitAcoshAD, acosh(x))
TEST(ExpressionsMathJit, acoshAD)
{
    using xad::acosh;
    mathTest_jit(1.3, acosh(1.3), 1.0 / std::sqrt(1.3 * 1.3 - 1.0), jitAcoshAD);
}

JIT_TEST_FUNCTOR1(jitAtanhAD, atanh(x))
TEST(ExpressionsMathJit, atanhAD)
{
    using xad::atanh;
    mathTest_jit(0.3, atanh(0.3), 1.0 / (1.0 - 0.3 * 0.3), jitAtanhAD);
}

// =============================================================================
// Special functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitErfAD, erf(x))
TEST(ExpressionsMathJit, erfAD)
{
    mathTest_jit(0.3, std::erf(0.3), 2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3), jitErfAD);
}

JIT_TEST_FUNCTOR1(jitErfcAD, erfc(x))
TEST(ExpressionsMathJit, erfcAD)
{
    mathTest_jit(0.3, std::erfc(0.3), -2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3), jitErfcAD);
}

JIT_TEST_FUNCTOR1(jitExpm1AD, expm1(x))
TEST(ExpressionsMathJit, expm1AD)
{
    using xad::expm1;
    mathTest_jit(0.3, expm1(0.3), std::exp(0.3), jitExpm1AD);
}

JIT_TEST_FUNCTOR1(jitLog1pAD, log1p(x))
TEST(ExpressionsMathJit, log1pAD)
{
    using xad::log1p;
    mathTest_jit(0.3, log1p(0.3), 1.0 / (1.0 + 0.3), jitLog1pAD);
}

JIT_TEST_FUNCTOR1(jitExp2AD, exp2(x))
TEST(ExpressionsMathJit, exp2AD)
{
    using xad::exp2;
    mathTest_jit(0.3, exp2(0.3), std::log(2.0) * exp2(0.3), jitExp2AD);
}

// =============================================================================
// Rounding functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitFloorAD, floor(x))
TEST(ExpressionsMathJit, floorAD)
{
    mathTest_jit(1.7, 1.0, 0.0, jitFloorAD);
}

JIT_TEST_FUNCTOR1(jitCeilAD, ceil(x))
TEST(ExpressionsMathJit, ceilAD)
{
    mathTest_jit(1.3, 2.0, 0.0, jitCeilAD);
}

JIT_TEST_FUNCTOR1(jitTruncAD, trunc(x))
TEST(ExpressionsMathJit, truncAD)
{
    using xad::trunc;
    mathTest_jit(1.7, 1.0, 0.0, jitTruncAD);
}

JIT_TEST_FUNCTOR1(jitRoundAD, round(x))
TEST(ExpressionsMathJit, roundAD)
{
    using xad::round;
    mathTest_jit(1.7, 2.0, 0.0, jitRoundAD);
}

// =============================================================================
// Absolute value (with special handling at x=0)
// =============================================================================

JIT_TEST_FUNCTOR1(jitAbsAD, abs(x))
TEST(ExpressionsMathJit, absAD)
{
    mathTest_jit(1.3, 1.3, 1.0, jitAbsAD);
    mathTest_jit(-1.3, 1.3, -1.0, jitAbsAD);
    mathTest_jit(0.0, 0.0, 0.0, jitAbsAD);  // derivative at 0 is 0
}

JIT_TEST_FUNCTOR1(jitFabsAD, fabs(x))
TEST(ExpressionsMathJit, fabsAD)
{
    mathTest_jit(1.3, 1.3, 1.0, jitFabsAD);
    mathTest_jit(-1.3, 1.3, -1.0, jitFabsAD);
    mathTest_jit(0.0, 0.0, 0.0, jitFabsAD);  // derivative at 0 is 0
}

// =============================================================================
// Power functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitPowScalarExpAD, pow(x, 2.1))
TEST(ExpressionsMathJit, powScalarExpAD)
{
    mathTest_jit(0.3, std::pow(0.3, 2.1), 2.1 * std::pow(0.3, 1.1), jitPowScalarExpAD);
}

JIT_TEST_FUNCTOR1(jitPowScalarBaseAD, pow(2.1, x))
TEST(ExpressionsMathJit, powScalarBaseAD)
{
    mathTest_jit(0.3, std::pow(2.1, 0.3), std::log(2.1) * std::pow(2.1, 0.3), jitPowScalarBaseAD);
}

JIT_TEST_FUNCTOR2(jitPowADAD, pow(x1, x2))
TEST(ExpressionsMathJit, powADAD)
{
    mathTest2_jit(0.3, 2.1, std::pow(0.3, 2.1),
                  2.1 * std::pow(0.3, 1.1),              // d1
                  std::log(0.3) * std::pow(0.3, 2.1),    // d2
                  jitPowADAD);
}

// =============================================================================
// Two-variable functions
// =============================================================================

JIT_TEST_FUNCTOR2(jitAddADAD, x1 + x2)
TEST(ExpressionsMathJit, addADAD)
{
    mathTest2_jit(1.3, 0.7, 2.0, 1.0, 1.0, jitAddADAD);
}

JIT_TEST_FUNCTOR2(jitSubADAD, x1 - x2)
TEST(ExpressionsMathJit, subADAD)
{
    mathTest2_jit(1.3, 0.7, 0.6, 1.0, -1.0, jitSubADAD);
}

JIT_TEST_FUNCTOR2(jitMulADAD, x1 * x2)
TEST(ExpressionsMathJit, mulADAD)
{
    mathTest2_jit(1.3, 0.7, 1.3 * 0.7, 0.7, 1.3, jitMulADAD);
}

JIT_TEST_FUNCTOR2(jitDivADAD, x1 / x2)
TEST(ExpressionsMathJit, divADAD)
{
    mathTest2_jit(1.3, 0.7, 1.3 / 0.7, 1.0 / 0.7, -1.3 / (0.7 * 0.7), jitDivADAD);
}

JIT_TEST_FUNCTOR2(jitAtan2AD, xad::atan2(x1, x2))
TEST(ExpressionsMathJit, atan2AD)
{
    mathTest2_jit(0.3, 0.5, std::atan2(0.3, 0.5),
                  0.5 / (0.3 * 0.3 + 0.5 * 0.5),   // d1
                  -0.3 / (0.3 * 0.3 + 0.5 * 0.5),  // d2
                  jitAtan2AD);
}

JIT_TEST_FUNCTOR2(jitHypotAD, hypot(x1, x2))
TEST(ExpressionsMathJit, hypotAD)
{
    mathTest2_jit(0.3, 0.5, std::hypot(0.3, 0.5),
                  0.3 / std::hypot(0.3, 0.5),  // d1
                  0.5 / std::hypot(0.3, 0.5),  // d2
                  jitHypotAD);
}

JIT_TEST_FUNCTOR2(jitFmodAD, fmod(x1, x2))
TEST(ExpressionsMathJit, fmodAD)
{
    int n = static_cast<int>(1.3 / 0.5);
    mathTest2_jit(1.3, 0.5, std::fmod(1.3, 0.5),
                  1.0,             // d1
                  -double(n),      // d2
                  jitFmodAD);
}

JIT_TEST_FUNCTOR2(jitRemainderAD, remainder(x1, x2))
TEST(ExpressionsMathJit, remainderAD)
{
    int n = static_cast<int>(std::round(1.3 / 0.5));
    mathTest2_jit(1.3, 0.5, std::remainder(1.3, 0.5),
                  1.0,             // d1
                  -double(n),      // d2
                  jitRemainderAD);
}

JIT_TEST_FUNCTOR2(jitNextafterAD, nextafter(x1, x2))
TEST(ExpressionsMathJit, nextafterAD)
{
    mathTest2_jit(0.3, 0.5, std::nextafter(0.3, 0.5),
                  1.0,   // d1
                  0.0,   // d2
                  jitNextafterAD);
}

// =============================================================================
// Max/Min functions (AD vs AD - equal values case with 0.5/0.5 derivative split)
// =============================================================================

JIT_TEST_FUNCTOR2(jitMaxADAD, max(x1, x2))
TEST(ExpressionsMathJit, maxADAD)
{
    // x1 > x2: derivative flows to x1
    mathTest2_jit(1.7, 0.7, 1.7, 1.0, 0.0, jitMaxADAD);
    // x1 < x2: derivative flows to x2
    mathTest2_jit(0.3, 0.7, 0.7, 0.0, 1.0, jitMaxADAD);
    // x1 == x2: derivative splits 0.5/0.5
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitMaxADAD);
}

JIT_TEST_FUNCTOR2(jitMinADAD, min(x1, x2))
TEST(ExpressionsMathJit, minADAD)
{
    // x1 < x2: derivative flows to x1
    mathTest2_jit(0.3, 0.7, 0.3, 1.0, 0.0, jitMinADAD);
    // x1 > x2: derivative flows to x2
    mathTest2_jit(1.7, 0.7, 0.7, 0.0, 1.0, jitMinADAD);
    // x1 == x2: derivative splits 0.5/0.5
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitMinADAD);
}

JIT_TEST_FUNCTOR2(jitFmaxADAD, fmax(x1, x2))
TEST(ExpressionsMathJit, fmaxADAD)
{
    mathTest2_jit(0.3, 0.7, 0.7, 0.0, 1.0, jitFmaxADAD);
    mathTest2_jit(1.7, 0.7, 1.7, 1.0, 0.0, jitFmaxADAD);
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitFmaxADAD);
}

JIT_TEST_FUNCTOR2(jitFminADAD, fmin(x1, x2))
TEST(ExpressionsMathJit, fminADAD)
{
    mathTest2_jit(0.3, 0.7, 0.3, 1.0, 0.0, jitFminADAD);
    mathTest2_jit(1.7, 0.7, 0.7, 0.0, 1.0, jitFminADAD);
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitFminADAD);
}

// =============================================================================
// Ldexp (x * 2^exp) - works because exp is a compile-time integer
// =============================================================================

JIT_TEST_FUNCTOR1(jitLdexpAD, ldexp(x, 3))
TEST(ExpressionsMathJit, ldexpAD)
{
    mathTest_jit(1.1, 1.1 * 8, 8.0, jitLdexpAD);
}

JIT_TEST_FUNCTOR1(jitLdexpExpr, ldexp(2.0 * x, 3))
TEST(ExpressionsMathJit, ldexpExpr)
{
    mathTest_jit(1.1, 2.2 * 8, 16.0, jitLdexpExpr);
}

// =============================================================================
// Scalbn (similar to ldexp)
// =============================================================================

JIT_TEST_FUNCTOR1(jitScalbnAD, scalbn(x, 3))
TEST(ExpressionsMathJit, scalbnAD)
{
    mathTest_jit(1.1, std::scalbn(1.1, 3), std::scalbn(1.0, 3), jitScalbnAD);
}

// =============================================================================
// Degrees and Radians
// =============================================================================

JIT_TEST_FUNCTOR1(jitDegreesAD, degrees(x))
TEST(ExpressionsMathJit, degreesAD)
{
    mathTest_jit(3.141592653589793238462643, 180.0,
                 57.2957795130823208767981548141051703324054725, jitDegreesAD);
}

JIT_TEST_FUNCTOR1(jitDegreesExpr, degrees(0.5 * x))
TEST(ExpressionsMathJit, degreesExpr)
{
    mathTest_jit(3.141592653589793238462643, 90.0,
                 0.5 * 57.2957795130823208767981548141051703324054725, jitDegreesExpr);
}

JIT_TEST_FUNCTOR1(jitRadiansAD, radians(x))
TEST(ExpressionsMathJit, radiansAD)
{
    mathTest_jit(180.0, 3.141592653589793238462643,
                 0.0174532925199432957692369076848861271344287, jitRadiansAD);
}

JIT_TEST_FUNCTOR1(jitRadiansExpr, radians(2.0 * x))
TEST(ExpressionsMathJit, radiansExpr)
{
    mathTest_jit(180.0, 2.0 * 3.141592653589793238462643,
                 2.0 * 0.0174532925199432957692369076848861271344287, jitRadiansExpr);
}

// =============================================================================
// Expression variants for trig functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitTanExpr, tan(2.3 * x))
TEST(ExpressionsMathJit, tanExpr)
{
    mathTest_jit(0.3, std::tan(2.3 * 0.3), 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3),
                 jitTanExpr);
}

JIT_TEST_FUNCTOR1(jitAsinExpr, asin(2.3 * x))
TEST(ExpressionsMathJit, asinExpr)
{
    mathTest_jit(0.1, std::asin(0.1 * 2.3), 2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 jitAsinExpr);
}

JIT_TEST_FUNCTOR1(jitAcosExpr, acos(2.3 * x))
TEST(ExpressionsMathJit, acosExpr)
{
    mathTest_jit(0.1, std::acos(0.1 * 2.3), -2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 jitAcosExpr);
}

JIT_TEST_FUNCTOR1(jitAtanExpr, atan(2.3 * x))
TEST(ExpressionsMathJit, atanExpr)
{
    mathTest_jit(0.1, std::atan(0.1 * 2.3), 2.3 / (1.0 + 2.3 * 2.3 * 0.1 * 0.1), jitAtanExpr);
}

// =============================================================================
// Expression variants for hyperbolic functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitSinhExpr, sinh(2.3 * x))
TEST(ExpressionsMathJit, sinhExpr)
{
    mathTest_jit(0.1, std::sinh(0.1 * 2.3), 2.3 * std::cosh(2.3 * 0.1), jitSinhExpr);
}

JIT_TEST_FUNCTOR1(jitCoshExpr, cosh(2.3 * x))
TEST(ExpressionsMathJit, coshExpr)
{
    mathTest_jit(0.3, std::cosh(2.3 * 0.3), 2.3 * std::sinh(2.3 * 0.3), jitCoshExpr);
}

JIT_TEST_FUNCTOR1(jitTanhExpr, tanh(2.3 * x))
TEST(ExpressionsMathJit, tanhExpr)
{
    mathTest_jit(0.3, std::tanh(2.3 * 0.3),
                 2.3 * (1.0 - std::tanh(2.3 * 0.3) * std::tanh(2.3 * 0.3)), jitTanhExpr);
}

JIT_TEST_FUNCTOR1(jitAsinhExpr, asinh(2.3 * x))
TEST(ExpressionsMathJit, asinhExpr)
{
    using xad::asinh;
    mathTest_jit(0.3, asinh(2.3 * 0.3), 2.3 / std::sqrt(1.0 + 2.3 * 2.3 * 0.3 * 0.3), jitAsinhExpr);
}

JIT_TEST_FUNCTOR1(jitAcoshExpr, acosh(2.3 * x))
TEST(ExpressionsMathJit, acoshExpr)
{
    using xad::acosh;
    mathTest_jit(1.3, acosh(2.3 * 1.3), 2.3 / std::sqrt(2.3 * 2.3 * 1.3 * 1.3 - 1.0), jitAcoshExpr);
}

JIT_TEST_FUNCTOR1(jitAtanhExpr, atanh(2.3 * x))
TEST(ExpressionsMathJit, atanhExpr)
{
    using xad::atanh;
    mathTest_jit(0.3, atanh(2.3 * 0.3), 2.3 / (1.0 - 2.3 * 2.3 * 0.3 * 0.3), jitAtanhExpr);
}

// =============================================================================
// Expression variants for special functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitErfExpr, erf(2.3 * x))
TEST(ExpressionsMathJit, erfExpr)
{
    mathTest_jit(0.3, std::erf(2.3 * 0.3),
                 2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3), jitErfExpr);
}

JIT_TEST_FUNCTOR1(jitErfcExpr, erfc(2.3 * x))
TEST(ExpressionsMathJit, erfcExpr)
{
    mathTest_jit(0.3, std::erfc(2.3 * 0.3),
                 -2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3), jitErfcExpr);
}

JIT_TEST_FUNCTOR1(jitExpm1Expr, expm1(2.3 * x))
TEST(ExpressionsMathJit, expm1Expr)
{
    using xad::expm1;
    mathTest_jit(0.3, expm1(2.3 * 0.3), 2.3 * std::exp(2.3 * 0.3), jitExpm1Expr);
}

JIT_TEST_FUNCTOR1(jitLog1pExpr, log1p(2.3 * x))
TEST(ExpressionsMathJit, log1pExpr)
{
    using xad::log1p;
    mathTest_jit(0.3, log1p(2.3 * 0.3), 2.3 / (1.0 + 2.3 * 0.3), jitLog1pExpr);
}

JIT_TEST_FUNCTOR1(jitExp2Expr, exp2(2.3 * x))
TEST(ExpressionsMathJit, exp2Expr)
{
    using xad::exp2;
    mathTest_jit(0.3, exp2(2.3 * 0.3), 2.3 * std::log(2.0) * exp2(2.3 * 0.3), jitExp2Expr);
}

JIT_TEST_FUNCTOR1(jitLog10Expr, log10(2.3 * x))
TEST(ExpressionsMathJit, log10Expr)
{
    mathTest_jit(1.0, std::log10(2.3), 2.3 / std::log(10.0) / 2.3, jitLog10Expr);
}

JIT_TEST_FUNCTOR1(jitLog2Expr, log2(2.3 * x))
TEST(ExpressionsMathJit, log2Expr)
{
    using xad::log2;
    mathTest_jit(1.3, log2(2.3 * 1.3), 2.3 / log(2.0) / 1.3 / 2.3, jitLog2Expr);
}

// =============================================================================
// Expression variants for rounding functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitFloorExpr, floor(2.3 * x))
TEST(ExpressionsMathJit, floorExpr)
{
    mathTest_jit(1.3, std::floor(2.3 * 1.3), 0.0, jitFloorExpr);
}

JIT_TEST_FUNCTOR1(jitCeilExpr, ceil(2.3 * x))
TEST(ExpressionsMathJit, ceilExpr)
{
    mathTest_jit(1.3, std::ceil(2.3 * 1.3), 0.0, jitCeilExpr);
}

JIT_TEST_FUNCTOR1(jitTruncExpr, trunc(2.3 * x))
TEST(ExpressionsMathJit, truncExpr)
{
    using xad::trunc;
    mathTest_jit(1.3, trunc(2.3 * 1.3), 0.0, jitTruncExpr);
}

JIT_TEST_FUNCTOR1(jitRoundExpr, round(2.3 * x))
TEST(ExpressionsMathJit, roundExpr)
{
    using xad::round;
    mathTest_jit(1.3, round(2.3 * 1.3), 0.0, jitRoundExpr);
}

// =============================================================================
// Absolute value expression variants
// =============================================================================

JIT_TEST_FUNCTOR1(jitAbsExpr, abs(2.3 * x))
TEST(ExpressionsMathJit, absExpr)
{
    mathTest_jit(0.3, std::abs(2.3 * 0.3), 2.3, jitAbsExpr);
    mathTest_jit(-0.3, std::abs(2.3 * -0.3), -2.3, jitAbsExpr);
    mathTest_jit(0.0, std::abs(2.3 * 0.0), 0.0, jitAbsExpr);
}

JIT_TEST_FUNCTOR1(jitFabsExpr, fabs(2.3 * x))
TEST(ExpressionsMathJit, fabsExpr)
{
    mathTest_jit(0.3, std::fabs(2.3 * 0.3), 2.3, jitFabsExpr);
    mathTest_jit(-0.3, std::fabs(2.3 * -0.3), -2.3, jitFabsExpr);
    mathTest_jit(0.0, std::fabs(2.3 * 0.0), 0.0, jitFabsExpr);
}

// =============================================================================
// Smooth absolute value functions
// =============================================================================

JIT_TEST_FUNCTOR1(jitSmoothAbsAD, smooth_abs(x))
TEST(ExpressionsMathJit, smoothAbsAD)
{
    mathTest_jit(0.3, std::abs(0.3), 1.0, jitSmoothAbsAD);
    mathTest_jit(-0.3, std::abs(-0.3), -1.0, jitSmoothAbsAD);
    mathTest_jit(0.0, std::abs(0.0), 0.0, jitSmoothAbsAD);
}

JIT_TEST_FUNCTOR1(jitSmoothAbsExpr, smooth_abs(2.3 * x))
TEST(ExpressionsMathJit, smoothAbsExpr)
{
    mathTest_jit(0.3, std::abs(2.3 * 0.3), 2.3, jitSmoothAbsExpr);
    mathTest_jit(-0.3, std::abs(2.3 * -0.3), -2.3, jitSmoothAbsExpr);
    mathTest_jit(0.0, std::abs(2.3 * 0.0), 0.0, jitSmoothAbsExpr);
}

JIT_TEST_FUNCTOR2(jitSmoothAbsADAD, smooth_abs(x1, x2))
TEST(ExpressionsMathJit, smoothAbsADAD)
{
    mathTest2_jit(0.3, 0.001, 0.3, 1.0, 0.0, jitSmoothAbsADAD);
    mathTest2_jit(-0.3, 0.001, 0.3, -1.0, 0.0, jitSmoothAbsADAD);
    mathTest2_jit(0.0, 0.001, 0.0, 0.0, 0.0, jitSmoothAbsADAD);
}

// =============================================================================
// Power function expression variants
// =============================================================================

JIT_TEST_FUNCTOR1(jitPowScalarBaseExpr, pow(2.1, 2.3 * x))
TEST(ExpressionsMathJit, powScalarBaseExpr)
{
    mathTest_jit(0.3, std::pow(2.1, 2.3 * 0.3), 2.3 * std::log(2.1) * std::pow(2.1, 2.3 * 0.3),
                 jitPowScalarBaseExpr);
}

JIT_TEST_FUNCTOR1(jitPowScalarExpExpr, pow(2.3 * x, 2.1))
TEST(ExpressionsMathJit, powScalarExpExpr)
{
    mathTest_jit(0.3, std::pow(2.3 * 0.3, 2.1), 2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1),
                 jitPowScalarExpExpr);
}

JIT_TEST_FUNCTOR2(jitPowADExpr, pow(x1, 2.3 * x2))
TEST(ExpressionsMathJit, powADExpr)
{
    mathTest2_jit(1.3, 2.1, std::pow(1.3, 2.3 * 2.1),
                  2.3 * 2.1 * std::pow(1.3, 2.3 * 2.1 - 1.0),      // d1
                  2.3 * std::log(1.3) * std::pow(1.3, 2.3 * 2.1),  // d2
                  jitPowADExpr);
}

JIT_TEST_FUNCTOR2(jitPowExprAD, pow(2.3 * x1, x2))
TEST(ExpressionsMathJit, powExprAD)
{
    mathTest2_jit(0.3, 2.1, std::pow(2.3 * 0.3, 2.1),
                  2.3 * 2.1 * std::pow(2.3 * 0.3, 1.1),          // d1
                  std::log(2.3 * 0.3) * std::pow(2.3 * 0.3, 2.1),  // d2
                  jitPowExprAD);
}

JIT_TEST_FUNCTOR2(jitPowExprExpr, pow(1.2 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, powExprExpr)
{
    mathTest2_jit(0.3, 2.1, std::pow(1.2 * 0.3, 2.3 * 2.1),
                  1.2 * 2.3 * 2.1 * std::pow(1.2 * 0.3, 2.3 * 2.1 - 1.0),
                  2.3 * std::log(1.2 * 0.3) * std::pow(1.2 * 0.3, 2.3 * 2.1), jitPowExprExpr);
}

JIT_TEST_FUNCTOR1(jitPownAD, pown(x, 2))
TEST(ExpressionsMathJit, pownAD)
{
    mathTest_jit(0.3, std::pow(0.3, 2), 2. * 0.3, jitPownAD);
}

JIT_TEST_FUNCTOR1(jitPownExpr, pown(2.3 * x, 2))
TEST(ExpressionsMathJit, pownExpr)
{
    mathTest_jit(0.3, std::pow(2.3 * 0.3, 2), 2.3 * 2 * 2.3 * 0.3, jitPownExpr);
}

JIT_TEST_FUNCTOR1(jitPown1AD, pow(x, 2))
TEST(ExpressionsMathJit, pown1AD)
{
    mathTest_jit(0.3, std::pow(0.3, 2), 2 * std::pow(0.3, 1), jitPown1AD);
}

JIT_TEST_FUNCTOR1(jitPown1Expr, pow(2.3 * x, 2))
TEST(ExpressionsMathJit, pown1Expr)
{
    mathTest_jit(0.3, std::pow(2.3 * 0.3, 2), 2.3 * 2 * std::pow(2.3 * 0.3, 1), jitPown1Expr);
}

// =============================================================================
// Cbrt expression variant
// =============================================================================

JIT_TEST_FUNCTOR1(jitCbrtExpr, cbrt(2.1 * x))
TEST(ExpressionsMathJit, cbrtExpr)
{
    using xad::cbrt;
    mathTest_jit(1.3, cbrt(2.1 * 1.3), 2.1 / 3.0 / std::pow(2.1 * 1.3, 2.0 / 3.0), jitCbrtExpr);
}

// =============================================================================
// Two-variable function expression variants
// =============================================================================

JIT_TEST_FUNCTOR2(jitAtan2ExprExpr, xad::atan2(1.3 * x1, 1.3 * x2))
TEST(ExpressionsMathJit, atan2ExprExpr)
{
    mathTest2_jit(0.3, 0.5, std::atan2(1.3 * 0.3, 1.3 * 0.5),
                  1.3 * 0.5 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),   // d1
                  -1.3 * 0.3 / (1.3 * 0.3 * 0.3 + 1.3 * 0.5 * 0.5),  // d2
                  jitAtan2ExprExpr);
}

JIT_TEST_FUNCTOR1(jitAtan2ADScalar, xad::atan2(x, 0.5))
TEST(ExpressionsMathJit, atan2ADScalar)
{
    mathTest_jit(0.3, std::atan2(0.3, 0.5), 0.5 / (0.3 * 0.3 + 0.5 * 0.5), jitAtan2ADScalar);
}

JIT_TEST_FUNCTOR1(jitAtan2ScalarAD, xad::atan2(0.3, x))
TEST(ExpressionsMathJit, atan2ScalarAD)
{
    mathTest_jit(0.5, std::atan2(0.3, 0.5), -0.3 / (0.3 * 0.3 + 0.5 * 0.5), jitAtan2ScalarAD);
}

JIT_TEST_FUNCTOR2(jitHypotExprExpr, xad::hypot(1.3 * x1, 1.3 * x2))
TEST(ExpressionsMathJit, hypotExprExpr)
{
    mathTest2_jit(0.3, 0.5, std::hypot(1.3 * 0.3, 1.3 * 0.5),
                  1.3 * 1.3 * 0.3 / std::hypot(1.3 * 0.3, 1.3 * 0.5),  // d1
                  1.3 * 1.3 * 0.5 / std::hypot(1.3 * 0.3, 1.3 * 0.5),  // d2
                  jitHypotExprExpr);
}

JIT_TEST_FUNCTOR1(jitHypotADScalar, xad::hypot(x, 0.5))
TEST(ExpressionsMathJit, hypotADScalar)
{
    mathTest_jit(0.3, std::hypot(0.3, 0.5), 0.3 / std::hypot(0.3, 0.5), jitHypotADScalar);
}

JIT_TEST_FUNCTOR1(jitHypotScalarAD, xad::hypot(0.3, x))
TEST(ExpressionsMathJit, hypotScalarAD)
{
    mathTest_jit(0.5, std::hypot(0.3, 0.5), 0.5 / std::hypot(0.3, 0.5), jitHypotScalarAD);
}

JIT_TEST_FUNCTOR2(jitFmodExprExpr, xad::fmod(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, fmodExprExpr)
{
    mathTest2_jit(1.3, 0.5, std::fmod(2.3 * 1.3, 2.3 * 0.5),
                  2.3,                                // d1
                  -2.3 * std::floor(1.3 / 0.5),  // d2
                  jitFmodExprExpr);
}

JIT_TEST_FUNCTOR1(jitFmodADScalar, xad::fmod(x, 0.5))
TEST(ExpressionsMathJit, fmodADScalar)
{
    mathTest_jit(1.3, std::fmod(1.3, 0.5), 1.0, jitFmodADScalar);
}

JIT_TEST_FUNCTOR1(jitFmodScalarAD, xad::fmod(1.3, x))
TEST(ExpressionsMathJit, fmodScalarAD)
{
    mathTest_jit(0.5, std::fmod(1.3, 0.5), -std::floor(1.3 / 0.5), jitFmodScalarAD);
}

JIT_TEST_FUNCTOR2(jitRemainderExprExpr, remainder(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, remainderExprExpr)
{
    int n1;
    auto r1 = std::remquo(2.3 * 1.3, 2.3 * 0.5, &n1);
    mathTest2_jit(1.3, 0.5, r1, 2.3, -2.3 * double(n1), jitRemainderExprExpr);
}

JIT_TEST_FUNCTOR1(jitRemainderADScalar, remainder(x, 0.5))
TEST(ExpressionsMathJit, remainderADScalar)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_jit(1.3, res, 1.0, jitRemainderADScalar);
}

JIT_TEST_FUNCTOR1(jitRemainderScalarAD, remainder(1.3, x))
TEST(ExpressionsMathJit, remainderScalarAD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_jit(0.5, res, -double(n), jitRemainderScalarAD);
}

// =============================================================================
// Nextafter expression variants
// =============================================================================

JIT_TEST_FUNCTOR2(jitNextafterADExpr, nextafter(x1, 2.3 * x2))
TEST(ExpressionsMathJit, nextafterADExpr)
{
    mathTest2_jit(0.1, 0.2, std::nextafter(0.1, 2.3 * 0.2), 1.0, 0.0, jitNextafterADExpr);
}

JIT_TEST_FUNCTOR2(jitNextafterExprAD, nextafter(2.3 * x1, x2))
TEST(ExpressionsMathJit, nextafterExprAD)
{
    mathTest2_jit(0.1, 0.2, std::nextafter(2.3 * 0.1, 0.2), 2.3, 0.0, jitNextafterExprAD);
}

JIT_TEST_FUNCTOR2(jitNextafterExprExpr, nextafter(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, nextafterExprExpr)
{
    mathTest2_jit(0.1, 0.2, std::nextafter(2.3 * 0.1, 2.3 * 0.2), 2.3, 0.0, jitNextafterExprExpr);
}

JIT_TEST_FUNCTOR1(jitNextafterScalarAD, nextafter(0.1, x))
TEST(ExpressionsMathJit, nextafterScalarAD)
{
    mathTest_jit(0.2, std::nextafter(0.1, 0.2), 0.0, jitNextafterScalarAD);
}

JIT_TEST_FUNCTOR1(jitNextafterADScalar, nextafter(x, 0.2))
TEST(ExpressionsMathJit, nextafterADScalar)
{
    mathTest_jit(0.1, std::nextafter(0.1, 0.2), 1.0, jitNextafterADScalar);
}

// =============================================================================
// Scalbn expression variant
// =============================================================================

JIT_TEST_FUNCTOR1(jitScalbnExpr, scalbn(x * 2.3, 2))
TEST(ExpressionsMathJit, scalbnExpr)
{
    mathTest_jit(0.1, std::scalbn(0.1 * 2.3, 2), std::pow(double(FLT_RADIX), 2.0) * 2.3,
                 jitScalbnExpr);
}

// =============================================================================
// Max/Min with expressions
// =============================================================================

JIT_TEST_FUNCTOR2(jitMaxADExpr, max(x1, 2.3 * x2))
TEST(ExpressionsMathJit, maxADExpr)
{
    mathTest2_jit(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, jitMaxADExpr);
    mathTest2_jit(1.7, -0.7, 1.7, 1.0, 0.0, jitMaxADExpr);
}

JIT_TEST_FUNCTOR2(jitMaxExprAD, max(2.3 * x1, x2))
TEST(ExpressionsMathJit, maxExprAD)
{
    mathTest2_jit(0.3, 0.7, 0.7, 0.0, 1.0, jitMaxExprAD);
    mathTest2_jit(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, jitMaxExprAD);
}

JIT_TEST_FUNCTOR2(jitMaxExprExpr, max(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, maxExprExpr)
{
    mathTest2_jit(0.3, 0.7, 2.3 * 0.7, 0.0, 2.3, jitMaxExprExpr);
    mathTest2_jit(1.7, -0.7, 2.3 * 1.7, 2.3, 0.0, jitMaxExprExpr);
}

JIT_TEST_FUNCTOR1(jitMaxScalarAD, max(0.7, x))
TEST(ExpressionsMathJit, maxScalarAD)
{
    mathTest_jit(1.1, 1.1, 1.0, jitMaxScalarAD);
    mathTest_jit(0.6, 0.7, 0.0, jitMaxScalarAD);
}

JIT_TEST_FUNCTOR1(jitMaxADScalar, max(x, 0.7))
TEST(ExpressionsMathJit, maxADScalar)
{
    mathTest_jit(1.1, 1.1, 1.0, jitMaxADScalar);
    mathTest_jit(0.6, 0.7, 0.0, jitMaxADScalar);
}

JIT_TEST_FUNCTOR1(jitMaxScalarExpr, max(0.7, 2.0 * x))
TEST(ExpressionsMathJit, maxScalarExpr)
{
    mathTest_jit(1.1, 2.0 * 1.1, 2.0, jitMaxScalarExpr);
    mathTest_jit(0.3, 0.7, 0.0, jitMaxScalarExpr);
}

JIT_TEST_FUNCTOR1(jitMaxExprScalar, max(2.0 * x, 0.7))
TEST(ExpressionsMathJit, maxExprScalar)
{
    mathTest_jit(1.1, 2.0 * 1.1, 2.0, jitMaxExprScalar);
    mathTest_jit(0.3, 0.7, 0.0, jitMaxExprScalar);
}

JIT_TEST_FUNCTOR2(jitMinADExpr, min(x1, 2.3 * x2))
TEST(ExpressionsMathJit, minADExpr)
{
    mathTest2_jit(0.3, 0.7, 0.3, 1.0, 0.0, jitMinADExpr);
    mathTest2_jit(1.7, -0.7, -0.7 * 2.3, 0.0, 2.3, jitMinADExpr);
}

JIT_TEST_FUNCTOR2(jitMinExprAD, min(2.3 * x1, x2))
TEST(ExpressionsMathJit, minExprAD)
{
    mathTest2_jit(0.5, 0.7, 0.7, 0.0, 1.0, jitMinExprAD);
    mathTest2_jit(1.7, -0.7, -0.7, 0.0, 1.0, jitMinExprAD);
}

JIT_TEST_FUNCTOR2(jitMinExprExpr, min(2.3 * x1, 2.3 * x2))
TEST(ExpressionsMathJit, minExprExpr)
{
    mathTest2_jit(0.3, 0.7, 2.3 * 0.3, 2.3, 0.0, jitMinExprExpr);
    mathTest2_jit(1.7, -0.7, 2.3 * -0.7, 0.0, 2.3, jitMinExprExpr);
}

JIT_TEST_FUNCTOR1(jitMinScalarAD, min(0.7, x))
TEST(ExpressionsMathJit, minScalarAD)
{
    mathTest_jit(1.1, 0.7, 0.0, jitMinScalarAD);
    mathTest_jit(0.6, 0.6, 1.0, jitMinScalarAD);
}

JIT_TEST_FUNCTOR1(jitMinADScalar, min(x, 0.7))
TEST(ExpressionsMathJit, minADScalar)
{
    mathTest_jit(1.1, 0.7, 0.0, jitMinADScalar);
    mathTest_jit(0.6, 0.6, 1.0, jitMinADScalar);
}

JIT_TEST_FUNCTOR1(jitMinScalarExpr, min(0.7, 2.0 * x))
TEST(ExpressionsMathJit, minScalarExpr)
{
    mathTest_jit(1.1, 0.7, 0.0, jitMinScalarExpr);
    mathTest_jit(0.3, 2.0 * 0.3, 2.0, jitMinScalarExpr);
}

JIT_TEST_FUNCTOR1(jitMinExprScalar, min(2.0 * x, 0.7))
TEST(ExpressionsMathJit, minExprScalar)
{
    mathTest_jit(1.1, 0.7, 0.0, jitMinExprScalar);
    mathTest_jit(0.3, 2.0 * 0.3, 2.0, jitMinExprScalar);
}

// =============================================================================
// Smooth max/min functions
// =============================================================================

JIT_TEST_FUNCTOR2(jitSmoothMaxADAD, smooth_max(x1, x2))
TEST(ExpressionsMathJit, smoothMaxADAD)
{
    mathTest2_jit(0.3, 0.7, 0.7, 0.0, 1.0, jitSmoothMaxADAD);
    mathTest2_jit(1.7, -0.7, 1.7, 1.0, 0.0, jitSmoothMaxADAD);
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitSmoothMaxADAD);
}

JIT_TEST_FUNCTOR1(jitSmoothMaxScalarAD, smooth_max(0.7, x))
TEST(ExpressionsMathJit, smoothMaxScalarAD)
{
    mathTest_jit(1.1, 1.1, 1.0, jitSmoothMaxScalarAD);
    mathTest_jit(0.6, 0.7, 0.0, jitSmoothMaxScalarAD);
}

JIT_TEST_FUNCTOR1(jitSmoothMaxADScalar, smooth_max(x, 0.7))
TEST(ExpressionsMathJit, smoothMaxADScalar)
{
    mathTest_jit(1.1, 1.1, 1.0, jitSmoothMaxADScalar);
    mathTest_jit(0.6, 0.7, 0.0, jitSmoothMaxADScalar);
}

JIT_TEST_FUNCTOR2(jitSmoothMinADAD, smooth_min(x1, x2))
TEST(ExpressionsMathJit, smoothMinADAD)
{
    mathTest2_jit(0.3, 0.7, 0.3, 1.0, 0.0, jitSmoothMinADAD);
    mathTest2_jit(1.7, -0.7, -0.7, 0.0, 1.0, jitSmoothMinADAD);
    mathTest2_jit(1.7, 1.7, 1.7, 0.5, 0.5, jitSmoothMinADAD);
}

JIT_TEST_FUNCTOR1(jitSmoothMinScalarAD, smooth_min(0.7, x))
TEST(ExpressionsMathJit, smoothMinScalarAD)
{
    mathTest_jit(1.1, 0.7, 0.0, jitSmoothMinScalarAD);
    mathTest_jit(0.6, 0.6, 1.0, jitSmoothMinScalarAD);
}

JIT_TEST_FUNCTOR1(jitSmoothMinADScalar, smooth_min(x, 0.7))
TEST(ExpressionsMathJit, smoothMinADScalar)
{
    mathTest_jit(1.1, 0.7, 0.0, jitSmoothMinADScalar);
    mathTest_jit(0.6, 0.6, 1.0, jitSmoothMinADScalar);
}

// =============================================================================
// Copysign - TODO: needs ABool.If support for JIT
// The copysign function has conditional logic that is evaluated at recording
// time, not at JIT execution time. This needs to be fixed using ABool.If.
// =============================================================================

struct jitTestFunctor_jitCopysignScalarAD
{
    explicit jitTestFunctor_jitCopysignScalarAD(double op1) : op1_(op1) {}
    double op1_ = 0.0;
    template <class T>
    double operator()(const T& x) const
    {
        return copysign(op1_, x);
    }
};

TEST(ExpressionsMathJit, copysignScalarAD)
{
    mathTest_jit(1.2, 42.2, 0.0, jitTestFunctor_jitCopysignScalarAD(42.2));
    mathTest_jit(-1.2, -42.2, 0.0, jitTestFunctor_jitCopysignScalarAD(42.2));
}

struct jitTestFunctor_jitCopysignADScalar
{
    explicit jitTestFunctor_jitCopysignADScalar(double op2) : op2_(op2) {}
    double op2_ = 0.0;
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, op2_);
    }
};

TEST(ExpressionsMathJit, copysignADScalar)
{
    mathTest_jit(1.2, 1.2, 1.0, jitTestFunctor_jitCopysignADScalar(5.9));
    mathTest_jit(1.2, 1.2, 1.0, jitTestFunctor_jitCopysignADScalar(0.0));
    mathTest_jit(1.2, -1.2, -1.0, jitTestFunctor_jitCopysignADScalar(-5.9));
    mathTest_jit(1.2, -1.2, -1.0, jitTestFunctor_jitCopysignADScalar(-0.0000001));
}

struct jitTestFunctor_jitCopysignADAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, x);
    }
} jitCopysignADAD;

TEST(ExpressionsMathJit, copysignADAD)
{
    mathTest_jit(1.2, 1.2, 1.0, jitCopysignADAD);
    mathTest_jit(-1.2, -1.2, 1.0, jitCopysignADAD);
}

struct jitTestFunctor_jitCopysignADExpr
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(x, -x);
    }
} jitCopysignADExpr;

TEST(ExpressionsMathJit, copysignADExpr)
{
    mathTest_jit(1.2, -1.2, -1.0, jitCopysignADExpr);
}

struct jitTestFunctor_jitCopysignExprAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, x);
    }
} jitCopysignExprAD;

TEST(ExpressionsMathJit, copysignExprAD)
{
    mathTest_jit(1.2, 1.2, 1.0, jitCopysignExprAD);
}

struct jitTestFunctor_jitCopysignExprExpr
{
    template <class T>
    T operator()(const T& x) const
    {
        return copysign(-x, -x);
    }
} jitCopysignExprExpr;

TEST(ExpressionsMathJit, copysignExprExpr)
{
    mathTest_jit(1.2, -1.2, -1.0, jitCopysignExprExpr);
}

// =============================================================================
// frexp - pointer output function (writes exponent to pointer at recording time)
// Note: frexp writes to a pointer at recording time, so JIT won't re-execute that write.
// However, the value and derivative of the mantissa should still be correct.
// =============================================================================

struct jitTestFunctor_jitFrexpAD
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(x, &exponent);
    }
} jitFrexpAD;
int jitTestFunctor_jitFrexpAD::exponent = 0;

TEST(ExpressionsMathJit, frexpAD)
{
    mathTest_jit(1024.0, 0.5, 1.0 / (1 << 11), jitFrexpAD);
    EXPECT_EQ(jitTestFunctor_jitFrexpAD::exponent, 11);
}

struct jitTestFunctor_jitFrexpExpr
{
    static int exponent;
    template <class T>
    T operator()(const T& x) const
    {
        return frexp(1.0 * x, &exponent);
    }
} jitFrexpExpr;
int jitTestFunctor_jitFrexpExpr::exponent = 0;

TEST(ExpressionsMathJit, frexpExpr)
{
    mathTest_jit(1024.0, 0.5, 1.0 / (1 << 11), jitFrexpExpr);
    EXPECT_EQ(jitTestFunctor_jitFrexpExpr::exponent, 11);
}

// =============================================================================
// modf - pointer output function (writes integer part to pointer at recording time)
// Note: modf writes to a pointer at recording time, so JIT won't re-execute that write.
// However, the value and derivative of the fractional part should still be correct.
// =============================================================================

struct jitTestFunctor_jitModfADScalar
{
    static double ipart;
    template <class T>
    T operator()(const T& x) const
    {
        return modf(x, &ipart);
    }
} jitModfADScalar;
double jitTestFunctor_jitModfADScalar::ipart = 0.0;

TEST(ExpressionsMathJit, modfADScalar)
{
    mathTest_jit(1.2, 0.2, 1.0, jitModfADScalar);
    EXPECT_NEAR(jitTestFunctor_jitModfADScalar::ipart, 1.0, 1e-9);
}

struct jitTestFunctor_jitModfADAD
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
} jitModfADAD;
double jitTestFunctor_jitModfADAD::ipart = 0.0;

TEST(ExpressionsMathJit, modfADAD)
{
    mathTest_jit(1.2, 0.2, 1.0, jitModfADAD);
    EXPECT_NEAR(jitTestFunctor_jitModfADAD::ipart, 1.0, 1e-9);
}

// =============================================================================
// remquo - pointer output function (writes quotient to pointer at recording time)
// Note: remquo writes to a pointer at recording time, so JIT won't re-execute that write.
// However, the value and derivative of the remainder should still be correct.
// =============================================================================

int jitRmqn_ = 0;

struct jitTestFunctor_jitRemquoAD
{
    template <class T>
    T operator()(const T& x1, const T& x2) const
    {
        return xad::remquo(x1, x2, &jitRmqn_);
    }
} jitRemquoAD;

TEST(ExpressionsMathJit, remquoAD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest2_jit(1.3, 0.5, res, 1.0, -double(n), jitRemquoAD);
    EXPECT_EQ(n, jitRmqn_);
    jitRmqn_ = 0;
}

struct jitTestFunctor_jitRemquoADScalar
{
    template <class T>
    T operator()(const T& x) const
    {
        return xad::remquo(x, 0.5, &jitRmqn_);
    }
} jitRemquoADScalar;

TEST(ExpressionsMathJit, remquoADScalar)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_jit(1.3, res, 1.0, jitRemquoADScalar);
    EXPECT_EQ(n, jitRmqn_);
    jitRmqn_ = 0;
}

struct jitTestFunctor_jitRemquoScalarAD
{
    template <class T>
    T operator()(const T& x) const
    {
        return xad::remquo(1.3, x, &jitRmqn_);
    }
} jitRemquoScalarAD;

TEST(ExpressionsMathJit, remquoScalarAD)
{
    int n;
    auto res = std::remquo(1.3, 0.5, &n);
    mathTest_jit(0.5, res, -double(n), jitRemquoScalarAD);
    EXPECT_EQ(n, jitRmqn_);
    jitRmqn_ = 0;
}

// =============================================================================
// Scalar operations (scalar + AD, scalar * AD, etc.)
// =============================================================================

JIT_TEST_FUNCTOR1(jitScalarAddAD, 2.3 + x)
TEST(ExpressionsMathJit, scalarAddAD)
{
    mathTest_jit(1.0, 3.3, 1.0, jitScalarAddAD);
}

JIT_TEST_FUNCTOR1(jitScalarSubAD, 2.3 - x)
TEST(ExpressionsMathJit, scalarSubAD)
{
    mathTest_jit(1.0, 1.3, -1.0, jitScalarSubAD);
}

JIT_TEST_FUNCTOR1(jitScalarMulAD, 2.3 * x)
TEST(ExpressionsMathJit, scalarMulAD)
{
    mathTest_jit(1.0, 2.3, 2.3, jitScalarMulAD);
}

JIT_TEST_FUNCTOR1(jitScalarDivAD, 2.3 / x)
TEST(ExpressionsMathJit, scalarDivAD)
{
    mathTest_jit(1.0, 2.3, -2.3, jitScalarDivAD);
}

JIT_TEST_FUNCTOR1(jitADAddScalar, x + 2.3)
TEST(ExpressionsMathJit, adAddScalar)
{
    mathTest_jit(1.0, 3.3, 1.0, jitADAddScalar);
}

JIT_TEST_FUNCTOR1(jitADSubScalar, x - 2.3)
TEST(ExpressionsMathJit, adSubScalar)
{
    mathTest_jit(1.0, -1.3, 1.0, jitADSubScalar);
}

JIT_TEST_FUNCTOR1(jitADMulScalar, x * 2.3)
TEST(ExpressionsMathJit, adMulScalar)
{
    mathTest_jit(1.0, 2.3, 2.3, jitADMulScalar);
}

JIT_TEST_FUNCTOR1(jitADDivScalar, x / 2.3)
TEST(ExpressionsMathJit, adDivScalar)
{
    mathTest_jit(1.0, 1.0 / 2.3, 1.0 / 2.3, jitADDivScalar);
}

// =============================================================================
// Negation
// =============================================================================

JIT_TEST_FUNCTOR1(jitNegAD, -x)
TEST(ExpressionsMathJit, negAD)
{
    mathTest_jit(1.3, -1.3, -1.0, jitNegAD);
}

// =============================================================================
// Fma (fused multiply-add)
// =============================================================================

struct jitTestFunctor_jitFmaADADAD
{
    template <class T>
    T operator()(const T& x1, const T& x2) const
    {
        // fma(a, b, c) = a * b + c, but we only have 2 inputs
        // So test fma(x1, x2, x1) = x1 * x2 + x1
        return fma(x1, x2, x1);
    }
} jitFmaADADAD;

TEST(ExpressionsMathJit, fmaADADAD)
{
    // fma(x1, x2, x1) = x1 * x2 + x1
    // d/dx1 = x2 + 1
    // d/dx2 = x1
    mathTest2_jit(1.3, 0.7, 1.3 * 0.7 + 1.3, 0.7 + 1.0, 1.3, jitFmaADADAD);
}

#endif  // XAD_ENABLE_JIT

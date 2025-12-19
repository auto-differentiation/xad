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
// Copysign - only AD vs AD works (no expression variants due to control flow)
// =============================================================================

JIT_TEST_FUNCTOR2(jitCopysignADAD, copysign(x1, x2))
TEST(ExpressionsMathJit, copysignADAD)
{
    // Positive sign source
    mathTest2_jit(1.2, 0.5, 1.2, 1.0, 0.0, jitCopysignADAD);
    mathTest2_jit(-1.2, 0.5, 1.2, -1.0, 0.0, jitCopysignADAD);
    // Negative sign source
    mathTest2_jit(1.2, -0.5, -1.2, -1.0, 0.0, jitCopysignADAD);
    mathTest2_jit(-1.2, -0.5, -1.2, 1.0, 0.0, jitCopysignADAD);
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

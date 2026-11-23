/*******************************************************************************

   Unit tests for math function derivatives (Part 1 - split due to long compile
   times).

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
#include <cmath>
#include <limits>

LOCAL_TEST_FUNCTOR1(degreesAD, degrees(x))
TEST(ExpressionsMath, degreesAD)
{
    mathTest_all_aad(3.141592653589793238462643, 180.0,
                     57.2957795130823208767981548141051703324054725, 0.0, degreesAD);
}

LOCAL_TEST_FUNCTOR1(degreesExpr, degrees(0.5 * x))
TEST(ExpressionsMath, degreesExpr)
{
    mathTest_all_aad(3.141592653589793238462643, 90.0,
                     0.5 * 57.2957795130823208767981548141051703324054725, 0.0, degreesExpr);
}

LOCAL_TEST_FUNCTOR1(radiansAD, radians(x));
TEST(ExpressionsMath, radiansAD)
{
    mathTest_all_aad(180.0, 3.141592653589793238462643,
                     0.0174532925199432957692369076848861271344287, 0.0, radiansAD);
}

LOCAL_TEST_FUNCTOR1(radiansExpr, radians(2.0 * x))
TEST(ExpressionsMath, radiansExpr)
{
    mathTest_all_aad(180.0, 2.0 * 3.141592653589793238462643,
                     2.0 * 0.0174532925199432957692369076848861271344287, 0.0, radiansExpr);
}

LOCAL_TEST_FUNCTOR1(cosAD, cos(x))
TEST(ExpressionsMath, cosAD)
{
    mathTest_all(1.0, std::cos(1.0), -std::sin(1.0), -std::cos(1.0), cosAD);
}

LOCAL_TEST_FUNCTOR1(cosExpr, cos(2.3 * x))
TEST(ExpressionsMath, cosExpr)
{
    mathTest_all(1.0, std::cos(2.3), -2.3 * std::sin(2.3), -2.3 * 2.3 * cos(2.3), cosExpr);
}

LOCAL_TEST_FUNCTOR1(sinAD, sin(x))
TEST(ExpressionsMath, sinAD)
{
    mathTest_all(1.0, std::sin(1.0), std::cos(1.0), -std::sin(1.0), sinAD);
}

LOCAL_TEST_FUNCTOR1(sinExpr, sin(2.3 * x))
TEST(ExpressionsMath, sinExpr)
{
    mathTest_all(1.0, std::sin(2.3), 2.3 * std::cos(2.3), -2.3 * 2.3 * std::sin(2.3), sinExpr);
}

LOCAL_TEST_FUNCTOR1(expAD, exp(x))
TEST(ExpressionsMath, expAD)
{
    mathTest_all(1.0, std::exp(1.0), std::exp(1.0), std::exp(1.0), expAD);
}

LOCAL_TEST_FUNCTOR1(expExpr, exp(2.3 * x))
TEST(ExpressionsMath, expExpr)
{
    mathTest_all(1.0, std::exp(2.3), 2.3 * std::exp(2.3), 2.3 * 2.3 * std::exp(2.3), expExpr);
}

LOCAL_TEST_FUNCTOR1(logAD, log(x))
TEST(ExpressionsMath, logAD)
{
    mathTest_all(1.3, std::log(1.3), 1.0 / 1.3, -1.0 / (1.3 * 1.3), logAD);
}

LOCAL_TEST_FUNCTOR1(logExpr, log(2.3 * x))
TEST(ExpressionsMath, logExpr) { mathTest_all(1.0, std::log(2.3), 1.0, -1.0, logExpr); }

LOCAL_TEST_FUNCTOR1(log10AD, log10(x))
TEST(ExpressionsMath, log10AD)
{
    mathTest_all(1.3, std::log10(1.3), 1.0 / std::log(10.0) / 1.3,
                 -1.0 / std::log(10.0) / (1.3 * 1.3), log10AD);
}

LOCAL_TEST_FUNCTOR1(log10Expr, log10(2.3 * x))
TEST(ExpressionsMath, log10Expr)
{
    mathTest_all(1.0, std::log10(2.3), 2.3 / std::log(10.0) / 2.3,
                 -2.3 * 2.3 / std::log(10.0) / (2.3 * 2.3), log10Expr);
}

LOCAL_TEST_FUNCTOR1(log2AD, log2(x))
TEST(ExpressionsMath, log2AD)
{
    using xad::log2;
    mathTest_all(1.3, log2(1.3), 1.0 / log(2.0) / 1.3, -1.0 / std::log(2.0) / (1.3 * 1.3), log2AD);
}

LOCAL_TEST_FUNCTOR1(log2Expr, log2(2.3 * x))
TEST(ExpressionsMath, log2Expr)
{
    using xad::log2;
    mathTest_all(1.3, log2(2.3 * 1.3), 2.3 / log(2.0) / 1.3 / 2.3,
                 -2.3 * 2.3 / std::log(2.0) / (1.3 * 2.3 * 1.3 * 2.3), log2Expr);
}

LOCAL_TEST_FUNCTOR1(asinAD, asin(x))
TEST(ExpressionsMath, asinAD)
{
    mathTest_all(0.3, std::asin(0.3), 1.0 / std::sqrt(1.0 - 0.3 * 0.3),
                 0.3 / std::pow(1 - 0.3 * 0.3, 3.0 / 2.0), asinAD);
}

LOCAL_TEST_FUNCTOR1(asinExpr, asin(2.3 * x))
TEST(ExpressionsMath, asinExpr)
{
    mathTest_all(0.1, std::asin(0.1 * 2.3), 2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 2.3 * 2.3 * 2.3 * 0.1 / std::pow(1 - 2.3 * 2.3 * 0.1 * 0.1, 3.0 / 2.0), asinExpr);
}

LOCAL_TEST_FUNCTOR1(acosAD, acos(x))
TEST(ExpressionsMath, acosAD)
{
    mathTest_all(0.3, std::acos(0.3), -1.0 / std::sqrt(1.0 - 0.3 * 0.3),
                 -0.3 / std::pow(1.0 - 0.3 * 0.3, 3.0 / 2.0), acosAD);
}

LOCAL_TEST_FUNCTOR1(acosExpr, acos(2.3 * x))
TEST(ExpressionsMath, acosExpr)
{
    mathTest_all(0.1, std::acos(0.1 * 2.3), -2.3 / std::sqrt(1.0 - 2.3 * 2.3 * 0.1 * 0.1),
                 -2.3 * 2.3 * 2.3 * 0.1 / std::pow(1 - 2.3 * 2.3 * 0.1 * 0.1, 3.0 / 2.0), acosExpr);
}

LOCAL_TEST_FUNCTOR1(atanAD, atan(x))
TEST(ExpressionsMath, atanAD)
{
    mathTest_all(0.3, std::atan(0.3), 1.0 / (1.0 + 0.3 * 0.3),
                 -2.0 * 0.3 / (0.3 * 0.3 + 1) / (0.3 * 0.3 + 1), atanAD);
}

LOCAL_TEST_FUNCTOR1(atanExpr, atan(2.3 * x))
TEST(ExpressionsMath, atanExpr)
{
    mathTest_all(0.1, std::atan(0.1 * 2.3), 2.3 / (1.0 + 2.3 * 2.3 * 0.1 * 0.1),
                 -2.0 * 2.3 * 2.3 * 2.3 * 0.1 / (2.3 * 2.3 * 0.1 * 0.1 + 1.0) /
                     (2.3 * 2.3 * 0.1 * 0.1 + 1.0),
                 atanExpr);
}

LOCAL_TEST_FUNCTOR1(sinhAD, sinh(x))
TEST(ExpressionsMath, sinhAD)
{
    mathTest_all(0.3, std::sinh(0.3), std::cosh(0.3), std::sinh(0.3), sinhAD);
}

LOCAL_TEST_FUNCTOR1(sinhExpr, sinh(2.3 * x))
TEST(ExpressionsMath, sinhExpr)
{
    mathTest_all(0.1, std::sinh(0.1 * 2.3), 2.3 * std::cosh(2.3 * 0.1),
                 2.3 * 2.3 * std::sinh(2.3 * 0.1), sinhExpr);
}

LOCAL_TEST_FUNCTOR1(coshAD, cosh(x))
TEST(ExpressionsMath, coshAD)
{
    mathTest_all(0.3, std::cosh(0.3), std::sinh(0.3), std::cosh(0.3), coshAD);
}

LOCAL_TEST_FUNCTOR1(coshExpr, cosh(2.3 * x))
TEST(ExpressionsMath, coshExpr)
{
    mathTest_all(0.3, std::cosh(2.3 * 0.3), 2.3 * std::sinh(2.3 * 0.3),
                 2.3 * 2.3 * std::cosh(2.3 * 0.3), coshExpr);
}

LOCAL_TEST_FUNCTOR1(expm1AD, expm1(x))
TEST(ExpressionsMath, expm1AD)
{
    using xad::expm1;
    mathTest_all(0.3, expm1(0.3), std::exp(0.3), std::exp(0.3), expm1AD);
}

LOCAL_TEST_FUNCTOR1(expm1Expr, expm1(2.3 * x))
TEST(ExpressionsMath, expm1Expr)
{
    using xad::expm1;
    mathTest_all(0.3, expm1(2.3 * 0.3), 2.3 * std::exp(2.3 * 0.3), 2.3 * 2.3 * std::exp(2.3 * 0.3),
                 expm1Expr);
}

LOCAL_TEST_FUNCTOR1(exp2AD, exp2(x))
TEST(ExpressionsMath, exp2AD)
{
    using xad::exp2;
    mathTest_all(0.3, exp2(0.3), std::log(2.0) * exp2(0.3),
                 std::log(2.0) * std::log(2.0) * exp2(0.3), exp2AD);
}

LOCAL_TEST_FUNCTOR1(exp2Expr, exp2(2.3 * x))
TEST(ExpressionsMath, exp2Expr)
{
    using xad::exp2;
    mathTest_all(0.3, exp2(2.3 * 0.3), 2.3 * std::log(2.0) * exp2(2.3 * 0.3),
                 2.3 * 2.3 * std::log(2.0) * std::log(2.0) * exp2(2.3 * 0.3), exp2Expr);
}

LOCAL_TEST_FUNCTOR1(log1pAD, log1p(x))
TEST(ExpressionsMath, log1pAD)
{
    using xad::log1p;
    mathTest_all(0.3, log1p(0.3), 1.0 / (1.0 + 0.3), -1.0 / (0.3 + 1.0) / (0.3 + 1.0), log1pAD);
}

LOCAL_TEST_FUNCTOR1(log1pExpr, log1p(2.3 * x))
TEST(ExpressionsMath, log1pExpr)
{
    using xad::log1p;
    mathTest_all(0.3, log1p(2.3 * 0.3), 2.3 / (1.0 + 2.3 * 0.3),
                 -2.3 * 2.3 / (2.3 * 0.3 + 1.0) / (2.3 * 0.3 + 1.0), log1pExpr);
}

LOCAL_TEST_FUNCTOR1(asinhAD, asinh(x))
TEST(ExpressionsMath, asinhAD)
{
    using xad::asinh;
    mathTest_all(0.3, asinh(0.3), 1.0 / std::sqrt(1.0 + 0.3 * 0.3),
                 -0.3 / std::pow(0.3 * .3 + 1.0, 3.0 / 2.0), asinhAD);
}

LOCAL_TEST_FUNCTOR1(asinhExpr, asinh(2.3 * x))
TEST(ExpressionsMath, asinhExpr)
{
    using xad::asinh;
    mathTest_all(0.3, asinh(2.3 * 0.3), 2.3 / std::sqrt(1.0 + 2.3 * 2.3 * 0.3 * 0.3),
                 -2.3 * 2.3 * 2.3 * 0.3 / std::pow(2.3 * 2.3 * 0.3 * 0.3 + 1.0, 3. / 2.),
                 asinhExpr);
}

LOCAL_TEST_FUNCTOR1(acoshAD, acosh(x))
TEST(ExpressionsMath, acoshAD)
{
    using xad::acosh;
    mathTest_all(1.3, acosh(1.3), 1.0 / std::sqrt(1.3 * 1.3 - 1.0),
                 -1.3 / std::pow(1.3 * 1.3 - 1., 3. / 2.), acoshAD);
}

LOCAL_TEST_FUNCTOR1(acoshExpr, acosh(2.3 * x))
TEST(ExpressionsMath, acoshExpr)
{
    using xad::acosh;
    mathTest_all(1.3, acosh(2.3 * 1.3), 2.3 / std::sqrt(2.3 * 2.3 * 1.3 * 1.3 - 1.0),
                 -2.3 * 2.3 * 2.3 * 1.3 / std::pow(2.3 * 2.3 * 1.3 * 1.3 - 1., 3. / 2.), acoshExpr);
}

LOCAL_TEST_FUNCTOR1(atanhAD, atanh(x))
TEST(ExpressionsMath, atanhAD)
{
    using xad::atanh;
    mathTest_all(0.3, atanh(0.3), 1.0 / (1.0 - 0.3 * 0.3),
                 2. * 0.3 / (0.3 * 0.3 - 1.) / (0.3 * 0.3 - 1.), atanhAD);
}

LOCAL_TEST_FUNCTOR1(atanhExpr, atanh(2.3 * x))
TEST(ExpressionsMath, atanhExpr)
{
    using xad::atanh;
    mathTest_all(
        0.3, atanh(2.3 * 0.3), 2.3 / (1.0 - 2.3 * 2.3 * 0.3 * 0.3),
        2. * 2.3 * 2.3 * 2.3 * 0.3 / (2.3 * 2.3 * 0.3 * 0.3 - 1.) / (2.3 * 2.3 * 0.3 * 0.3 - 1.),
        atanhExpr);
}

LOCAL_TEST_FUNCTOR1(erfAD, erf(x))
TEST(ExpressionsMath, erfAD)
{
    mathTest_all(0.3, std::erf(0.3), 2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3),
                 -4. * 0.3 * exp(-0.3 * 0.3) / std::sqrt(M_PI), erfAD);
}

LOCAL_TEST_FUNCTOR1(erfExpr, erf(2.3 * x))
TEST(ExpressionsMath, erfExpr)
{
    mathTest_all(
        0.3, std::erf(2.3 * 0.3), 2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3),
        -4. * 2.3 * 2.3 * 2.3 * 0.3 * std::exp(-2.3 * 2.3 * 0.3 * 0.3) / std::sqrt(M_PI), erfExpr);
}

LOCAL_TEST_FUNCTOR1(erfcAD, erfc(x))
TEST(ExpressionsMath, erfcAD)
{
    mathTest_all(0.3, std::erfc(0.3), -2.0 / std::sqrt(M_PI) * std::exp(-0.3 * 0.3),
                 4. * 0.3 * std::exp(-0.3 * 0.3) / std::sqrt(M_PI), erfcAD);
}

LOCAL_TEST_FUNCTOR1(erfcExpr, erfc(2.3 * x))
TEST(ExpressionsMath, erfcExpr)
{
    mathTest_all(
        0.3, std::erfc(2.3 * 0.3), -2.3 * 2.0 / std::sqrt(M_PI) * std::exp(-2.3 * 2.3 * 0.3 * 0.3),
        4. * 2.3 * 2.3 * 2.3 * 0.3 * std::exp(-2.3 * 2.3 * 0.3 * 0.3) / std::sqrt(M_PI), erfcExpr);
}

LOCAL_TEST_FUNCTOR1(tgammaAD, tgamma(x))
TEST(ExpressionsMath, tgammaAD)
{
    // derivatives tgamma(x) * digamma(x) and tgamma(x) * (digamma(x)^2 + trigamma(x))
    mathTest_all(2.7, std::tgamma(2.7), 1.2307796833525972, 1.6714666893832903, tgammaAD);
    mathTest_all(0.3, std::tgamma(0.3), -10.478042841758517, 73.332451673574937, tgammaAD);
    mathTest_all(-1.3, std::tgamma(-1.3), 9.5940952077655055, 76.039137001296766, tgammaAD);
}

LOCAL_TEST_FUNCTOR1(tgammaExpr, tgamma(0.5 * x))
TEST(ExpressionsMath, tgammaExpr)
{
    mathTest_all(2.7, std::tgamma(0.5 * 2.7), -0.050763556394748485, 0.242877383360948, tgammaExpr);
}

LOCAL_TEST_FUNCTOR1(lgammaAD, lgamma(x))
TEST(ExpressionsMath, lgammaAD)
{
    // derivatives digamma(x) and trigamma(x), also for negative x (log of |tgamma(x)|)
    mathTest_all(3.5, std::lgamma(3.5), 1.1031566406452431, 0.33035775610023488, lgammaAD);
    mathTest_all(0.3, std::lgamma(0.3), -3.5025242222001332, 12.24536454610773, lgammaAD);
    mathTest_all(-2.5, std::lgamma(-2.5), 1.1031566406452431, 9.5392466449891238, lgammaAD);
}

LOCAL_TEST_FUNCTOR1(lgammaExpr, lgamma(2.3 * x))
TEST(ExpressionsMath, lgammaExpr)
{
    mathTest_all(0.3, std::lgamma(2.3 * 0.3), -2.8719873893028152, 15.339506080762495, lgammaExpr);
}

LOCAL_TEST_FUNCTOR1(digammaAD, xad::digamma(x))
TEST(ExpressionsMath, digammaAD)
{
    // derivatives trigamma(x) and tetragamma(x)
    mathTest_all(0.3, xad::digamma(0.3), 12.24536454610773, -75.272536588726027, digammaAD);
    mathTest_all(-0.7, xad::digamma(-0.7), 14.286180872638342, -69.441632798638565, digammaAD);
}

TEST(ExpressionsMath, digammaKnownValues)
{
    // reference values from mpmath (40 digits), rounded to double
    // -3 + 1e-7 and -1 - 1e-6 test the reflection close to the poles
    const double x[] = {1.0,  0.5,  2.0,     10.5, 25.0,      1e6,      0.3,
                        -0.7, -2.5, -100.25, 1e-8, -3 + 1e-7, -1 - 1e-6};
    const double psi[] = {-0.57721566490153287, -1.9635100260214235, 0.42278433509846713,
                          2.3030010342976865,   3.198742512851974,   13.815510057964191,
                          -3.5025242222001332,  -2.0739527936287043, 1.1031566406452431,
                          7.7542389592086458,   -100000000.57721564, -9999998.7602478197,
                          1000000.4228639568};
    const double psi1[] = {1.6449340668482264,   4.934802200544679,    0.64493406684822641,
                           0.099916956059126733, 0.040810663257225578, 1.0000005000001667e-06,
                           12.24536454610773,    14.286180872638342,   9.5392466449891238,
                           19.729283325349364,   10000000000000002.0,  100000000327318.78,
                           1000000000167.1782};
    for (std::size_t i = 0; i < sizeof(x) / sizeof(x[0]); ++i)
    {
        EXPECT_NEAR(psi[i], xad::digamma(x[i]), 1e-14 * std::abs(psi[i])) << "digamma " << x[i];
        EXPECT_NEAR(psi1[i], xad::detail::trigamma(x[i]), 1e-14 * std::abs(psi1[i]))
            << "trigamma " << x[i];
    }
    EXPECT_NEAR(-0.57721566490153287, xad::digamma(1.0f), 1e-6);
}

namespace
{
// value, first and second derivative of f at x0 with nested forward mode
template <class T, class F>
void gammaFwdFwd(F f, T x0, T& y, T& d1, T& d2)
{
    xad::FReal<xad::FReal<T> > x = x0;
    derivative(value(x)) = T(1);
    value(derivative(x)) = T(1);
    xad::FReal<xad::FReal<T> > r = f(x);
    y = value(value(r));
    d1 = derivative(value(r));
    d2 = derivative(derivative(r));
}

// first derivative of f at x0 in adjoint mode
template <class F>
double gammaAdj(F f, double x0)
{
    typedef xad::adj<double> mode;
    mode::tape_type tape;
    mode::active_type x = x0;
    tape.registerInput(x);
    tape.newRecording();
    mode::active_type y = f(x);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();
    return derivative(x);
}

// EXPECT_NEAR converts to double, so compare long double values directly
::testing::AssertionResult expectNearLd(long double ref, long double act, long double tol)
{
    const long double diff = std::abs(ref - act);
    if (diff <= tol)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "difference " << diff << " exceeds " << tol << " (value " << act << ')';
}

struct digammaFn
{
    template <class T>
    T operator()(const T& x) const
    {
        return xad::digamma(x);
    }
};

struct lgammaFn
{
    template <class T>
    T operator()(const T& x) const
    {
        return lgamma(x);
    }
};

struct tgammaFn
{
    template <class T>
    T operator()(const T& x) const
    {
        return tgamma(x);
    }
};

struct gammaSumFn
{
    template <class T>
    T operator()(const T& x) const
    {
        return tgamma(x) + lgamma(x);
    }
};
}  // namespace

TEST(ExpressionsMath, digammaDerivativesNearPoles)
{
    // reference values from mpmath (40 digits): psi, psi1, psi2 and log|tgamma| at the double
    // values of -3 + 1e-7 and -1 - 1e-6, where the reflection's cotangent term dominates
    const double x[] = {-3 + 1e-7, -1 - 1e-6};
    const double psi[] = {-9999998.7602478197, 1000000.4228639568};
    const double psi1[] = {100000000327318.78, 1000000000167.1782};
    const double psi2[] = {-2.0000000098194733e+21, 2.0000000004935998e+18};
    const double lg[] = {14.326336308978625, 13.815510135263528};
    for (std::size_t i = 0; i < 2; ++i)
    {
        double y, d1, d2;
        gammaFwdFwd(digammaFn(), x[i], y, d1, d2);
        EXPECT_NEAR(psi[i], y, 1e-14 * std::abs(psi[i])) << "digamma " << x[i];
        EXPECT_NEAR(psi1[i], d1, 1e-14 * std::abs(psi1[i])) << "digamma d1 " << x[i];
        EXPECT_NEAR(psi2[i], d2, 1e-14 * std::abs(psi2[i])) << "digamma d2 " << x[i];
        EXPECT_NEAR(psi1[i], gammaAdj(digammaFn(), x[i]), 1e-14 * std::abs(psi1[i]))
            << "digamma adj " << x[i];

        gammaFwdFwd(lgammaFn(), x[i], y, d1, d2);
        // at -3 + 1e-7 the value is the platform's std::lgamma, which is less
        // accurate near the poles on some libms (e.g. macOS)
        EXPECT_NEAR(lg[i], y, (i == 0 ? 1e-9 : 1e-14) * std::abs(lg[i]))
            << "lgamma " << x[i];
        EXPECT_NEAR(psi[i], d1, 1e-14 * std::abs(psi[i])) << "lgamma d1 " << x[i];
        EXPECT_NEAR(psi1[i], d2, 1e-14 * std::abs(psi1[i])) << "lgamma d2 " << x[i];
        EXPECT_NEAR(psi[i], gammaAdj(lgammaFn(), x[i]), 1e-14 * std::abs(psi[i]))
            << "lgamma adj " << x[i];
    }
}

TEST(ExpressionsMath, digammaNaNPropagatesToHigherOrders)
{
    // at the poles and for NaN input the derivatives of nested active types are NaN, too
    for (double x0 : {0.0, -1.0, -2.0, std::numeric_limits<double>::quiet_NaN()})
    {
        double y, d1, d2;
        gammaFwdFwd(digammaFn(), x0, y, d1, d2);
        EXPECT_TRUE(std::isnan(y)) << x0;
        EXPECT_TRUE(std::isnan(d1)) << x0;
        EXPECT_TRUE(std::isnan(d2)) << x0;

        xad::FReal<double> x = x0;
        derivative(x) = 1.0;
        xad::FReal<double> t = xad::detail::trigamma(x);
        EXPECT_TRUE(std::isnan(value(t))) << x0;
        EXPECT_TRUE(std::isnan(derivative(t))) << x0;
    }
}

TEST(ExpressionsMath, digammaOverloads)
{
    // integral arguments are computed as double
    EXPECT_DOUBLE_EQ(xad::digamma(2.0), xad::digamma(2));
    EXPECT_DOUBLE_EQ(xad::digamma(10.0), xad::digamma(10L));
    EXPECT_TRUE(std::isnan(xad::digamma(-1)));

    // pi is not truncated to double for long double
    EXPECT_EQ(3.141592653589793238462643383279502884197L, xad::detail::piOf<long double>());
    EXPECT_TRUE(expectNearLd(-2.07395279362870441754L, xad::digamma(-0.7L), 1e-15L));
    EXPECT_TRUE(expectNearLd(14.2861808726383427104L, xad::detail::trigamma(-0.7L), 1e-14L));
}

TEST(ExpressionsMath, gammaLongDouble)
{
    // reference values from mpmath (40 digits) at 2.7
    const long double tg = 1.54468584585059376496L, tg1 = 1.23077968335259725262L,
                      tg2 = 1.67146668938329031272L;
    const long double lg = 0.434820553655104531705L, lg1 = 0.796783168991141015503L,
                      lg2 = 0.447212068918323641463L;
    long double y, d1, d2;
    gammaFwdFwd(tgammaFn(), 2.7L, y, d1, d2);
    EXPECT_TRUE(expectNearLd(tg, y, 1e-15L));
    EXPECT_TRUE(expectNearLd(tg1, d1, 1e-15L));
    EXPECT_TRUE(expectNearLd(tg2, d2, 1e-15L));
    gammaFwdFwd(lgammaFn(), 2.7L, y, d1, d2);
    EXPECT_TRUE(expectNearLd(lg, y, 1e-15L));
    EXPECT_TRUE(expectNearLd(lg1, d1, 1e-15L));
    EXPECT_TRUE(expectNearLd(lg2, d2, 1e-15L));

    xad::FReal<long double> x = 2.7L;
    derivative(x) = 1.0L;
    xad::FReal<long double> r = gammaSumFn()(x);
    EXPECT_TRUE(expectNearLd(tg + lg, value(r), 1e-15L));
    EXPECT_TRUE(expectNearLd(tg1 + lg1, derivative(r), 1e-15L));

    // AReal<long double> uses the same derivatives; the library has no long double tape, so
    // check them through the operators it instantiates
    EXPECT_TRUE(expectNearLd(tg1, xad::tgamma_op<long double>().derivative(2.7L, tg), 1e-15L));
    EXPECT_TRUE(expectNearLd(lg1, xad::lgamma_op<long double>().derivative(2.7L), 1e-15L));
    EXPECT_TRUE(expectNearLd(lg2, xad::digamma_op<long double>().derivative(2.7L), 1e-15L));
}

TEST(ExpressionsMath, digammaPoles)
{
    for (double x : {0.0, -1.0, -2.0, -1e10})
    {
        EXPECT_TRUE(std::isnan(xad::digamma(x))) << x;
        EXPECT_TRUE(std::isnan(xad::detail::trigamma(x))) << x;
    }
    EXPECT_TRUE(std::isnan(xad::digamma(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_EQ(std::numeric_limits<double>::infinity(),
              xad::digamma(std::numeric_limits<double>::infinity()));
}

TEST(ExpressionsMath, gammaDerivativesMatchFiniteDifferences)
{
    typedef xad::adj<double> mode;
    for (double x0 : {0.05, 0.7, 1.4616321449683623, 3.0, 7.25, 42.0, -0.4, -3.6})
    {
        mode::tape_type tape;
        mode::active_type x = x0;
        tape.registerInput(x);
        tape.newRecording();
        mode::active_type y = tgamma(x) + lgamma(x) + xad::digamma(x);
        tape.registerOutput(y);
        derivative(y) = 1.0;
        tape.computeAdjoints();

        auto f = [](double v) { return std::tgamma(v) + std::lgamma(v) + xad::digamma(v); };
        const double h = 1e-6 * (std::max)(1.0, std::abs(x0));
        const double fd = (f(x0 + h) - f(x0 - h)) / (2 * h);
        EXPECT_NEAR(fd, derivative(x), 1e-6 * (std::max)(1.0, std::abs(fd))) << x0;
    }
}

LOCAL_TEST_FUNCTOR1(tanhAD, tanh(x))
TEST(ExpressionsMath, tanhAD)
{
    mathTest_all(0.3, std::tanh(0.3), 1.0 - std::tanh(0.3) * std::tanh(0.3),
                 -2. / std::cosh(0.3) / std::cosh(0.3) * std::tanh(.3), tanhAD);
}

LOCAL_TEST_FUNCTOR1(tanhExpr, tanh(2.3 * x))
TEST(ExpressionsMath, tanhExpr)
{
    mathTest_all(
        0.3, std::tanh(2.3 * 0.3), 2.3 * (1.0 - std::tanh(2.3 * 0.3) * std::tanh(2.3 * 0.3)),
        -2. * 2.3 * 2.3 / std::cosh(2.3 * 0.3) / std::cosh(2.3 * 0.3) * std::tanh(2.3 * 0.3),
        tanhExpr);
}

LOCAL_TEST_FUNCTOR1(tanAD, tan(x))
TEST(ExpressionsMath, tanAD)
{
    mathTest_all(0.3, std::tan(0.3), 1.0 / std::cos(0.3) / std::cos(0.3),
                 2. / std::cos(0.3) / std::cos(0.3) * std::tan(0.3), tanAD);
}

LOCAL_TEST_FUNCTOR1(tanExpr, tan(2.3 * x))
TEST(ExpressionsMath, tanExpr)
{
    mathTest_all(0.3, std::tan(2.3 * 0.3), 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3),
                 2. * 2.3 * 2.3 / std::cos(2.3 * 0.3) / std::cos(2.3 * 0.3) * std::tan(2.3 * 0.3),
                 tanExpr);
}

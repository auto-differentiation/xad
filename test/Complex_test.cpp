/*******************************************************************************

   Tests for std::complex with AD types.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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

#define _USE_MATH_DEFINES
#include <XAD/Complex.hpp>
#include <XAD/XAD.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <complex>
#include <type_traits>
#include <typeinfo>

// disable potential divide by zero warning, as we're testing this with the compliance tests
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

// NOTE: A lot of these tests could be refactored to compare to the std::complex
// results instead, including the compliance tests
// Then this could be parameterised tests for each function, where we'd just
// give input values (incl. special values) and the test compares to
// std::complex results.

using namespace ::testing;

typedef xad::AReal<double> dblAD;
typedef dblAD::tape_type tape_type;
typedef xad::FReal<double> dblFAD;
typedef ::testing::Types<double, dblAD, dblFAD> test_types;
#ifdef _MSC_VER
typedef ::testing::Types<dblAD, dblFAD> compliance_test_types;
#else
typedef test_types compliance_test_types;
#endif

template <class T>
class ComplexTest : public testing::Test
{
};

TYPED_TEST_SUITE(ComplexTest, test_types);

template <class T>
class ComplexComplianceTest : public ComplexTest<T>
{
};

TYPED_TEST_SUITE(ComplexComplianceTest, compliance_test_types);

MATCHER(IsPositiveZero, "")
{
    (void)result_listener;  // silence warning
    return xad::fpclassify(arg) == FP_ZERO && !xad::signbit(arg);
}
MATCHER(IsNegativeZero, "")
{
    (void)result_listener;  // silence warning
    return xad::fpclassify(arg) == FP_ZERO && xad::signbit(arg);
}
MATCHER(IsPositiveInf, "")
{
    (void)result_listener;  // silence warning
    return xad::isinf(arg) && arg > 0.0;
}

MATCHER(IsNegativeInf, "")
{
    (void)result_listener;  // silence warning
    return xad::isinf(arg) && arg < 0.0;
}

#if (defined(__GNUC__) && __GNUC__ < 5) && !defined(__clang__)  // with gcc 4, we're using gtest 1.10, which doesn't have IsNan
MATCHER(IsNan, "")
{
    (void)result_listener;  // silence warning
    return xad::fpclassify(arg) == FP_NAN;
}
#endif

// ------------------ matchers themselves -----------
TYPED_TEST(ComplexComplianceTest, Matchers)
{
    auto posInf = TypeParam(std::numeric_limits<double>::infinity());
    auto negInf = TypeParam(-std::numeric_limits<double>::infinity());
    auto num = TypeParam(1.2);
    auto posZero = TypeParam(0.0);
    auto negZero = TypeParam(-0.0);
    auto nan = TypeParam(std::numeric_limits<double>::quiet_NaN());

    // some sanity checks
    EXPECT_THAT(xad::value(posInf), Gt(0.0));
    EXPECT_THAT(xad::value(posInf), Not(Lt(0.0)));
    EXPECT_THAT(xad::value(negInf), Lt(0.0));
    EXPECT_THAT(xad::value(negInf), Not(Gt(0.0)));

    // cannot be both positive and negative inf
    EXPECT_THAT(xad::value(num), Not(AllOf(IsPositiveInf(), IsNegativeInf())));
    EXPECT_THAT(xad::value(posInf), Not(AllOf(IsPositiveInf(), IsNegativeInf())));
    EXPECT_THAT(xad::value(negInf), Not(AllOf(IsPositiveInf(), IsNegativeInf())));
    EXPECT_THAT(xad::value(negZero), Not(AllOf(IsPositiveInf(), IsNegativeInf())));
    EXPECT_THAT(xad::value(posZero), Not(AllOf(IsPositiveInf(), IsNegativeInf())));
    EXPECT_THAT(xad::value(nan), Not(AllOf(IsPositiveInf(), IsNegativeInf())));

    // cannot be both positive and negative zero
    EXPECT_THAT(xad::value(num), Not(AllOf(IsPositiveZero(), IsNegativeZero())));
    EXPECT_THAT(xad::value(posInf), Not(AllOf(IsPositiveZero(), IsNegativeZero())));
    EXPECT_THAT(xad::value(negInf), Not(AllOf(IsPositiveZero(), IsNegativeZero())));
    EXPECT_THAT(xad::value(negZero), Not(AllOf(IsPositiveZero(), IsNegativeZero())));
    EXPECT_THAT(xad::value(posZero), Not(AllOf(IsPositiveZero(), IsNegativeZero())));
    EXPECT_THAT(xad::value(nan), Not(AllOf(IsPositiveZero(), IsNegativeZero())));

    // cannot be both NaN and any of the others
    EXPECT_THAT(xad::value(num), Not(AllOf(IsNan(), IsPositiveInf())));
    EXPECT_THAT(xad::value(posInf), Not(AllOf(IsNan(), IsNegativeInf())));
    EXPECT_THAT(xad::value(negInf), Not(AllOf(IsNan(), IsPositiveZero())));
    EXPECT_THAT(xad::value(negZero), Not(AllOf(IsNan(), IsNegativeZero())));
    EXPECT_THAT(xad::value(posZero), Not(AllOf(IsNan(), IsNegativeZero())));

    // matchers checks
    EXPECT_THAT(xad::value(posInf), IsPositiveInf());
    EXPECT_THAT(xad::value(posInf), Not(IsNegativeInf()));
    EXPECT_THAT(xad::value(posInf), Not(IsNan()));
    EXPECT_THAT(xad::value(posInf), Not(IsPositiveZero()));
    EXPECT_THAT(xad::value(posInf), Not(IsNegativeZero()));

    EXPECT_THAT(xad::value(negInf), IsNegativeInf());
    EXPECT_THAT(xad::value(negInf), Not(IsPositiveInf()));
    EXPECT_THAT(xad::value(negInf), Not(IsNan()));
    EXPECT_THAT(xad::value(negInf), Not(IsPositiveZero()));
    EXPECT_THAT(xad::value(negInf), Not(IsNegativeZero()));

    EXPECT_THAT(xad::value(num), Not(IsNegativeInf()));
    EXPECT_THAT(xad::value(num), Not(IsPositiveInf()));
    EXPECT_THAT(xad::value(num), Not(IsNan()));
    EXPECT_THAT(xad::value(num), Not(IsPositiveZero()));
    EXPECT_THAT(xad::value(num), Not(IsNegativeZero()));

    EXPECT_THAT(xad::value(posZero), Not(IsNegativeInf()));
    EXPECT_THAT(xad::value(posZero), Not(IsPositiveInf()));
    EXPECT_THAT(xad::value(posZero), Not(IsNan()));
    EXPECT_THAT(xad::value(posZero), IsPositiveZero());
    EXPECT_THAT(xad::value(posZero), Not(IsNegativeZero()));

    EXPECT_THAT(xad::value(negZero), Not(IsNegativeInf()));
    EXPECT_THAT(xad::value(negZero), Not(IsPositiveInf()));
    EXPECT_THAT(xad::value(negZero), Not(IsNan()));
    EXPECT_THAT(xad::value(negZero), Not(IsPositiveZero()));
    EXPECT_THAT(xad::value(negZero), IsNegativeZero());
}

// ------------------ constructors -----------------

TYPED_TEST(ComplexTest, DefaultConstructorGivesZeroParts)
{
    std::complex<TypeParam> z;

    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromReal)
{
    TypeParam r = 42.0;
    std::complex<TypeParam> z(r);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(42.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromScalarDouble)
{
    std::complex<TypeParam> z(42.0);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(42.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromInt)
{
    std::complex<TypeParam> z(42);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(42.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ImplicitConvertFromInt)
{
    auto func_test = [](std::complex<TypeParam> in) { return in; };
    std::complex<TypeParam> z = func_test(42);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(42.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromScalarExpression)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(42.0 * x);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(84.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromLongRealExpression)
{
    using std::exp;
    // this is an epxression found in Heston model, were initialising from failed at some point
    TypeParam kappa = 0.2, dt = 0.1;
    const std::complex<TypeParam> beta = 4.0 * kappa * exp(-0.5 * kappa * dt);
    const std::complex<double> expected =
        4.0 * xad::value(kappa) * std::exp(-0.5 * xad::value(kappa) * xad::value(dt));
    EXPECT_THAT(xad::value(beta.real()), DoubleEq(expected.real()));
    EXPECT_THAT(xad::value(beta.imag()), DoubleEq(expected.imag()));
}

TYPED_TEST(ComplexTest, ConstructFromBraceExpressionDouble)
{
    std::complex<TypeParam> z;
    using std::cos;
    z = {cos(2.4), sin(2.4)};
    EXPECT_THAT(xad::value(z.real()), DoubleEq(std::cos(2.4)));
    EXPECT_THAT(xad::value(z.imag()), DoubleEq(std::sin(2.4)));
}

TYPED_TEST(ComplexTest, ConstructFromBraceExpressionExpr)
{
    std::complex<TypeParam> z;
    TypeParam s = 1.2;
    using std::cos;
    z = {cos(2.4 * s), sin(2.4 * s)};
    EXPECT_THAT(xad::value(z.real()), DoubleEq(std::cos(2.4 * 1.2)));
    EXPECT_THAT(xad::value(z.imag()), DoubleEq(std::sin(2.4 * 1.2)));
}

TYPED_TEST(ComplexTest, ConstructFromComplexArgs)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromComplex)
{
    auto t = std::complex<TypeParam>(1.2, -1.2);
    std::complex<TypeParam> z(t);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromExpressionFirst)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(x * x);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromExpressionFirst_WithImag)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(x * x, 1.0);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(1.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromExpressionSecond)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(1.0, x * x);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(4.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConstructFromExpressionBoth)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(3.0 * x, x * x);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(6.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(4.0, 1e-9));
}

// ------------------ set real/imag -----------------

TYPED_TEST(ComplexTest, SetRealImagFromScalars)
{
    TypeParam xr = 2.0;
    TypeParam xi = 3.0;
    std::complex<TypeParam> z;
    z.real(xr);
    z.imag(xi);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(3.0, 1e-9));
}

TYPED_TEST(ComplexTest, SetRealImagFromDouble)
{
    double xr = 2.0;
    double xi = 3.0;
    std::complex<TypeParam> z;
    z.real(xr);
    z.imag(xi);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(3.0, 1e-9));
}

TYPED_TEST(ComplexTest, SetRealImagFromInteger)
{
    int xr = 2;
    int xi = 3;
    std::complex<TypeParam> z;
    z.real(xr);
    z.imag(xi);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(3.0, 1e-9));
}

TYPED_TEST(ComplexTest, SetRealImagFromExpression)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z;
    z.real(x * x);
    z.imag(x * x * 2.0);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(8.0, 1e-9));
}

// --------------- copy constructor ----------

TYPED_TEST(ComplexTest, CopyConstruct)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    std::complex<TypeParam> z_cpy(z);
    EXPECT_THAT(xad::value(z_cpy.real()), DoubleNear(xad::value(z.real()), 1e-9));
    EXPECT_THAT(xad::value(z_cpy.imag()), DoubleNear(xad::value(z.imag()), 1e-9));
}

TYPED_TEST(ComplexTest, CopyConstructFromDifferentType)
{
    auto z = std::complex<float>(1.2f, -1.2f);
    std::complex<TypeParam> z_cpy(z);
    EXPECT_THAT(xad::value(z_cpy.real()), DoubleNear(xad::value(z.real()), 1e-9));
    EXPECT_THAT(xad::value(z_cpy.imag()), DoubleNear(xad::value(z.imag()), 1e-9));
}

// ------------- copy-assign --------------

TYPED_TEST(ComplexTest, CopyAssignment)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    std::complex<TypeParam> z_cpy;
    z_cpy = z;
    EXPECT_THAT(xad::value(z_cpy.real()), DoubleNear(xad::value(z.real()), 1e-9));
    EXPECT_THAT(xad::value(z_cpy.imag()), DoubleNear(xad::value(z.imag()), 1e-9));
}

TYPED_TEST(ComplexTest, CopyAssignFromDifferentType)
{
    auto z = std::complex<float>(1.2f, -1.2f);
    std::complex<TypeParam> z_cpy;
    z_cpy = z;
    EXPECT_THAT(xad::value(z_cpy.real()), DoubleNear(xad::value(z.real()), 1e-9));
    EXPECT_THAT(xad::value(z_cpy.imag()), DoubleNear(xad::value(z.imag()), 1e-9));
}

// ------------ assignment -------------

TYPED_TEST(ComplexTest, AssignFromScalar)
{
    std::complex<TypeParam> z(12.1, 123.0);
    TypeParam x = 1.2;
    z = x;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, AssignFromDouble)
{
    std::complex<TypeParam> z(12.1, 123.0);
    double x = 1.2;
    z = x;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, AssignFromInt)
{
    std::complex<TypeParam> z(12.1, 123.0);
    int x = 2;
    z = x;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.0, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, AssignFromScalarExpression)
{
    std::complex<TypeParam> z(12.1, 123.0);
    TypeParam x = 1.2;
    z = x * 2.0;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.0, 1e-9));
}

// ------------ operator += --------------

TYPED_TEST(ComplexTest, PlusEqualsFromSameType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<TypeParam>(1.0, 1.0);
    z += z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.2, 1e-9));
}

TYPED_TEST(ComplexTest, PlusEqualsFromDifferentType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<float>(1.0f, 1.0f);
    z += z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.2, 1e-9));
}

TYPED_TEST(ComplexTest, PlusEqualsWithScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z += z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(3.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, PlusEqualsWithDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double z1 = 1.0;
    z += z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, PlusEqualsScalarExpressions)
{
    TypeParam x = 2.0;
    std::complex<TypeParam> z(1.2, -1.2);
    z += (x * x);
    EXPECT_THAT(xad::value(z.real()), DoubleNear(5.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, PlusEqualsReturnType)
{
    double xd = 2.0;
    TypeParam xt = 2.0;
    std::complex<TypeParam> z(1.2, 1.2);

    static_assert(std::is_same<decltype(z += 1.0), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z += xd), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z += 1), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z += xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z += xt * xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
}

// ------------ operator -= --------------

TYPED_TEST(ComplexTest, MinusEqualsFromSameType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<TypeParam>(1.0, 1.0);
    z -= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-2.2, 1e-9));
}

TYPED_TEST(ComplexTest, MinusEqualsFromDifferentType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<float>(1.0f, 1.0f);
    z -= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-2.2, 1e-9));
}

TYPED_TEST(ComplexTest, MinusEqualsFromDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double z1 = 1.0;
    z -= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.2, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, MinusEqualsFromScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z -= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(-0.8, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, MinusEqualsFromScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z -= z1 * 1.0;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(-0.8, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, MinusEqualsReturnType)
{
    double xd = 2.0;
    TypeParam xt = 2.0;
    std::complex<TypeParam> z(1.2, 1.2);

    static_assert(std::is_same<decltype(z -= 1.0), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z -= xd), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z -= 1), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z -= xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z -= xt * xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
}

// ------------ operator /= --------------

TYPED_TEST(ComplexTest, DivEqualsFromSameType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<TypeParam>(2.0, 2.5);
    z /= z1;
    // got that from Python
    EXPECT_THAT(xad::value(z.real()), DoubleNear(-0.05853658536585366, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.5268292682926831, 1e-9));
}

TYPED_TEST(ComplexTest, DivEqualsFromDifferentType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<float>(2.0f, 2.5f);
    z /= z1;
    // got that from Python
    EXPECT_THAT(xad::value(z.real()), DoubleNear(-0.05853658536585366, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.5268292682926831, 1e-9));
}

TYPED_TEST(ComplexTest, DivEqualsFromDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double z1 = 2.0;
    z /= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.6, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.6, 1e-9));
}

TYPED_TEST(ComplexTest, DivEqualsWithScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z /= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.6, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.6, 1e-9));
}

TYPED_TEST(ComplexTest, DivEqualsWithScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z /= z1 * 1.0;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(0.6, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.6, 1e-9));
}

TYPED_TEST(ComplexTest, DivEqualsReturnType)
{
    double xd = 2.0;
    TypeParam xt = 2.0;
    std::complex<TypeParam> z(1.2, 1.2);

    static_assert(std::is_same<decltype(z /= 1.0), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z /= xd), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z /= 1), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z /= xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z /= xt * xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
}

// ------------ operator *= --------------

TYPED_TEST(ComplexTest, MulEqualsFromSameType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<TypeParam>(2.0, 2.5);
    z *= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(5.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.6, 1e-9));
}

TYPED_TEST(ComplexTest, MulEqualsFromDifferentType)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = std::complex<float>(2.0, 2.5);
    z *= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(5.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.6, 1e-9));
}

TYPED_TEST(ComplexTest, MulEqualsFromDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double z1 = 2.0;
    z *= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-2.4, 1e-9));
}

TYPED_TEST(ComplexTest, MulEqualsWithScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z *= z1;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-2.4, 1e-9));
}

TYPED_TEST(ComplexTest, MulEqualsWithScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto z1 = TypeParam(2.0);
    z *= z1 * 1.0;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-2.4, 1e-9));
}

TYPED_TEST(ComplexTest, MulEqualsReturnType)
{
    double xd = 2.0;
    TypeParam xt = 2.0;
    std::complex<TypeParam> z(1.2, 1.2);

    static_assert(std::is_same<decltype(z *= 1.0), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z *= xd), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z *= 1), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z *= xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
    static_assert(std::is_same<decltype(z *= xt * xt), std::complex<TypeParam>&>::value,
                  "return types not as expected");
}

// ------------- real --------------

TYPED_TEST(ComplexTest, NonMemberReal)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto ret = std::real(z);
    EXPECT_THAT(xad::value(ret), DoubleNear(1.2, 1e-9));
}

TYPED_TEST(ComplexTest, NonMemberRealScalarExpressions)
{
    TypeParam x = 2.0;
    auto ret = x * x;
    EXPECT_THAT(xad::value(ret), DoubleNear(4.0, 1e-9));
}

// ------------- imag --------------

TYPED_TEST(ComplexTest, NonMemberImag)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    auto ret = std::imag(z);
    EXPECT_THAT(xad::value(ret), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, NonMemberImagScalarExpressions)
{
    TypeParam x = 2.0;
    auto z = std::imag(x * x);
    EXPECT_THAT(xad::value(z), DoubleNear(0.0, 1e-9));
}

// ----------------- unary plus / minus ----------

TYPED_TEST(ComplexTest, UnaryPlusDoesNothing)
{
    auto in = std::complex<TypeParam>(1.2, -1.2);
    auto out = +in;
    EXPECT_THAT(xad::value(out.real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(out.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, UnaryMinusNegates)
{
    auto in = std::complex<TypeParam>(1.2, -1.2);
    auto out = -in;
    EXPECT_THAT(xad::value(out.real()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(out.imag()), DoubleNear(1.2, 1e-9));
}

// -------------- operator== ----------------------

TYPED_TEST(ComplexTest, EqualityCompareComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    EXPECT_TRUE(z1 == z1);
    EXPECT_FALSE(z1 == z2);
}

TYPED_TEST(ComplexTest, EqualityCompareWithScalar)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    TypeParam s1 = 1.2;
    TypeParam s2 = 15.5;
    EXPECT_FALSE(s1 == z1);
    EXPECT_TRUE(s1 == z2);
    EXPECT_FALSE(s2 == z2);
    EXPECT_FALSE(z1 == s1);
    EXPECT_TRUE(z2 == s1);
    EXPECT_FALSE(z2 == s2);
}

TYPED_TEST(ComplexTest, EqualityCompareWithScalarExpression)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    TypeParam s1 = 1.2;
    TypeParam s2 = 15.5;
    EXPECT_FALSE(s1 * 1.0 == z1);
    EXPECT_TRUE(s1 * 1.0 == z2);
    EXPECT_FALSE(s2 * 1.0 == z2);
    EXPECT_FALSE(z1 == s1 * 1.0);
    EXPECT_TRUE(z2 == s1 * 1.0);
    EXPECT_FALSE(z2 == s2 * 1.0);
}

TYPED_TEST(ComplexTest, EqualityCompareWithDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    double s1 = 1.2;
    double s2 = 15.5;
    EXPECT_FALSE(s1 == z1);
    EXPECT_TRUE(s1 == z2);
    EXPECT_FALSE(s2 == z2);
    EXPECT_FALSE(z1 == s1);
    EXPECT_TRUE(z2 == s1);
    EXPECT_FALSE(z2 == s2);
}

// ----------------- operator != --------------

TYPED_TEST(ComplexTest, NonEqualityCompareComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    EXPECT_FALSE(z1 != z1);
    EXPECT_TRUE(z1 != z2);
}

TYPED_TEST(ComplexTest, NonEqualityCompareWithScalar)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    TypeParam s1 = 1.2;
    TypeParam s2 = 15.5;
    EXPECT_TRUE(s1 != z1);
    EXPECT_FALSE(s1 != z2);
    EXPECT_TRUE(s2 != z2);
    EXPECT_TRUE(z1 != s1);
    EXPECT_FALSE(z2 != s1);
    EXPECT_TRUE(z2 != s2);
}

TYPED_TEST(ComplexTest, NonEqualityCompareWithScalarExpression)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    TypeParam s1 = 1.2;
    TypeParam s2 = 15.5;
    EXPECT_TRUE(s1 * 1.0 != z1);
    EXPECT_FALSE(s1 * 1.0 != z2);
    EXPECT_TRUE(s2 * 1.0 != z2);
    EXPECT_TRUE(z1 != s1 * 1.0);
    EXPECT_FALSE(z2 != s1 * 1.0);
    EXPECT_TRUE(z2 != s2 * 1.0);
}

TYPED_TEST(ComplexTest, NonEqualityCompareWithDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 0.0);
    double s1 = 1.2;
    double s2 = 15.5;
    EXPECT_TRUE(s1 != z1);
    EXPECT_FALSE(s1 != z2);
    EXPECT_TRUE(s2 != z2);
    EXPECT_TRUE(z1 != s1);
    EXPECT_FALSE(z2 != s1);
    EXPECT_TRUE(z2 != s2);
}

// -------------- operator+ ---------------

TYPED_TEST(ComplexTest, AddComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    auto ret = z1 + z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(40.8, 1e-9));
}

TYPED_TEST(ComplexTest, AddScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret1 = z + s;
    auto ret2 = s + z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, AddScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    std::complex<TypeParam> ret = z + (s * 1.0);
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, AddDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    std::complex<TypeParam> ret = z + 1.2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-1.2, 1e-9));
}

// -------------- operator- ---------------

TYPED_TEST(ComplexTest, SubstractComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    auto ret = z1 - z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-43.2, 1e-9));
}

TYPED_TEST(ComplexTest, SubstractScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret1 = z - s;
    auto ret2 = s - z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(1.2, 1e-9));
}

TYPED_TEST(ComplexTest, SubstractScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret2 = (s * 1.0) - z;
    auto ret = z - (s * 1.0);
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(1.2, 1e-9));
}

TYPED_TEST(ComplexTest, SubstractDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double s = 1.2;
    auto ret1 = z - s;
    auto ret2 = s - z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(1.2, 1e-9));
}

// -------------- operator* ---------------

TYPED_TEST(ComplexTest, MultiplyComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    auto ret = z1 * z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(51.84, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(48.96, 1e-9));
}

TYPED_TEST(ComplexTest, MultiplyScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret1 = z * s;
    auto ret2 = s * z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.44, 1e-9));
}

TYPED_TEST(ComplexTest, MultiplyScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret = z * (s * 1.0);
    auto ret2 = (s * 1.0) * z;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.44, 1e-9));
}

TYPED_TEST(ComplexTest, MultiplyDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double s = 1.2;
    auto ret1 = z * s;
    auto ret2 = s * z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.44, 1e-9));
}

// -------------- operator/ ---------------

TYPED_TEST(ComplexTest, DivideComplex)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(1.2, 42.0);
    auto ret = z1 / z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(-0.027732463295269166, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-0.029363784665579117, 1e-9));
}

TYPED_TEST(ComplexTest, DivideScalar)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret1 = z / s;
    auto ret2 = s / z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.5, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(0.5, 1e-9));
}

TYPED_TEST(ComplexTest, DivideScalarExpression)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    TypeParam s = 1.2;
    auto ret = z / (s * 1.0);
    auto ret2 = (s * 1.0) / z;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-1.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.5, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(0.5, 1e-9));
}

TYPED_TEST(ComplexTest, DivideDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    double s = 1.2;
    auto ret1 = z / s;
    auto ret2 = s / z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.0, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(0.5, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(0.5, 1e-9));
}

///////////// Math functions

// ------------------ abs ----------------------

TYPED_TEST(ComplexTest, Abs)
{
    auto z = std::complex<TypeParam>(3.0, -4.0);
    TypeParam a = std::abs(z);
    EXPECT_THAT(xad::value(a), DoubleNear(5.0, 1e-9));
}

TYPED_TEST(ComplexTest, AbsOfExpr)
{
    auto z = std::complex<TypeParam>(3.0, -4.0);
    TypeParam a = std::abs(z - z + z);
    EXPECT_THAT(xad::value(a), DoubleNear(5.0, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, Abs)
{
    // behaves like std::hypot(real, imag) according to specs
    auto z1 = std::complex<TypeParam>(3.0, -4.0);
    auto z2 = std::complex<TypeParam>(-4.0, 3.0);
    auto z3 = std::complex<TypeParam>(-3.0, 0.0);
    auto z4 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 0.0);
    auto z5 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), -12.0);
    auto z6 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    auto z7 = std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity());
    auto z8 = std::complex<TypeParam>(12.12, -std::numeric_limits<double>::infinity());
    auto z9 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      -std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(std::abs(z1)), DoubleNear(xad::value(std::abs(z2)), 1e-9));
    EXPECT_THAT(xad::value(std::abs(z3)), DoubleNear(std::fabs(xad::value(z3.real())), 1e-9));
    EXPECT_THAT(xad::value(std::abs(z4)), IsPositiveInf());
    // positive infinity
    EXPECT_TRUE(xad::isinf(xad::value(std::abs(z5))));
    EXPECT_TRUE(xad::isinf(xad::value(std::abs(z6))));
    EXPECT_TRUE(xad::isinf(xad::value(std::abs(z7))));
    EXPECT_TRUE(xad::isinf(xad::value(std::abs(z8))));
    EXPECT_TRUE(xad::isinf(xad::value(std::abs(z9))));
    EXPECT_THAT(xad::value(std::abs(z5)), Gt(0.0));
    EXPECT_THAT(xad::value(std::abs(z6)), Gt(0.0));
    EXPECT_THAT(xad::value(std::abs(z7)), Gt(0.0));
    EXPECT_THAT(xad::value(std::abs(z8)), Gt(0.0));
    EXPECT_THAT(xad::value(std::abs(z9)), Gt(0.0));
}

// --------------- arg -----------------------

TYPED_TEST(ComplexTest, ArgOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(arg(z)), DoubleNear(M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(z1)), DoubleNear(-M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(z2)), DoubleNear(3 * M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(z3)), DoubleNear(-3 * M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(z4)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ArgOfDoubleOrInteger)
{
    auto z = 1.2;
    auto z1 = 1;
    auto z2 = -1.2;
    auto z3 = -1;
    auto z4 = 0.0;

    using std::arg;

    EXPECT_THAT(arg(z), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(arg(z1), DoubleNear(0.0, 1e-9));
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    EXPECT_THAT(arg(z2), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(arg(z3), DoubleNear(0.0, 1e-9));
#else
    EXPECT_THAT(arg(z2), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(arg(z3), DoubleNear(M_PI, 1e-9));
#endif
    EXPECT_THAT(arg(z4), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ArgOfFloat)
{
    auto z = 1.2f;
    auto z1 = -1.2f;
    auto z2 = 0.0f;

    using std::arg;

    EXPECT_THAT(double(arg(z)), DoubleNear(0.0, 1e-6));

#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    EXPECT_THAT(double(arg(z1)), DoubleNear(0.0, 1e-6));
#else
    EXPECT_THAT(double(arg(z1)), DoubleNear(M_PI, 1e-6));
#endif
    EXPECT_THAT(double(arg(z2)), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexTest, ArgOfScalar)
{
    TypeParam z = 1.2;
    TypeParam z1 = -1.2;
    TypeParam z2 = 0.0;

    using std::arg;

    EXPECT_THAT(xad::value(arg(z)), DoubleNear(0.0, 1e-6));
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    EXPECT_THAT(xad::value(arg(z1)), DoubleNear(0.0, 1e-6));
#else
    EXPECT_THAT(xad::value(arg(z1)), DoubleNear(M_PI, 1e-6));
#endif
    EXPECT_THAT(xad::value(arg(z2)), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexTest, ArgOfScalarExpression)
{
    TypeParam z = 1.2;
    TypeParam z1 = -1.2;
    TypeParam z2 = 0.0;

    using std::arg;

    EXPECT_THAT(xad::value(arg(z * 1.0)), DoubleNear(0.0, 1e-6));
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    EXPECT_THAT(xad::value(arg(z1 + 0.0)), DoubleNear(0.0, 1e-6));
#else
    EXPECT_THAT(xad::value(arg(z1 + 0.0)), DoubleNear(M_PI, 1e-6));
#endif
    EXPECT_THAT(xad::value(arg(z2 * 1.0)), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexComplianceTest, ArgOfScalarOrExpr)
{
    using std::arg;
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    EXPECT_THAT(xad::value(arg(TypeParam(-1.0))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-0.0))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-1.0) * 1.0)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-0.0) * 1.0)), DoubleNear(0.0, 1e-9));
#else
    EXPECT_THAT(xad::value(arg(TypeParam(-1.0))), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-0.0))), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-1.0) * 1.0)), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(TypeParam(-0.0) * 1.0)), DoubleNear(M_PI, 1e-9));
#endif

    EXPECT_THAT(xad::value(arg(TypeParam(1.0))), IsPositiveZero());
    EXPECT_THAT(xad::value(arg(TypeParam(0.0))), IsPositiveZero());
    EXPECT_THAT(xad::value(arg(TypeParam(1.0) * 1.0)), IsPositiveZero());
    EXPECT_THAT(xad::value(arg(TypeParam(0.0) * 1.0)), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, ArgOfZeroImag)
{
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-1.0, 0.0))), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-0.0, 0.0))), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-1.0, -0.0))), DoubleNear(-M_PI, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-0.0, -0.0))), DoubleNear(-M_PI, 1e-9));

    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(1.0, 0.0))), IsPositiveZero());
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(0.0, 0.0))), IsPositiveZero());
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(1.0, -0.0))), IsNegativeZero());
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(0.0, -0.0))), IsNegativeZero());
}

TYPED_TEST(ComplexComplianceTest, ArgOfInfinityImag)
{
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity()))),
        DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(1.2, -std::numeric_limits<double>::infinity()))),
        DoubleNear(-M_PI / 2, 1e-9));

    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity()))),
                DoubleNear(3 * M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       -std::numeric_limits<double>::infinity()))),
                DoubleNear(-3 * M_PI / 4, 1e-9));

    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity()))),
                DoubleNear(M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       -std::numeric_limits<double>::infinity()))),
                DoubleNear(-M_PI / 4, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, ArgOfPosNegZeroReal)
{
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(0.0, -1.2))), DoubleNear(-M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-0.0, -1.2))), DoubleNear(-M_PI / 2, 1e-9));

    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(0.0, 1.2))), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(-0.0, 1.2))), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, ArgOfInfReal)
{
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2))),
        DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), -1.2))),
        DoubleNear(-M_PI, 1e-9));

    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2))),
        IsPositiveZero());
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::infinity(), -1.2))),
        IsNegativeZero());
}

TYPED_TEST(ComplexComplianceTest, ArgOfNaN)
{
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2))),
        IsNan());
    EXPECT_THAT(
        xad::value(arg(std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN()))),
        IsNan());
    EXPECT_THAT(xad::value(arg(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                                       std::numeric_limits<double>::quiet_NaN()))),
                IsNan());
}

// --------------- norm -----------------------

TYPED_TEST(ComplexTest, NormOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    using std::norm;

    EXPECT_THAT(xad::value(std::norm(z)), DoubleNear(2.88, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z1)), DoubleNear(2.88, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z2)), DoubleNear(2.88, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z3)), DoubleNear(2.88, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z4)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, NormWithExplicitTemplateParam)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);

    EXPECT_THAT(xad::value(std::norm<TypeParam>(z)), DoubleNear(2.88, 1e-9));
}

TYPED_TEST(ComplexTest, NormOfDoubleOrInteger)
{
    auto z = 1.2;
    auto z1 = 1;
    auto z2 = -1.2;
    auto z3 = -1;
    auto z4 = 0.0;

    EXPECT_THAT(std::norm(z), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(std::norm(z1), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(std::norm(z2), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(std::norm(z3), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(std::norm(z4), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, NormOfFloat)
{
    auto z = 1.2f;
    auto z1 = -1.2f;
    auto z2 = 0.0f;

    EXPECT_THAT(double(std::norm(z)), DoubleNear(1.44, 1e-6));
    EXPECT_THAT(double(std::norm(z1)), DoubleNear(1.44, 1e-6));
    EXPECT_THAT(double(std::norm(z2)), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexTest, NormOfScalar)
{
    TypeParam z = 1.2;
    TypeParam z1 = 1;
    TypeParam z2 = -1.2;
    TypeParam z3 = -1;
    TypeParam z4 = 0.0;

    EXPECT_THAT(xad::value(std::norm(z)), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z1)), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z2)), DoubleNear(1.44, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z3)), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(std::norm(z4)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, NormOfScalarExpressions)
{
    TypeParam z = 1.2;
    TypeParam z1 = -1.2;
    TypeParam z2 = 0.0;

    EXPECT_THAT(xad::value(std::norm(z + 0.0)), DoubleNear(1.44, 1e-6));
    EXPECT_THAT(xad::value(std::norm(z1 * 1.0)), DoubleNear(1.44, 1e-6));
    EXPECT_THAT(xad::value(std::norm(z2 + 0.0)), DoubleNear(0.0, 1e-6));
}

// --------------- conj -----------------------

TYPED_TEST(ComplexTest, ConjOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(conj(z).real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z).imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z1).real()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z1).imag()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z2).real()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z2).imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z3).real()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z3).imag()), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(conj(z4).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(conj(z4).imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConjOfDoubleOrInteger)
{
    auto z = 1.2;
    auto z1 = -1.2;
    auto z2 = 0.0;

    EXPECT_THAT(std::real(std::conj(z)), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(std::imag(std::conj(z)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(std::real(std::conj(z1)), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(std::imag(std::conj(z1)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(std::real(std::conj(z2)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(std::imag(std::conj(z2)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConjOfFloat)
{
    auto z = 1.2f;
    auto z1 = -1.2f;
    auto z2 = 0.0f;

    EXPECT_THAT(double(std::real(std::conj(z))), DoubleNear(1.2, 1e-6));
    EXPECT_THAT(double(std::imag(std::conj(z))), DoubleNear(0.0, 1e-6));
    EXPECT_THAT(double(std::real(std::conj(z1))), DoubleNear(-1.2, 1e-6));
    EXPECT_THAT(double(std::imag(std::conj(z1))), DoubleNear(0.0, 1e-6));
    EXPECT_THAT(double(std::real(std::conj(z2))), DoubleNear(0.0, 1e-6));
    EXPECT_THAT(double(std::imag(std::conj(z2))), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexTest, ConjOfScalar)
{
    TypeParam z = 1.2;
    TypeParam z1 = -1.2;
    TypeParam z2 = 0.0;

    EXPECT_THAT(xad::value(std::real(std::conj(z))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::conj(z))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::real(std::conj(z1))), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::conj(z1))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::real(std::conj(z2))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::conj(z2))), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ConjugateOfScalarExpressions)
{
    TypeParam x = 2.0;
    auto r = x * 2.0;
    auto c = std::conj(r);
    EXPECT_THAT(xad::value(std::real(c)), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(std::imag(c)), DoubleNear(0.0, 1e-9));
}

// --------------- proj -----------------------

TYPED_TEST(ComplexTest, ProjOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), -1.2);
    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    auto z3 = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    auto z4 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity());
    auto z5 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), -1.2);
    auto z6 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2);
    auto z7 = std::complex<TypeParam>(1.2, -std::numeric_limits<double>::infinity());
    auto z8 = std::complex<TypeParam>(-1.2, -std::numeric_limits<double>::infinity());

    EXPECT_THAT(xad::value(std::real(proj(z))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::imag(proj(z))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::real(proj(z1))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z1))), IsNegativeZero());
    EXPECT_THAT(xad::value(std::real(proj(z2))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z2))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(proj(z3))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z3))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(proj(z4))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z4))), IsPositiveZero());

    EXPECT_THAT(xad::value(std::real(proj(z5))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z5))), IsNegativeZero());
    EXPECT_THAT(xad::value(std::real(proj(z6))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z6))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(proj(z7))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z7))), IsNegativeZero());
    EXPECT_THAT(xad::value(std::real(proj(z8))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(proj(z8))), IsNegativeZero());
}

TYPED_TEST(ComplexTest, ProjOfDoubleOrInteger)
{
    auto z = 1.2;
    auto z1 = std::numeric_limits<double>::infinity();
    auto z1n = -std::numeric_limits<double>::infinity();
    auto z2 = 0.0;

    EXPECT_THAT(std::real(std::proj(z)), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(std::imag(std::proj(z)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(std::real(std::proj(z1)), IsPositiveInf());
    EXPECT_THAT(std::imag(std::proj(z1)), IsPositiveZero());
#if ((defined(_MSC_VER) && _MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) && !defined(__clang__)
    // VS 2017 evaluates this differently
    EXPECT_THAT(std::real(std::proj(z1n)), IsNegativeInf());
#else
    EXPECT_THAT(std::real(std::proj(z1n)), IsPositiveInf());
#endif
    EXPECT_THAT(std::imag(std::proj(z1n)), IsPositiveZero());
    EXPECT_THAT(std::real(std::proj(z2)), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(std::imag(std::proj(z2)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ProjOfFloat)
{
    auto z = 1.2f;
    auto z1 = std::numeric_limits<float>::infinity();
    auto z1n = -std::numeric_limits<float>::infinity();
    auto z2 = 0.0f;

    EXPECT_THAT(double(std::real(std::proj(z))), DoubleNear(1.2, 1e-6));
    EXPECT_THAT(double(std::imag(std::proj(z))), DoubleNear(0.0, 1e-6));
    EXPECT_THAT(double(std::real(std::proj(z1))), IsPositiveInf());
    EXPECT_THAT(double(std::imag(std::proj(z1))), IsPositiveZero());
#if ((defined(_MSC_VER) && _MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) && !defined(__clang__)
    // VS 2017 evaluates this differently
    EXPECT_THAT(double(std::real(std::proj(z1n))), IsNegativeInf());
#else
    EXPECT_THAT(double(std::real(std::proj(z1n))), IsPositiveInf());
#endif
    EXPECT_THAT(double(std::imag(std::proj(z1n))), IsPositiveZero());
    EXPECT_THAT(double(std::real(std::proj(z2))), DoubleNear(0.0, 1e-6));
    EXPECT_THAT(double(std::imag(std::proj(z2))), DoubleNear(0.0, 1e-6));
}

TYPED_TEST(ComplexTest, ProjOfScalar)
{
    TypeParam z = 1.2;
    TypeParam z1 = std::numeric_limits<double>::infinity();
    TypeParam z1n = -std::numeric_limits<double>::infinity();
    TypeParam z2 = 0.0;

    EXPECT_THAT(xad::value(std::real(std::proj(z))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::proj(z))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::real(std::proj(z1))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(std::proj(z1))), IsPositiveZero());
#if ((defined(_MSC_VER) && _MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) && !defined(__clang__)
    // VS 2017 evaluates this differently
    EXPECT_THAT(xad::value(std::real(std::proj(z1n))), IsNegativeInf());
#else
    EXPECT_THAT(xad::value(std::real(std::proj(z1n))), IsPositiveInf());
#endif
    EXPECT_THAT(xad::value(std::imag(std::proj(z1n))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(std::proj(z2))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::proj(z2))), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, ProjOfScalarExpressions)
{
    TypeParam x = 2.0;
    auto r = x * 2.0;
    auto c = std::proj(r);
    EXPECT_THAT(xad::value(std::real(c)), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(std::imag(c)), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, ProjOfNanArguments)
{
    auto x = TypeParam(std::numeric_limits<double>::quiet_NaN());
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    auto z2 = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    auto z3 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity());
    auto z4 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      -std::numeric_limits<double>::infinity());
    auto z5 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    auto z6 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());

    EXPECT_THAT(xad::value(std::real(std::proj(x))), IsNan());
    EXPECT_THAT(xad::value(std::imag(std::proj(x))), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::real(std::proj(z1))), IsNan());
    EXPECT_THAT(xad::value(std::imag(std::proj(z1))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::real(std::proj(z2))), DoubleNear(1.2, 1e-9));
    EXPECT_THAT(xad::value(std::imag(std::proj(z2))), IsNan());
    EXPECT_THAT(xad::value(std::real(std::proj(z3))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(std::proj(z3))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(std::proj(z4))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(std::proj(z4))), IsNegativeZero());
    EXPECT_THAT(xad::value(std::real(std::proj(z5))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(std::proj(z5))), IsPositiveZero());
    EXPECT_THAT(xad::value(std::real(std::proj(z6))), IsPositiveInf());
    EXPECT_THAT(xad::value(std::imag(std::proj(z6))), IsPositiveZero());
}

// --------------- polar -----------------------

TYPED_TEST(ComplexTest, PolarOfComplexDouble)
{
    EXPECT_THAT(xad::value(std::polar(1.0, M_PI / 4).real()), DoubleNear(0.7071067811865476, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, M_PI / 4).imag()), DoubleNear(0.7071067811865476, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, 3 * M_PI / 4).real()),
                DoubleNear(-0.7071067811865475, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, 3 * M_PI / 4).imag()),
                DoubleNear(0.7071067811865476, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, -M_PI / 4).real()),
                DoubleNear(0.7071067811865476, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, -M_PI / 4).imag()),
                DoubleNear(-0.7071067811865476, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, -3 * M_PI / 4).real()),
                DoubleNear(-0.7071067811865475, 1e-9));
    EXPECT_THAT(xad::value(std::polar(1.0, -3 * M_PI / 4).imag()),
                DoubleNear(-0.7071067811865475, 1e-9));
    EXPECT_THAT(xad::value(std::polar(0.0, 0.0).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::polar(0.0, 0.0).imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexTest, PolarOfScalarExpressions)
{
    TypeParam x = 2.0;
    TypeParam r = 4.0;
    auto r_expr = x * x;
    TypeParam phi = 4.0;
    auto phi_expr = phi * 1.0;
    double phi_d = 4.0;
    double r_d = 4.0;

    EXPECT_THAT(xad::value(std::polar(r, phi).real()), DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r, phi).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi).real()), DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r, phi_expr).real()), DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r, phi_expr).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi_expr).real()),
                DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi_expr).imag()),
                DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d, phi_expr).real()),
                DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d, phi_expr).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d, phi).real()), DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d, phi).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r, phi_d).real()), DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r, phi_d).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi_d).real()),
                DoubleNear(-2.6145744834544478, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_expr, phi_d).imag()), DoubleNear(-3.027209981231713, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r).real()), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r).imag()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d).real()), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(std::polar(r_d).imag()), DoubleNear(0.0, 1e-9));
}

// --------------- exp -----------------------

TYPED_TEST(ComplexTest, ExpOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(exp(z).real()), DoubleNear(1.203070112722819, 1e-9));
    EXPECT_THAT(xad::value(exp(z).imag()), DoubleNear(3.0944787419716917, 1e-9));
    EXPECT_THAT(xad::value(exp(z1).real()), DoubleNear(1.203070112722819, 1e-9));
    EXPECT_THAT(xad::value(exp(z1).imag()), DoubleNear(-3.0944787419716917, 1e-9));
    EXPECT_THAT(xad::value(exp(z2).real()), DoubleNear(0.10914005828987695, 1e-9));
    EXPECT_THAT(xad::value(exp(z2).imag()), DoubleNear(0.2807247779692679, 1e-9));
    EXPECT_THAT(xad::value(exp(z3).real()), DoubleNear(0.10914005828987695, 1e-9));
    EXPECT_THAT(xad::value(exp(z3).imag()), DoubleNear(-0.2807247779692679, 1e-9));
    EXPECT_THAT(xad::value(exp(z4).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(exp(z4).imag()), DoubleNear(0.0, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, ExpOfZeros)
{
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(0.0, 0.0)).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(0.0, 0.0)).imag()), IsPositiveZero());
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-0.0, 0.0)).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-0.0, 0.0)).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, ExpOfInfImag)
{
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity())))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity())))
            .imag(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity())))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity())))
            .imag(),
        IsNan());
}

TYPED_TEST(ComplexComplianceTest, ExpOfNanImag)
{
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN())))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN())))
            .imag(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN())))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN())))
            .imag(),
        IsNan());
}

TYPED_TEST(ComplexComplianceTest, ExpOfInfReal)
{
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 0.0)))
            .real(),
        IsPositiveInf());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 0.0)))
            .imag(),
        IsPositiveZero());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.7)))
            .real(),
        IsNegativeZero());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.7)))
            .imag(),
        IsPositiveZero());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), -0.2)))
            .real(),
        IsPositiveZero());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), -0.2)))
            .imag(),
        IsNegativeZero());
}

TYPED_TEST(ComplexComplianceTest, ExpOfInfBoth)
{
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity())))
                    .real(),
                AnyOf(IsPositiveZero(), IsNegativeZero()));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity())))
                    .imag(),
                AnyOf(IsPositiveZero(), IsNegativeZero()));

    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity())))
                    .real(),
                AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::infinity())))
                    .imag(),
                IsNan());
}

TYPED_TEST(ComplexComplianceTest, ExpOfNaN)
{
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .real(),
                AnyOf(IsPositiveZero(), IsNegativeZero()));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .imag(),
                AnyOf(IsPositiveZero(), IsNegativeZero()));

    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .real(),
                AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .imag(),
                IsNan());

    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0)))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0)))
            .imag(),
        IsPositiveZero());

    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2)))
            .real(),
        IsNan());
    EXPECT_THAT(
        xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2)))
            .imag(),
        IsNan());
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .real(),
                IsNan());
    EXPECT_THAT(xad::value(exp(std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                                       std::numeric_limits<double>::quiet_NaN())))
                    .imag(),
                IsNan());
}

// --------------- log -----------------------

TYPED_TEST(ComplexTest, LogOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(log(z).real()), DoubleNear(0.5288951470739273, 1e-9));
    EXPECT_THAT(xad::value(log(z).imag()), DoubleNear(0.7853981633974483, 1e-9));
    EXPECT_THAT(xad::value(log(z1).real()), DoubleNear(0.5288951470739273, 1e-9));
    EXPECT_THAT(xad::value(log(z1).imag()), DoubleNear(-0.7853981633974483, 1e-9));
    EXPECT_THAT(xad::value(log(z2).real()), DoubleNear(0.5288951470739273, 1e-9));
    EXPECT_THAT(xad::value(log(z2).imag()), DoubleNear(2.356194490192345, 1e-9));
    EXPECT_THAT(xad::value(log(z3).real()), DoubleNear(0.5288951470739273, 1e-9));
    EXPECT_THAT(xad::value(log(z3).imag()), DoubleNear(-2.356194490192345, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, LogOfZero)
{
    auto z4 = std::complex<TypeParam>(-0.0, +0.0);
    EXPECT_THAT(xad::value(log(z4).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(log(z4).imag()), DoubleNear(M_PI, 1e-9));

    auto z5 = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(log(z5).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(log(z5).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, LogOfInfImag)
{
    auto z6 = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(log(z6).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z6).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, LogOfNanImag)
{
    auto z6a = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(log(z6a).real()), IsNan());
    EXPECT_THAT(xad::value(log(z6a).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, LogOfInfReal)
{

    auto z7 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(log(z7).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z7).imag()), DoubleNear(M_PI, 1e-9));

    auto z8 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(log(z8).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z8).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, LogOfInfBoth)
{
    auto z9 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(log(z9).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z9).imag()), DoubleNear(3 * M_PI / 4, 1e-9));

    auto z10 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(log(z10).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z10).imag()), DoubleNear(M_PI / 4, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, LogOfNan)
{
    auto z11 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(log(z11).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z11).imag()), IsNan());

    auto z12 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(log(z12).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z12).imag()), IsNan());

    auto z13 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(log(z13).real()), IsNan());
    EXPECT_THAT(xad::value(log(z13).imag()), IsNan());

    auto z14 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(log(z14).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(log(z14).imag()), IsNan());

    auto z15 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(log(z15).real()), IsNan());
    EXPECT_THAT(xad::value(log(z15).imag()), IsNan());
}

// --------------- log10 -----------------------

TYPED_TEST(ComplexTest, Log10OfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(log10(z).real()), DoubleNear(0.2296962438796154, 1e-9));
    EXPECT_THAT(xad::value(log10(z).imag()), DoubleNear(0.3410940884604603, 1e-9));
    EXPECT_THAT(xad::value(log10(z1).real()), DoubleNear(0.2296962438796154, 1e-9));
    EXPECT_THAT(xad::value(log10(z1).imag()), DoubleNear(-0.3410940884604603, 1e-9));
    EXPECT_THAT(xad::value(log10(z2).real()), DoubleNear(0.2296962438796154, 1e-9));
    EXPECT_THAT(xad::value(log10(z2).imag()), DoubleNear(1.023282265381381, 1e-9));
    EXPECT_THAT(xad::value(log10(z3).real()), DoubleNear(0.2296962438796154, 1e-9));
    EXPECT_THAT(xad::value(log10(z3).imag()), DoubleNear(-1.023282265381381, 1e-9));
}

// Note: compliance not needed here, as log10 is based on log, which checks them
// all

// --------------- pow -----------------------

TYPED_TEST(ComplexTest, PowComplex)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    TypeParam s = 1.2;
    auto z = pow(x, x);
    auto z1 = pow(x, s);
    auto z2 = pow(s, x);

    EXPECT_THAT(xad::value(z.real()), DoubleNear(-0.0046717473364405165, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(0.7350448091182941, 1e-9));
    EXPECT_THAT(xad::value(z1.real()), DoubleNear(1.108802452728176, 1e-9));
    EXPECT_THAT(xad::value(z1.imag()), DoubleNear(1.5261356493091542, 1e-9));
    EXPECT_THAT(xad::value(z2.real()), DoubleNear(1.21489642633261, 1e-9));
    EXPECT_THAT(xad::value(z2.imag()), DoubleNear(0.2701260507006947, 1e-9));
}

TYPED_TEST(ComplexTest, PromotePowArguments)
{
    auto x = std::complex<double>(1.2, 1.2);
    int s = 2;
    auto z1 = pow(x, s);
    auto z2 = pow(s, x);

    static_assert((std::is_same<decltype(z1), std::complex<double>>::value),
                  "mismatch in expected return type");
    static_assert((std::is_same<decltype(z2), std::complex<double>>::value),
                  "mismatch in expected return type");

    EXPECT_THAT(xad::value(z1.real()), DoubleNear(1.7634913907721887e-16, 1e-9));
    EXPECT_THAT(xad::value(z1.imag()), DoubleNear(2.88, 1e-9));
    EXPECT_THAT(xad::value(z2.real()), DoubleNear(1.5474429697835443, 1e-9));
    EXPECT_THAT(xad::value(z2.imag()), DoubleNear(1.6980729955920808, 1e-9));
}

TYPED_TEST(ComplexTest, PromotePowArgumentsExpression1)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    std::vector<std::complex<TypeParam>> z;
    using std::pow;
    z.emplace_back(pow(x * 1.0, 2));
    z.emplace_back(pow(x * 1.0, static_cast<short>(2)));
    z.emplace_back(pow(x * 1.0, 2u));
    z.emplace_back(pow(x * 1.0, 2ul));
    z.emplace_back(pow(x * 1.0, 2l));
#if !defined(_MSC_VER) || _MSC_VER >= 1920
    // VS 2017 issues a warning for these in double and converts them to double
    // version, so no need to test this
    z.emplace_back(pow(x * 1.0, 2ull));
    z.emplace_back(pow(x * 1.0, 2ll));
#endif
    z.emplace_back(pow(x * 1.0, static_cast<unsigned short>(2)));

    for (auto& zi : z)
    {
        EXPECT_THAT(xad::value(zi.real()), DoubleNear(1.7634913907721887e-16, 1e-9));
        EXPECT_THAT(xad::value(zi.imag()), DoubleNear(2.88, 1e-9));
    }
}

TYPED_TEST(ComplexTest, PromotePowArgumentsExpression2)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    std::vector<std::complex<TypeParam>> z;
    using std::pow;
    z.emplace_back(pow(2, x * 1.0));
    z.emplace_back(pow(static_cast<short>(2), x * 1.0));
    z.emplace_back(pow(2u, x * 1.0));
    z.emplace_back(pow(2ul, x * 1.0));
    z.emplace_back(pow(2l, x * 1.0));
#if !defined(_MSC_VER) || _MSC_VER >= 1920
    // VS 2017 issues a warning for these in double and converts them to double version,
    // so no need to test this
    z.emplace_back(pow(2ull, x * 1.0));
    z.emplace_back(pow(2ll, x * 1.0));
#endif
    z.emplace_back(pow(static_cast<unsigned short>(2), x * 1.0));

    for (auto& zi : z)
    {
        EXPECT_THAT(xad::value(zi.real()), DoubleNear(1.5474429697835443, 1e-9));
        EXPECT_THAT(xad::value(zi.imag()), DoubleNear(1.6980729955920808, 1e-9));
    }
}

TYPED_TEST(ComplexTest, PromoteADTypeAndScalar)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    auto y2 = std::complex<double>(1.0, 1.0);

    auto z2 = pow(x, y2);
    auto z4 = pow(y2, x);

    static_assert((std::is_same<decltype(z2), std::complex<TypeParam>>::value),
                  "mismatch in expected return type");
    static_assert((std::is_same<decltype(z4), std::complex<TypeParam>>::value),
                  "mismatch in expected return type");
}

TYPED_TEST(ComplexTest, PowOfScalarExpressions)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    TypeParam s = 1.2;
    auto z_0 = pow(x * 1.0, x);
    auto z_1 = pow(x * 1.0, x * 1.0);
    auto z1_0 = pow(x, s * 1.0);
    auto z1_1 = pow(x * 1.0, s);
    auto z2_0 = pow(s * 1.0, x);
    auto z2_1 = pow(s, x * 1.0);
    auto z3 = pow(x, s);
    auto z4 = pow(s, x);

    EXPECT_THAT(xad::value(z_0.real()), DoubleNear(-0.0046717473364405165, 1e-9));
    EXPECT_THAT(xad::value(z_0.imag()), DoubleNear(0.7350448091182941, 1e-9));
    EXPECT_THAT(xad::value(z_1.real()), DoubleNear(-0.0046717473364405165, 1e-9));
    EXPECT_THAT(xad::value(z_1.imag()), DoubleNear(0.7350448091182941, 1e-9));
    EXPECT_THAT(xad::value(z1_0.real()), DoubleNear(1.108802452728176, 1e-9));
    EXPECT_THAT(xad::value(z1_0.imag()), DoubleNear(1.5261356493091542, 1e-9));
    EXPECT_THAT(xad::value(z1_1.real()), DoubleNear(1.108802452728176, 1e-9));
    EXPECT_THAT(xad::value(z1_1.imag()), DoubleNear(1.5261356493091542, 1e-9));
    EXPECT_THAT(xad::value(z2_0.real()), DoubleNear(1.21489642633261, 1e-9));
    EXPECT_THAT(xad::value(z2_0.imag()), DoubleNear(0.2701260507006947, 1e-9));
    EXPECT_THAT(xad::value(z2_1.real()), DoubleNear(1.21489642633261, 1e-9));
    EXPECT_THAT(xad::value(z2_1.imag()), DoubleNear(0.2701260507006947, 1e-9));
    EXPECT_THAT(xad::value(z3.real()), DoubleNear(1.108802452728176, 1e-9));
    EXPECT_THAT(xad::value(z3.imag()), DoubleNear(1.5261356493091542, 1e-9));
    EXPECT_THAT(xad::value(z4.real()), DoubleNear(1.21489642633261, 1e-9));
    EXPECT_THAT(xad::value(z4.imag()), DoubleNear(0.2701260507006947, 1e-9));
}

TYPED_TEST(ComplexTest, PowWithDoubles)
{
    auto x = std::complex<TypeParam>(1.2, 1.2);
    double s = 1.2;
    auto z1 = pow(x, s);
    auto z2 = pow(s, x);

    EXPECT_THAT(xad::value(z1.real()), DoubleNear(1.108802452728176, 1e-9));
    EXPECT_THAT(xad::value(z1.imag()), DoubleNear(1.5261356493091542, 1e-9));
    EXPECT_THAT(xad::value(z2.real()), DoubleNear(1.21489642633261, 1e-9));
    EXPECT_THAT(xad::value(z2.imag()), DoubleNear(0.2701260507006947, 1e-9));
}

// compliance tests are not needed, as it's specified to behave as exp(log(x) *
// y), which is exactly what it is doing

// --------------- sqrt -----------------------

TYPED_TEST(ComplexTest, SqrtOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(sqrt(z).real()), DoubleNear(1.20354814503777, 1e-9));
    EXPECT_THAT(xad::value(sqrt(z).imag()), DoubleNear(0.49852596464362514, 1e-9));

    EXPECT_THAT(xad::value(sqrt(z1).real()), DoubleNear(1.20354814503777, 1e-9));
    EXPECT_THAT(xad::value(sqrt(z1).imag()), DoubleNear(-0.49852596464362514, 1e-9));

    EXPECT_THAT(xad::value(sqrt(z2).real()), DoubleNear(0.49852596464362514, 1e-9));
    EXPECT_THAT(xad::value(sqrt(z2).imag()), DoubleNear(1.20354814503777, 1e-9));

    EXPECT_THAT(xad::value(sqrt(z3).real()), DoubleNear(0.49852596464362514, 1e-9));
    EXPECT_THAT(xad::value(sqrt(z3).imag()), DoubleNear(-1.20354814503777, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, SqrtOfZero)
{
    // If z is (+-0,+0), the result is (+0,+0)
    auto z4 = std::complex<TypeParam>(+0.0, +0.0);
    auto z5 = std::complex<TypeParam>(-0.0, +0.0);

    EXPECT_THAT(xad::value(sqrt(z4).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(sqrt(z4).imag()), IsPositiveZero());

    EXPECT_THAT(xad::value(sqrt(z5).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(sqrt(z5).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, SqrtOfInfImag)
{
    // If z is (x,+INFINITY), the result is (+INFINITY,+INFINITY) even if x is NaN
    auto z6 = std::complex<TypeParam>(1.2, +std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(sqrt(z6).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sqrt(z6).imag()), IsPositiveInf());

    auto z7 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(sqrt(z6).real()),
                DoubleNear(+std::numeric_limits<double>::infinity(), 1e-9));
    EXPECT_THAT(xad::value(sqrt(z6).imag()),
                DoubleNear(+std::numeric_limits<double>::infinity(), 1e-9));
    EXPECT_THAT(xad::value(sqrt(z7).real()),
                DoubleNear(+std::numeric_limits<double>::infinity(), 1e-9));
    EXPECT_THAT(xad::value(sqrt(z7).imag()),
                DoubleNear(+std::numeric_limits<double>::infinity(), 1e-9));
}

TYPED_TEST(ComplexComplianceTest, SqrtOfNaNImag)
{
    auto z = std::complex<TypeParam>(1.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sqrt(z).real()), IsNan());
    EXPECT_THAT(xad::value(sqrt(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SqrtOfInfReal)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(sqrt(z1).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sqrt(z1).imag()), IsPositiveZero());

    auto z2 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(sqrt(z2).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(sqrt(z2).imag()), IsPositiveInf());
}

TYPED_TEST(ComplexComplianceTest, SqrtOfInfRealNanImag)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sqrt(z1).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sqrt(z1).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sqrt(z2).real()), IsNan());
    EXPECT_THAT(xad::value(sqrt(z2).imag()), AnyOf(IsPositiveInf(), IsNegativeInf()));
}

TYPED_TEST(ComplexComplianceTest, SqrtOfNanReal)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(sqrt(z1).real()), IsNan());
    EXPECT_THAT(xad::value(sqrt(z1).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SqrtOfNanBoth)
{
    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sqrt(z2).real()), IsNan());
    EXPECT_THAT(xad::value(sqrt(z2).imag()), IsNan());
}

// --------------- sin -----------------------

TYPED_TEST(ComplexTest, SinOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(sin(z).real()), DoubleNear(1.6876017599704798, 1e-9));
    EXPECT_THAT(xad::value(sin(z).imag()), DoubleNear(0.546965027216471, 1e-9));

    EXPECT_THAT(xad::value(sin(z1).real()), DoubleNear(1.6876017599704798, 1e-9));
    EXPECT_THAT(xad::value(sin(z1).imag()), DoubleNear(-0.546965027216471, 1e-9));

    EXPECT_THAT(xad::value(sin(z2).real()), DoubleNear(-1.6876017599704798, 1e-9));
    EXPECT_THAT(xad::value(sin(z2).imag()), DoubleNear(0.546965027216471, 1e-9));

    EXPECT_THAT(xad::value(sin(z3).real()), DoubleNear(-1.6876017599704798, 1e-9));
    EXPECT_THAT(xad::value(sin(z3).imag()), DoubleNear(-0.546965027216471, 1e-9));

    EXPECT_THAT(xad::value(sin(z4).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(sin(z4).imag()), DoubleNear(0.0, 1e-9));
}

// compliance tests not necessary if it behaves like -i * sinh(i*z),
// which is how it is implemented

// --------------- cos -----------------------

TYPED_TEST(ComplexTest, CosOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(cos(z).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cos(z).imag()), DoubleNear(-1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cos(z1).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cos(z1).imag()), DoubleNear(1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cos(z2).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cos(z2).imag()), DoubleNear(1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cos(z3).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cos(z3).imag()), DoubleNear(-1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cos(z4).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(cos(z4).imag()), DoubleNear(0.0, 1e-9));
}

// compliance tests not necessary if it behaves like cosh(i*z),
// which is how it is implemented

// --------------- tan -----------------------

TYPED_TEST(ComplexTest, TanOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(tan(z).real()), DoubleNear(0.14015057356642452, 1e-9));
    EXPECT_THAT(xad::value(tan(z).imag()), DoubleNear(1.134177526770811, 1e-9));

    EXPECT_THAT(xad::value(tan(z1).real()), DoubleNear(0.14015057356642452, 1e-9));
    EXPECT_THAT(xad::value(tan(z1).imag()), DoubleNear(-1.134177526770811, 1e-9));

    EXPECT_THAT(xad::value(tan(z2).real()), DoubleNear(-0.14015057356642452, 1e-9));
    EXPECT_THAT(xad::value(tan(z2).imag()), DoubleNear(1.134177526770811, 1e-9));

    EXPECT_THAT(xad::value(tan(z3).real()), DoubleNear(-0.14015057356642452, 1e-9));
    EXPECT_THAT(xad::value(tan(z3).imag()), DoubleNear(-1.134177526770811, 1e-9));

    EXPECT_THAT(xad::value(tan(z4).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(tan(z4).imag()), DoubleNear(0.0, 1e-9));
}

// compliance tests not necessary if it behaves like -i*tanh(i*z),
// which is how it is implemented

// --------------- asin -----------------------

TYPED_TEST(ComplexTest, AsinOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(asin(z).real()), DoubleNear(0.7006571388031325, 1e-9));
    EXPECT_THAT(xad::value(asin(z).imag()), DoubleNear(1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(asin(z1).real()), DoubleNear(0.7006571388031325, 1e-9));
    EXPECT_THAT(xad::value(asin(z1).imag()), DoubleNear(-1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(asin(z2).real()), DoubleNear(-0.7006571388031325, 1e-9));
    EXPECT_THAT(xad::value(asin(z2).imag()), DoubleNear(1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(asin(z3).real()), DoubleNear(-0.7006571388031325, 1e-9));
    EXPECT_THAT(xad::value(asin(z3).imag()), DoubleNear(-1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(asin(z4).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(asin(z4).imag()), DoubleNear(0.0, 1e-9));
}

// compliance tests not necessary if it behaves like -i*asinh(i*z),
// which is how it is implemented

// --------------- acos -----------------------

TYPED_TEST(ComplexTest, AcosOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(acos(z).real()), DoubleNear(0.8701391879917642, 1e-9));
    EXPECT_THAT(xad::value(acos(z).imag()), DoubleNear(-1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(acos(z1).real()), DoubleNear(0.8701391879917642, 1e-9));
    EXPECT_THAT(xad::value(acos(z1).imag()), DoubleNear(1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(acos(z2).real()), DoubleNear(2.271453465598029, 1e-9));
    EXPECT_THAT(xad::value(acos(z2).imag()), DoubleNear(-1.2328788717473873, 1e-9));

    EXPECT_THAT(xad::value(acos(z3).real()), DoubleNear(2.271453465598029, 1e-9));
    EXPECT_THAT(xad::value(acos(z3).imag()), DoubleNear(1.2328788717473873, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AcosOfZero)
{
    // If z is (+-0,+0), the result is (M_PI/2,-0)
    auto z4 = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(acos(z4).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z4).imag()), IsNegativeZero());

    auto z5 = std::complex<TypeParam>(-0.0, +0.0);
    EXPECT_THAT(xad::value(acos(z5).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z5).imag()), IsNegativeZero());
}

TYPED_TEST(ComplexComplianceTest, AcosOfZeroRealAndNaNImag)
{
    auto z1 = std::complex<TypeParam>(+0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z1).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z2).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcosOfInfiniteImag)
{
    auto z1 = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acos(z1).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNegativeInf());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acos(z2).real()), DoubleNear(M_PI / 2, 1e-9));
    EXPECT_THAT(xad::value(acos(z2).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, AcosOfNanImag)
{
    auto z1 = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z1).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcosOfInfiniteReal)
{
    auto z1 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(acos(z1).real()), DoubleNear(M_PI, 1e-9));
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNegativeInf());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(acos(z2).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(acos(z2).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, AcosOfInfinityBoth)
{
    // If z is (-INFINITY,+INFINITY), the result is (3*M_PI/4,-INFINITY)
    auto z1 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      +std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acos(z1).real()), DoubleNear(3.0 * M_PI / 4.0, 1e-9));
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNegativeInf());

    // If z is (+INFINITY,+INFINITY), the result is (M_PI/4,-INFINITY)
    auto z2 = std::complex<TypeParam>(+std::numeric_limits<double>::infinity(),
                                      +std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acos(z2).real()), DoubleNear(M_PI / 4, 1e-9));
    EXPECT_THAT(xad::value(acos(z2).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, AcosOfInfiniteRealAndNanImag)
{
    auto z1 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z1).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z1).imag()), AnyOf(IsNegativeInf(), IsPositiveInf()));

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z2).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z2).imag()), AnyOf(IsNegativeInf(), IsPositiveInf()));
}

TYPED_TEST(ComplexComplianceTest, AcosOfNanReal)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(acos(z1).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(acos(z2).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcosOfNanRealInfImag)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acos(z1).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, AcosOfNanBoth)
{
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acos(z1).real()), IsNan());
    EXPECT_THAT(xad::value(acos(z1).imag()), IsNan());
}

// --------------- atan -----------------------

TYPED_TEST(ComplexTest, AtanOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);
    auto z4 = std::complex<TypeParam>(0.0, 0.0);

    EXPECT_THAT(xad::value(atan(z).real()), DoubleNear(1.1176458629448267, 1e-9));
    EXPECT_THAT(xad::value(atan(z).imag()), DoubleNear(0.3613319731760209, 1e-9));

    EXPECT_THAT(xad::value(atan(z1).real()), DoubleNear(1.1176458629448267, 1e-9));
    EXPECT_THAT(xad::value(atan(z1).imag()), DoubleNear(-0.3613319731760209, 1e-9));

    EXPECT_THAT(xad::value(atan(z2).real()), DoubleNear(-1.1176458629448267, 1e-9));
    EXPECT_THAT(xad::value(atan(z2).imag()), DoubleNear(0.3613319731760209, 1e-9));

    EXPECT_THAT(xad::value(atan(z3).real()), DoubleNear(-1.1176458629448267, 1e-9));
    EXPECT_THAT(xad::value(atan(z3).imag()), DoubleNear(-0.3613319731760209, 1e-9));

    EXPECT_THAT(xad::value(atan(z4).real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(atan(z4).imag()), DoubleNear(0.0, 1e-9));
}

// compliance tests not necessary if it behaves like -i*atanh(i*z),
// which is how it is implemented

// --------------- sinh -----------------------

TYPED_TEST(ComplexTest, SinhOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(sinh(z).real()), DoubleNear(0.546965027216471, 1e-9));
    EXPECT_THAT(xad::value(sinh(z).imag()), DoubleNear(1.6876017599704798, 1e-9));

    EXPECT_THAT(xad::value(sinh(z1).real()), DoubleNear(0.546965027216471, 1e-9));
    EXPECT_THAT(xad::value(sinh(z1).imag()), DoubleNear(-1.6876017599704798, 1e-9));

    EXPECT_THAT(xad::value(sinh(z2).real()), DoubleNear(-0.546965027216471, 1e-9));
    EXPECT_THAT(xad::value(sinh(z2).imag()), DoubleNear(1.6876017599704798, 1e-9));

    EXPECT_THAT(xad::value(sinh(z3).real()), DoubleNear(-0.546965027216471, 1e-9));
    EXPECT_THAT(xad::value(sinh(z3).imag()), DoubleNear(-1.6876017599704798, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, SinhOfZero)
{
    // If z is (+0,+0), the result is (+0,+0)
    auto z4 = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(sinh(z4).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(sinh(z4).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, SinhOfZeroRealInfImag)
{
    auto z = std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(sinh(z).real()), AnyOf(IsPositiveZero(), IsNegativeZero()));
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfZeroRealNanImag)
{
    auto z = std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sinh(z).real()), AnyOf(IsPositiveZero(), IsNegativeZero()));
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfPosRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(sinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfPosRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfInfRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 0.0);
    EXPECT_THAT(xad::value(sinh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, SinhOfInfRealPositiveImag)
{
    // result should be +inf * (cos(1.2) + i* sin(1.2))
    // therefore real and imag can only be inf / -inf

    // both cos/sin of 1.2 are positive
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(sinh(z1).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sinh(z1).imag()), IsPositiveInf());

    // cos(1.7) is negative, sin(1.7) is positive
    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.7);
    EXPECT_THAT(xad::value(sinh(z2).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(sinh(z2).imag()), IsPositiveInf());

    // cos(3.2) is negative, sin(3.2) is negative
    auto z3 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 3.2);
    EXPECT_THAT(xad::value(sinh(z3).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(sinh(z3).imag()), IsNegativeInf());

    // cos(6.0) is positive, sin(6.0) is negative
    auto z4 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 6.0);
    EXPECT_THAT(xad::value(sinh(z4).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(sinh(z4).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, SinhOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(sinh(z).real()), AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfInfRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sinh(z).real()), AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfNaNRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    EXPECT_THAT(xad::value(sinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, SinhOfNaNRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(sinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(sinh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, SinhOfNaNRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(sinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(sinh(z).imag()), IsNan());
}

// --------------- cosh -----------------------

TYPED_TEST(ComplexTest, CoshOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(cosh(z).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cosh(z).imag()), DoubleNear(1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cosh(z1).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cosh(z1).imag()), DoubleNear(-1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cosh(z2).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cosh(z2).imag()), DoubleNear(-1.4068769820012117, 1e-9));

    EXPECT_THAT(xad::value(cosh(z3).real()), DoubleNear(0.6561050855063479, 1e-9));
    EXPECT_THAT(xad::value(cosh(z3).imag()), DoubleNear(1.4068769820012117, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, CoshOfZero)
{
    auto z = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(cosh(z).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(cosh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, CoshOfZeroRealInfImag)
{
    auto z = std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), AnyOf(IsPositiveZero(), IsNegativeZero()));
}

TYPED_TEST(ComplexComplianceTest, CoshOfZeroRealNanImag)
{
    auto z = std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), AnyOf(IsPositiveZero(), IsNegativeZero()));
}

TYPED_TEST(ComplexComplianceTest, CoshOfFiniteRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(cosh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, CoshOfFiniteRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());

    auto z1 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(cosh(z1).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z1).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, CoshOfInfRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 0.0);
    EXPECT_THAT(xad::value(cosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, CoshOfInfRealPositiveImag)
{
    // result should be +inf * (cos(1.2) + i* sin(1.2))
    // therefore real and imag can only be inf / -inf

    // both cos/sin of 1.2 are positive
    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(cosh(z1).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(cosh(z1).imag()), IsPositiveInf());

    // cos(1.7) is negative, sin(1.7) is positive
    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.7);
    EXPECT_THAT(xad::value(cosh(z2).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(cosh(z2).imag()), IsPositiveInf());

    // cos(3.2) is negative, sin(3.2) is negative
    auto z3 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 3.2);
    EXPECT_THAT(xad::value(cosh(z3).real()), IsNegativeInf());
    EXPECT_THAT(xad::value(cosh(z3).imag()), IsNegativeInf());

    // cos(6.0) is positive, sin(6.0) is negative
    auto z4 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 6.0);
    EXPECT_THAT(xad::value(cosh(z4).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(cosh(z4).imag()), IsNegativeInf());
}

TYPED_TEST(ComplexComplianceTest, CoshOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(cosh(z).real()), AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, CoshOfInfRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(cosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, CoshOfNaNRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), AnyOf(IsPositiveZero(), IsNegativeZero()));
}

TYPED_TEST(ComplexComplianceTest, CoshOfNaNRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(cosh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, CoshOfNaNRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(cosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(cosh(z).imag()), IsNan());
}

// --------------- tanh -----------------------

TYPED_TEST(ComplexTest, TanhOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(tanh(z).real()), DoubleNear(1.134177526770811, 1e-9));
    EXPECT_THAT(xad::value(tanh(z).imag()), DoubleNear(0.14015057356642452, 1e-9));

    EXPECT_THAT(xad::value(tanh(z1).real()), DoubleNear(1.134177526770811, 1e-9));
    EXPECT_THAT(xad::value(tanh(z1).imag()), DoubleNear(-0.14015057356642452, 1e-9));

    EXPECT_THAT(xad::value(tanh(z2).real()), DoubleNear(-1.134177526770811, 1e-9));
    EXPECT_THAT(xad::value(tanh(z2).imag()), DoubleNear(0.14015057356642452, 1e-9));

    EXPECT_THAT(xad::value(tanh(z3).real()), DoubleNear(-1.134177526770811, 1e-9));
    EXPECT_THAT(xad::value(tanh(z3).imag()), DoubleNear(-0.14015057356642452, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, TanhOfZero)
{
    auto z = std::complex<TypeParam>(0.0, 0.0);
    EXPECT_THAT(xad::value(tanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, TanhOfFiniteRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(tanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(tanh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z2).imag()), IsNan());

    auto z3 = std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity());
#if defined(__APPLE__)
    // on Mac, this return NaN (it shouldn't though)
    EXPECT_THAT(xad::value(tanh(z3).real()), IsNan());
#else
    EXPECT_THAT(xad::value(tanh(z3).real()), IsPositiveZero());
#endif
    EXPECT_THAT(xad::value(tanh(z3).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, TanhOfFiniteRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(tanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(tanh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z2).imag()), IsNan());

    auto z3 = std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN());
#if defined(__APPLE__)
    // Mac return Nan here
    EXPECT_THAT(xad::value(tanh(z3).real()), IsNan());
#else
    EXPECT_THAT(xad::value(tanh(z3).real()), IsPositiveZero());
#endif
    EXPECT_THAT(xad::value(tanh(z3).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, TanhOfInfRealPosImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(tanh(z).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(tanh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, TanhOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(tanh(z).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(tanh(z).imag()), AnyOf(IsPositiveZero(), IsNegativeZero()));
}

TYPED_TEST(ComplexComplianceTest, TanhOfInfRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(tanh(z).real()), DoubleNear(1.0, 1e-9));
    EXPECT_THAT(xad::value(tanh(z).imag()), AnyOf(IsPositiveZero(), IsNegativeZero()));
}

TYPED_TEST(ComplexComplianceTest, TanhOfNaNRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), +0.0);
    EXPECT_THAT(xad::value(tanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, TanhOfNaNRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(tanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsNan());

    auto z1 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(tanh(z1).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z1).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, TanhOfNaNRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(tanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(tanh(z).imag()), IsNan());
}

// --------------- asinh -----------------------

TYPED_TEST(ComplexTest, AsinhOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(asinh(z).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(asinh(z).imag()), DoubleNear(0.7006571388031325, 1e-9));

    EXPECT_THAT(xad::value(asinh(z1).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(asinh(z1).imag()), DoubleNear(-0.7006571388031325, 1e-9));

    EXPECT_THAT(xad::value(asinh(z2).real()), DoubleNear(-1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(asinh(z2).imag()), DoubleNear(0.7006571388031325, 1e-9));

    EXPECT_THAT(xad::value(asinh(z3).real()), DoubleNear(-1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(asinh(z3).imag()), DoubleNear(-0.7006571388031325, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AsinhOfZero)
{
    auto z = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(asinh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfPosRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(asinh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(asinh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AsinhOfFiniteRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(asinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(asinh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfInfRealPosImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(asinh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(asinh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(asinh(z).imag()), DoubleNear(M_PI / 4, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AsinhOfInfRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(asinh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfNanRealZeroImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), +0.0);
    EXPECT_THAT(xad::value(asinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfNanRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(asinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(asinh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfNanRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(asinh(z).real()), AnyOf(IsPositiveInf(), IsNegativeInf()));
    EXPECT_THAT(xad::value(asinh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AsinhOfNanRealNanImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(asinh(z).real()), IsNan());
    EXPECT_THAT(xad::value(asinh(z).imag()), IsNan());
}

// --------------- acosh -----------------------

TYPED_TEST(ComplexTest, AcoshOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(acosh(z).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(acosh(z).imag()), DoubleNear(0.8701391879917642, 1e-9));

    EXPECT_THAT(xad::value(acosh(z1).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(acosh(z1).imag()), DoubleNear(-0.8701391879917642, 1e-9));

    EXPECT_THAT(xad::value(acosh(z2).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(acosh(z2).imag()), DoubleNear(2.271453465598029, 1e-9));

    EXPECT_THAT(xad::value(acosh(z3).real()), DoubleNear(1.2328788717473873, 1e-9));
    EXPECT_THAT(xad::value(acosh(z3).imag()), DoubleNear(-2.271453465598029, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AcoshOfZero)
{
    auto z = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(acosh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AcoshOfFiniteRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z).imag()), DoubleNear(M_PI / 2, 1e-9));

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acosh(z2).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z2).imag()), DoubleNear(M_PI / 2, 1e-9));

    auto z3 = std::complex<TypeParam>(0.0, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acosh(z3).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z3).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AcoshOfFiniteRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z2).imag()), IsNan());

    auto z3 = std::complex<TypeParam>(0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z3).real()), IsNan());
#if defined(__APPLE__)
    EXPECT_THAT(xad::value(acosh(z3).imag()), IsNan());
#else
    EXPECT_THAT(xad::value(acosh(z3).imag()), DoubleNear(M_PI / 2, 1e-9));
#endif
}

TYPED_TEST(ComplexComplianceTest, AcoshOfInfRealPosImag)
{
    auto z = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z).imag()), DoubleNear(M_PI, 1e-9));

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(acosh(z2).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z2).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AcoshOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z).imag()), DoubleNear(3 * M_PI / 4, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AcoshOfInfRealNanImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z2).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcoshOfNanRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(acosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(acosh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z2).imag()), IsNan());

    auto z3 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    EXPECT_THAT(xad::value(acosh(z3).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z3).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcoshOfNanRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(acosh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(acosh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AcoshOfNanRealNaNImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(acosh(z).real()), IsNan());
    EXPECT_THAT(xad::value(acosh(z).imag()), IsNan());
}

// --------------- atanh -----------------------

TYPED_TEST(ComplexTest, AtanhOfComplex)
{
    auto z = std::complex<TypeParam>(1.2, 1.2);
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<TypeParam>(-1.2, 1.2);
    auto z3 = std::complex<TypeParam>(-1.2, -1.2);

    EXPECT_THAT(xad::value(atanh(z).real()), DoubleNear(0.3613319731760209, 1e-9));
    EXPECT_THAT(xad::value(atanh(z).imag()), DoubleNear(1.1176458629448267, 1e-9));

    EXPECT_THAT(xad::value(atanh(z1).real()), DoubleNear(0.3613319731760209, 1e-9));
    EXPECT_THAT(xad::value(atanh(z1).imag()), DoubleNear(-1.1176458629448267, 1e-9));

    EXPECT_THAT(xad::value(atanh(z2).real()), DoubleNear(-0.3613319731760209, 1e-9));
    EXPECT_THAT(xad::value(atanh(z2).imag()), DoubleNear(1.1176458629448267, 1e-9));

    EXPECT_THAT(xad::value(atanh(z3).real()), DoubleNear(-0.3613319731760209, 1e-9));
    EXPECT_THAT(xad::value(atanh(z3).imag()), DoubleNear(-1.1176458629448267, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AtanhOfZero)
{
    auto z = std::complex<TypeParam>(+0.0, +0.0);
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfZeroRealNanImag)
{
    auto z = std::complex<TypeParam>(+0.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfOneRealZeroImag)
{
    auto z = std::complex<TypeParam>(1.0, +0.0);
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveInf());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsPositiveZero());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfPosRealInfImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AtanhOfFiniteRealNanImag)
{
    auto z = std::complex<TypeParam>(1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(atanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(-1.2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(atanh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z2).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfInfRealPosImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(), 1.2);
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AtanhOfInfRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AtanhOfInfRealNanImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(atanh(z).real()), IsPositiveZero());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfNanRealFiniteImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 1.2);
    EXPECT_THAT(xad::value(atanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsNan());

    auto z2 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), -1.2);
    EXPECT_THAT(xad::value(atanh(z2).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z2).imag()), IsNan());

    auto z3 = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    EXPECT_THAT(xad::value(atanh(z3).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z3).imag()), IsNan());
}

TYPED_TEST(ComplexComplianceTest, AtanhOfNanRealInfImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
    EXPECT_THAT(xad::value(atanh(z).real()), AnyOf(IsPositiveZero(), IsNegativeZero()));
    EXPECT_THAT(xad::value(atanh(z).imag()), DoubleNear(M_PI / 2, 1e-9));
}

TYPED_TEST(ComplexComplianceTest, AtanhOfNanRealNanImag)
{
    auto z = std::complex<TypeParam>(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN());
    EXPECT_THAT(xad::value(atanh(z).real()), IsNan());
    EXPECT_THAT(xad::value(atanh(z).imag()), IsNan());
}

// ------------------ value function ---------------

TYPED_TEST(ComplexTest, ValueFunction)
{
    std::complex<TypeParam> z(1.2, -1.2);
    auto zv = xad::value(z);
    static_assert((std::is_same<decltype(zv), std::complex<double>>::value),
                  "Value of complex is not of plain double type");
}

// ----------------- streams -----------------

TYPED_TEST(ComplexTest, StreamOutput)
{
    std::complex<TypeParam> z(1.2, -1.2);
    std::stringstream sstr;
    sstr << z;
    auto str = sstr.str();
    EXPECT_THAT(str, Eq("(1.2,-1.2)"));
}

#if defined(_MSC_VER) && _MSC_VER < 1920
// VS 2017 implementation of complex stream read has a long double to double conversion,
// so we disable this warning here as we build with warnings flagged as errors
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

TYPED_TEST(ComplexTest, StreamInput)
{
    std::complex<TypeParam> z;
    std::stringstream sstr;
    sstr << "(1.23,-0.923)";
    sstr >> z;
    EXPECT_THAT(xad::value(z.real()), DoubleNear(1.23, 1e-9));
    EXPECT_THAT(xad::value(z.imag()), DoubleNear(-0.923, 1e-9));
}

#if defined(_MSC_VER) && _MSC_VER < 1920
#pragma warning(pop)
#endif

TYPED_TEST(ComplexTest, canEvaluateTroublesomeComplexPow)
{
    std::complex<double> ad_result, double_result;
    {
        // AD version
        using Real = TypeParam;
        Real rho_ = 0.12, n_ = 1.2, k_ = 0.4, w = 1.2;
        const std::complex<Real> s{1.2, -9.2};
        std::complex<Real> term2 =
            (1 - rho_ * rho_) * pow(((n_ - k_ + 1) * s + n_ * w), 2) / (2 * n_ * n_);
        ad_result = xad::value(term2);
    }

    {
        // double version
        double rho_ = 0.12, n_ = 1.2, k_ = 0.4, w = 1.2;
        const std::complex<double> s{1.2, -9.2};
        std::complex<double> term2 =
            (1 - rho_ * rho_) * pow(((n_ - k_ + 1) * s + n_ * w), 2) / (2 * n_ * n_);
        double_result = term2;
    }
    EXPECT_THAT(ad_result.real(), DoubleEq(double_result.real()));
    EXPECT_THAT(ad_result.imag(), DoubleEq(double_result.imag()));
}

TYPED_TEST(ComplexTest, canEvaluateTroublesomeComplexAbs)
{
    double ad_res, double_res;
    {
        // AD version
        const std::complex<TypeParam> si(1.2, 2.5);
        std::complex<TypeParam> ref(1.2, 0.4);
        TypeParam diff = std::abs(si - ref) / std::abs(ref);
        ad_res = xad::value(diff);
    }
    {
        // double version
        const std::complex<double> si(1.2, 2.5);
        std::complex<double> ref(1.2, 0.4);
        double diff = std::abs(si - ref) / std::abs(ref);
        double_res = diff;
    }

    EXPECT_THAT(ad_res, DoubleEq(double_res));
}
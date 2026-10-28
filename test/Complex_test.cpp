/*******************************************************************************

   Tests for std::complex with AD types.

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
// results instead.
// Then this could be parameterised tests for each function, where we'd just
// give input values (incl. special values) and the test compares to
// std::complex results.

using namespace ::testing;

typedef xad::AReal<double> dblAD;
typedef xad::AReal<double, 2> dblAD2;
typedef xad::FReal<double> dblFAD;
typedef xad::FReal<double, 2> dblFAD2;
typedef ::testing::Types<double, dblAD, dblFAD, dblAD2, dblFAD2> test_types;

template <class T>
class ComplexTest : public testing::Test
{
};

TYPED_TEST_SUITE(ComplexTest, test_types);

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

TYPED_TEST(ComplexTest, AddComplexDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<double>(1.2, 42.0);
    auto ret = z1 + z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(40.8, 1e-9));
}

TYPED_TEST(ComplexTest, AddDoubleComplex)
{
    auto z1 = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, AddScalarDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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
    std::complex<TypeParam> ret1 = z + (s * 1.0);
    std::complex<TypeParam> ret2 = (s * 1.0) + z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, AddScalarExpressionDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
    TypeParam s = 1.2;
    std::complex<TypeParam> ret1 = z + (s * 1.0);
    std::complex<TypeParam> ret2 = (s * 1.0) + z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.2, 1e-9));
}

TYPED_TEST(ComplexTest, AddDouble)
{
    auto z = std::complex<TypeParam>(1.2, -1.2);
    std::complex<TypeParam> ret1 = z + 1.2;
    std::complex<TypeParam> ret2 = 1.2 + z;
    EXPECT_THAT(xad::value(ret1.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret1.imag()), DoubleNear(-1.2, 1e-9));
    EXPECT_THAT(xad::value(ret2.real()), DoubleNear(2.4, 1e-9));
    EXPECT_THAT(xad::value(ret2.imag()), DoubleNear(-1.2, 1e-9));
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

TYPED_TEST(ComplexTest, SubstractComplexDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<double>(1.2, 42.0);
    auto ret = z1 - z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(0.0, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-43.2, 1e-9));
}

TYPED_TEST(ComplexTest, SubstractDoubleComplex)
{
    auto z1 = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, SubstractScalarDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, SubstractScalarExpressionDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, MultiplyComplexDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<double>(1.2, 42.0);
    auto ret = z1 * z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(51.84, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(48.96, 1e-9));
}

TYPED_TEST(ComplexTest, MultiplyDoubleComplex)
{
    auto z1 = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, MultiplyScalarDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, MultiplyScalarExpressionDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, DivideComplexDouble)
{
    auto z1 = std::complex<TypeParam>(1.2, -1.2);
    auto z2 = std::complex<double>(1.2, 42.0);
    auto ret = z1 / z2;
    EXPECT_THAT(xad::value(ret.real()), DoubleNear(-0.027732463295269166, 1e-9));
    EXPECT_THAT(xad::value(ret.imag()), DoubleNear(-0.029363784665579117, 1e-9));
}

TYPED_TEST(ComplexTest, DivideDoubleComplex)
{
    auto z1 = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, DivideScalarDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, DivideScalarExpressionDoubleCplx)
{
    auto z = std::complex<double>(1.2, -1.2);
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

TYPED_TEST(ComplexTest, ProjOfScalarExpressions)
{
    TypeParam x = 2.0;
    auto r = x * 2.0;
    auto c = std::proj(r);
    EXPECT_THAT(xad::value(std::real(c)), DoubleNear(4.0, 1e-9));
    EXPECT_THAT(xad::value(std::imag(c)), DoubleNear(0.0, 1e-9));
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

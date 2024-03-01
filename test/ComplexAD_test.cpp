/*******************************************************************************

   Tests for complex - computing derivatives.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

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

#include <XAD/Complex.hpp>
#include <XAD/XAD.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <complex>

using namespace ::testing;

typedef xad::AReal<double> dblAAD;
typedef dblAAD::tape_type tape_type;
typedef xad::FReal<double> dblFAD;
typedef ::testing::Types<dblAAD, dblFAD> test_types;

struct ValueAndDerivatives
{
    std::complex<double> value1, value2;
    std::complex<double> d_real;
    std::complex<double> d_imag;
    std::complex<double> val_ref;

    void compare(double rr, double ri, double ir, double ii)
    {
        EXPECT_THAT(value1.real(), NanSensitiveDoubleNear(val_ref.real(), 1e-9));
        EXPECT_THAT(value2.real(), NanSensitiveDoubleNear(val_ref.real(), 1e-9));
        EXPECT_THAT(value1.imag(), NanSensitiveDoubleNear(val_ref.imag(), 1e-9));
        EXPECT_THAT(value2.imag(), NanSensitiveDoubleNear(val_ref.imag(), 1e-9));
        EXPECT_THAT(d_real.real(), NanSensitiveDoubleNear(rr, 1e-9));
        EXPECT_THAT(d_real.imag(), NanSensitiveDoubleNear(ri, 1e-9));
        EXPECT_THAT(d_imag.real(), NanSensitiveDoubleNear(ir, 1e-9));
        EXPECT_THAT(d_imag.imag(), NanSensitiveDoubleNear(ii, 1e-9));
    }
};

template <class F, class T>
std::complex<double> calcReference(F func, const std::complex<T>& in)
{
    std::complex<double> in2(xad::value(in.real()), xad::value(in.imag()));
    return func(in2);
}

template <class T>
class ComplexADTest;

template <>
class ComplexADTest<dblFAD> : public ::testing::Test
{
  protected:
    template <class F>
    ValueAndDerivatives calcDerivatives(F func, std::complex<dblFAD> in)
    {
        ValueAndDerivatives ret;
        // standard reference
        ret.val_ref = calcReference(func, in);

        // first d_real
        in.setDerivative(1.0, 0.0);
        std::complex<dblFAD> out = func(in);
        ret.value1 = value(out);
        ret.d_real = derivative(out);

        // now d_imag
        in.setDerivative(0.0, 1.0);
        out = func(in);
        ret.value2 = value(out);
        ret.d_imag = derivative(out);
        return ret;
    }
};

template <>
class ComplexADTest<dblAAD> : public ::testing::Test
{
  protected:
    template <class F>
    ValueAndDerivatives calcDerivatives(F func, std::complex<dblAAD> in)
    {
        ValueAndDerivatives ret;

        // standard reference
        ret.val_ref = calcReference(func, in);

        tape.registerInput(in);
        tape.newRecording();
        std::complex<dblAAD> out = func(in);
        tape.registerOutput(out);
        ret.value1 = value(out);
        ret.value2 = ret.value1;
        out.setDerivative(1.0, 0.0);
        tape.computeAdjoints();
        auto der = in.getDerivative();
        auto der2 = derivative(in);
        EXPECT_THAT(der2.real(), DoubleEq(der.real()));
        EXPECT_THAT(der2.imag(), DoubleEq(der.imag()));
        ret.d_real.real(der.real());
        ret.d_imag.real(der.imag());

        tape.clearDerivatives();
        out.setDerivative(0.0, 1.0);
        tape.computeAdjoints();
        der = in.getDerivative();
        der2 = derivative(in);
        EXPECT_THAT(der2.real(), DoubleEq(der.real()));
        EXPECT_THAT(der2.imag(), DoubleEq(der.imag()));
        ret.d_real.imag(der.real());
        ret.d_imag.imag(der.imag());
        return ret;
    }
    tape_type tape;
};

// because we can't use C++14 auto lambdas, we need this macro for all functions below
#define XAD_MAKE_FUNCTOR(_name, _operation)                                                        \
    namespace                                                                                      \
    {                                                                                              \
    struct _name                                                                                   \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(T in) const                                                                   \
        {                                                                                          \
            using V = typename T::value_type;                                                      \
            V xxx = 1.0;                                                                           \
            XAD_UNUSED_VARIABLE(xxx);                                                              \
            return _operation;                                                                     \
        }                                                                                          \
    };                                                                                             \
    }

TYPED_TEST_SUITE(ComplexADTest, test_types);

// ------------ multiply ------------------
XAD_MAKE_FUNCTOR(MultiplyDoubleFunc, 2.0 * in)
TYPED_TEST(ComplexADTest, MultiplyDouble)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(MultiplyDoubleFunc(), z);
    y.compare(2.0, 0.0, 0.0, 2.0);
}

XAD_MAKE_FUNCTOR(MultiplyScalarFunc, V(2.0) * in)
TYPED_TEST(ComplexADTest, MultiplyScalar)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(MultiplyScalarFunc(), z);
    y.compare(2.0, 0.0, 0.0, 2.0);
}

XAD_MAKE_FUNCTOR(MultiplyScalarExprFunc, (1.0 * V(2.0)) * in)
TYPED_TEST(ComplexADTest, MultiplyScalarExpr)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(MultiplyScalarExprFunc(), z);
    y.compare(2.0, 0.0, 0.0, 2.0);
}

XAD_MAKE_FUNCTOR(MultiplyComplexFunc, in* in)
TYPED_TEST(ComplexADTest, MultiplyComplex)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(MultiplyComplexFunc(), z);
    y.compare(2.0, 4.0, -4.0, 2.0);
}

// ------------ add ------------------

XAD_MAKE_FUNCTOR(AddDoubleFunc, 2.0 + in);
TYPED_TEST(ComplexADTest, AddDouble)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(AddDoubleFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.0);
}

XAD_MAKE_FUNCTOR(AddScalarFunc, V(2.0) + in)
TYPED_TEST(ComplexADTest, AddScalar)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(AddScalarFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.0);
}

XAD_MAKE_FUNCTOR(AddScalarExprFunc, (1.0 * V(2.0)) + in)
TYPED_TEST(ComplexADTest, AddScalarExpr)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(AddScalarExprFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.0);
}

XAD_MAKE_FUNCTOR(AddComplexFunc, in + in)
TYPED_TEST(ComplexADTest, AddComplex)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(AddComplexFunc(), z);
    y.compare(2.0, 0.0, 0.0, 2.0);
}

// ------------ sub ------------------

XAD_MAKE_FUNCTOR(SubDoubleFunc, 2.0 - in)
TYPED_TEST(ComplexADTest, SubDouble)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(SubDoubleFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

XAD_MAKE_FUNCTOR(SubScalarFunc, V(2.0) - in)
TYPED_TEST(ComplexADTest, SubScalar)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(SubScalarFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

XAD_MAKE_FUNCTOR(SubScalarExprFunc, (1.0 * T(2.0)) - in)
TYPED_TEST(ComplexADTest, SubScalarExpr)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(SubScalarExprFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

XAD_MAKE_FUNCTOR(SubComplexFunc, T(2., 3.) - in)
TYPED_TEST(ComplexADTest, SubComplex)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(SubComplexFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

// ------------ div ------------------

XAD_MAKE_FUNCTOR(DivDoubleFunc, in / 2.0)
TYPED_TEST(ComplexADTest, DivDouble)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(DivDoubleFunc(), z);
    y.compare(0.5, 0.0, 0.0, 0.5);
}

XAD_MAKE_FUNCTOR(DivScalarFunc, in / V(2.))
TYPED_TEST(ComplexADTest, DivScalar)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(DivScalarFunc(), z);
    y.compare(0.5, 0.0, 0.0, 0.5);
}

XAD_MAKE_FUNCTOR(DivScalarExprFunc, in / (1.0 * V(2.0)))
TYPED_TEST(ComplexADTest, DivScalarExpr)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(DivScalarExprFunc(), z);
    y.compare(0.5, 0.0, 0.0, 0.5);
}

XAD_MAKE_FUNCTOR(DivComplexFunc, in / T(2., 3.))
TYPED_TEST(ComplexADTest, DivComplex)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(DivComplexFunc(), z);
    auto den = 2.0 * 2.0 + 3.0 * 3.0;

    y.compare(2.0 / den, -3.0 / den, 3.0 / den, 2.0 / den);
}

// ------------ unary minus -------------
XAD_MAKE_FUNCTOR(UnaryMinusFunc, -in)
TYPED_TEST(ComplexADTest, UnaryMinus)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(UnaryMinusFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

// ------------ unary plus ---------

XAD_MAKE_FUNCTOR(UnaryPlusFunc, +in)
TYPED_TEST(ComplexADTest, UnaryPlus)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(UnaryPlusFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.0);
}

// ----------- unary math functions -------------

XAD_MAKE_FUNCTOR(AbsFunc, abs(in))
TYPED_TEST(ComplexADTest, Abs)
{
    std::complex<TypeParam> z(4.0, 4.0);
    auto y = this->calcDerivatives(AbsFunc(), z);
    y.compare(0.707106781186547524401, 0.0, 0.707106781186547524401, 0.0);
}

XAD_MAKE_FUNCTOR(ArgFunc, arg(in))
TYPED_TEST(ComplexADTest, Arg)
{
    std::complex<TypeParam> z(4.0, 4.0);
    auto y = this->calcDerivatives(ArgFunc(), z);
    y.compare(-0.125, 0.0, 0.125, 0.0);
}

TYPED_TEST(ComplexADTest, ArgNegReal)
{
    std::complex<TypeParam> z(-4.0, 0.0);
    auto y = this->calcDerivatives(ArgFunc(), z);
    y.compare(0.0, 0.0, -0.25, 0.0);
}

XAD_MAKE_FUNCTOR(NormFunc, norm(in))
TYPED_TEST(ComplexADTest, Norm)
{
    std::complex<TypeParam> z(4.0, 4.0);
    auto y = this->calcDerivatives(NormFunc(), z);
    y.compare(8.0, 0.0, 8.0, 0.0);
}

XAD_MAKE_FUNCTOR(ConjFunc, conj(in))
TYPED_TEST(ComplexADTest, Conj)
{
    std::complex<TypeParam> z(4.0, 4.0);
    auto y = this->calcDerivatives(ConjFunc(), z);
    y.compare(1.0, 0.0, 0.0, -1.0);
}

XAD_MAKE_FUNCTOR(ProjFunc, proj(in))
TYPED_TEST(ComplexADTest, Proj)
{
    std::complex<TypeParam> z(4.0, 4.0);
    auto y = this->calcDerivatives(ProjFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.0);
}

XAD_MAKE_FUNCTOR(ExpFunc, exp(in))
TYPED_TEST(ComplexADTest, Exp)
{
    std::complex<TypeParam> z(1.0, 2.0);
    auto y = this->calcDerivatives(ExpFunc(), z);
    y.compare(std::exp(1.0) * std::cos(2.0), std::exp(1.0) * std::sin(2.0),
              -std::exp(1.0) * std::sin(2.0), std::exp(1.0) * std::cos(2.0));
}

XAD_MAKE_FUNCTOR(LogFunc, log(in))
TYPED_TEST(ComplexADTest, Log)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(LogFunc(), z);
    y.compare(0.5, -0.5, 0.5, 0.5);
}

TYPED_TEST(ComplexADTest, LogOfZeroImag)
{
    std::complex<TypeParam> z(-1.0, 0.0);
    auto y = this->calcDerivatives(LogFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

TYPED_TEST(ComplexADTest, LogOfNegZeroImag)
{
    std::complex<TypeParam> z(-1.0, -0.0);
    auto y = this->calcDerivatives(LogFunc(), z);
    y.compare(-1.0, 0.0, 0.0, -1.0);
}

XAD_MAKE_FUNCTOR(Log10Func, log10(in))
TYPED_TEST(ComplexADTest, Log10)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(Log10Func(), z);
    y.compare(0.5 / std::log(10), -0.5 / std::log(10), 0.5 / std::log(10), 0.5 / std::log(10));
}

XAD_MAKE_FUNCTOR(SqrtFunc, sqrt(in))
TYPED_TEST(ComplexADTest, Sqrt)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(SqrtFunc(), z);
    y.compare(0.38844349350750929, -0.16089856322639562, 0.16089856322639562, 0.38844349350750929);
}

XAD_MAKE_FUNCTOR(SinFunc, sin(in))
TYPED_TEST(ComplexADTest, Sin)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(SinFunc(), z);
    y.compare(0.83373002513114902, -0.98889770576286506, 0.98889770576286506, 0.83373002513114902);
}

XAD_MAKE_FUNCTOR(CosFunc, cos(in))
TYPED_TEST(ComplexADTest, Cos)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(CosFunc(), z);
    y.compare(-1.2984575814159773, -0.63496391478473613, 0.63496391478473613, -1.2984575814159773);
}

XAD_MAKE_FUNCTOR(TanFunc, tan(in))
TYPED_TEST(ComplexADTest, Tan)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(TanFunc(), z);
    y.compare(-0.10104031192114840, 0.58911793298483539, -0.58911793298483539,
              -0.10104031192114915);
}

XAD_MAKE_FUNCTOR(ASinFunc, asin(in))
TYPED_TEST(ComplexADTest, ASin)
{
    std::complex<TypeParam> z(0.5, 0.5);
    auto y = this->calcDerivatives(ASinFunc(), z);
    y.compare(0.92044206525992567, 0.21728689675164028, -0.21728689675164015, 0.92044206525992611);
}

XAD_MAKE_FUNCTOR(ACosFunc, acos(in))
TYPED_TEST(ComplexADTest, ACos)
{
    std::complex<TypeParam> z(0.5, 0.5);
    auto y = this->calcDerivatives(ACosFunc(), z);
    y.compare(-0.92044206525992567, -0.21728689675164028, 0.21728689675164015,
              -0.92044206525992611);
}

XAD_MAKE_FUNCTOR(ATanFunc, atan(in))
TYPED_TEST(ComplexADTest, ATan)
{
    std::complex<TypeParam> z(0.5, 0.5);
    auto y = this->calcDerivatives(ATanFunc(), z);
    y.compare(0.8, -0.4, 0.4, 0.8);
}

XAD_MAKE_FUNCTOR(SinhFunc, sinh(in))
TYPED_TEST(ComplexADTest, Sinh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(SinhFunc(), z);
    y.compare(0.83373002513114902, 0.98889770576286506, -0.98889770576286506, 0.83373002513114902);
}

XAD_MAKE_FUNCTOR(CoshFunc, cosh(in))
TYPED_TEST(ComplexADTest, Cosh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(CoshFunc(), z);
    y.compare(0.63496391478473613, 1.2984575814159773, -1.2984575814159773, 0.63496391478473613);
}

XAD_MAKE_FUNCTOR(TanhFunc, tanh(in))
TYPED_TEST(ComplexADTest, Tanh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(TanhFunc(), z);
    y.compare(-0.10104031192114840, -0.58911793298483539, 0.58911793298483539,
              -0.10104031192114915);
}

XAD_MAKE_FUNCTOR(ASinhFunc, asinh(in))
TYPED_TEST(ComplexADTest, ASinh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(ASinhFunc(), z);
    y.compare(0.56886448100578302, -0.35157758425414298, 0.35157758425414287, 0.56886448100578302);
}

XAD_MAKE_FUNCTOR(ACoshFunc, acosh(in))
TYPED_TEST(ComplexADTest, ACosh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(ACoshFunc(), z);
    y.compare(0.35157758425414287, -0.56886448100578302, 0.56886448100578302, 0.35157758425414298);
}

XAD_MAKE_FUNCTOR(ATanhFunc, atanh(in))
TYPED_TEST(ComplexADTest, ATanh)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(ATanhFunc(), z);
    y.compare(0.2, 0.4, -0.4, 0.2);
}

// ------------ binary math ---------

namespace
{
struct PolarFunc
{
    template <class T>
    T operator()(T in) const
    {
        using std::polar;
        auto x = polar(in.real(), in.imag());
        return T(x.real(), x.imag());
    }
};
}  // namespace

TYPED_TEST(ComplexADTest, PolarFirst)
{
    // we're hijacking our test framework here, passing in r and theta as complex
    // arguments to allow re-using this
    std::complex<TypeParam> z(1.232, 0.0);
    auto y = this->calcDerivatives(PolarFunc(), z);
    y.compare(1.0, 0.0, 0.0, 1.232);
}

TYPED_TEST(ComplexADTest, PolarSecond)
{
    // we're hijacking our test framework here, passing in r and theta as complex
    // arguments to allow re-using this
    std::complex<TypeParam> z(2.0, 1.57079632679489661923);
    auto y = this->calcDerivatives(PolarFunc(), z);
    y.compare(0.0, 1.0, -2.0, 0.0);
}

namespace
{
struct PolarFuncExprExpr
{
    template <class T>
    T operator()(T in) const
    {
        using std::polar;
        auto x = polar(in.real() * 1.0, in.imag() + 0.0);
        return T(x.real(), x.imag());
    }
};
}  // namespace

TYPED_TEST(ComplexADTest, PolarSecondExprExpr)
{
    // we're hijacking our test framework here, passing in r and theta as complex
    // arguments to allow re-using this
    std::complex<TypeParam> z(2.0, 1.57079632679489661923);
    auto y = this->calcDerivatives(PolarFuncExprExpr(), z);
    y.compare(0.0, 1.0, -2.0, 0.0);
}

XAD_MAKE_FUNCTOR(PowComplexFunc, pow(in, 2.0))
TYPED_TEST(ComplexADTest, PowComplexScalar)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(PowComplexFunc(), z);
    y.compare(2.0, 2.0, -2.0, 2.0);
}

XAD_MAKE_FUNCTOR(PowScalarComplexFunc, pow(2.0, in))
TYPED_TEST(ComplexADTest, PowScalarComplex)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(PowScalarComplexFunc(), z);
    y.compare(1.0663915513149342, 0.88578841432756017, -0.88578841432756017, 1.0663915513149342);
}

XAD_MAKE_FUNCTOR(PowComplexComplexFunc, pow(in, in))
TYPED_TEST(ComplexADTest, PowComplexComplex)
{
    std::complex<TypeParam> z(1.0, 1.0);
    auto y = this->calcDerivatives(PowComplexComplexFunc(), z);
    y.compare(-0.089533901029444973, 1.0011615503783176, -1.0011615503783178,
              -0.089533901029445084);
}
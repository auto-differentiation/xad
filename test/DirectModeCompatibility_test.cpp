/*******************************************************************************

   Tests that the direct mode types ARealDirect / FRealDirect can be used as
   drop-in replacements for AReal / FReal in generic code.

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

#include <XAD/Complex.hpp>
#include <XAD/StdCompatibility.hpp>
#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "TestHelpers.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace ::testing;

namespace
{

// every test runs for AReal / FReal and the direct mode types, which must behave the same
template <class T>
class DirectModeCompatibility : public ::testing::Test
{
};

typedef ::testing::Types<xad::AD, xad::FAD, xad::ADD, xad::FDD> test_types;

#define XAD_TEST_FUNCTOR(name, body)                                                               \
    struct name                                                                                    \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x) const                                                             \
        {                                                                                          \
            body                                                                                   \
        }                                                                                          \
    }

template <class T, class F>
void checkFunction(F f, double x0, double yref, double dref)
{
    std::pair<double, double> res = evalWithDerivative<T>(f, x0);
    EXPECT_THAT(res.first, DoubleNear(yref, 1e-12));
    EXPECT_THAT(res.second, DoubleNear(dref, 1e-12));
}

}  // namespace

TYPED_TEST_SUITE(DirectModeCompatibility, test_types);

// ------------------- mode interface and tape

namespace
{
template <class T>
struct mode_of;
template <>
struct mode_of<xad::AD>
{
    typedef xad::adj<double> type;
};
template <>
struct mode_of<xad::ADD>
{
    typedef xad::adjd<double> type;
};
template <>
struct mode_of<xad::FAD>
{
    typedef xad::fwd<double> type;
};
template <>
struct mode_of<xad::FDD>
{
    typedef xad::fwdd<double> type;
};
}  // namespace

TYPED_TEST(DirectModeCompatibility, ModeInterface)
{
    typedef typename mode_of<TypeParam>::type mode;
    static_assert(std::is_same<typename mode::active_type, TypeParam>::value,
                  "mode interface active type");
    static_assert(std::is_same<typename mode::passive_type, double>::value, "passive type");
    static_assert(std::is_same<typename TypeParam::value_type, double>::value, "value_type");
    static_assert(std::is_same<typename TypeParam::nested_type, double>::value, "nested_type");
    static_assert(std::is_same<typename TypeParam::derivative_type, double>::value,
                  "derivative_type");
}

// adjoint mode only

template <class T>
class DirectModeCompatibilityAdj : public ::testing::Test
{
};

typedef ::testing::Types<xad::AD, xad::ADD> adj_test_types;

TYPED_TEST_SUITE(DirectModeCompatibilityAdj, adj_test_types);

TYPED_TEST(DirectModeCompatibilityAdj, TapeRegistrationAndMembers)
{
    typedef TypeParam T;
    typedef typename xad::adjd<double>::tape_type tape_type;
    static_assert(std::is_same<typename T::tape_type, tape_type>::value, "tape type");
    tape_type tape;
    T x = 2.0;
    EXPECT_FALSE(x.shouldRecord());
    EXPECT_EQ(x.getSlot(), typename T::slot_type(T::INVALID_SLOT));
    tape.registerInput(x);
    EXPECT_TRUE(x.shouldRecord());
    EXPECT_NE(x.getSlot(), typename T::slot_type(T::INVALID_SLOT));
    EXPECT_EQ(x.getTape(), &tape);
    std::vector<T> v(2, T(3.0));
    tape.registerInputs(v);
    tape.newRecording();
    T y = x * v[0] * v[1];
    std::vector<T> out(1, T(x * 2.0));
    tape.registerOutput(y);
    tape.registerOutputs(out);
    y.setAdjoint(1.0);
    out[0].setAdjoint(1.0);
    tape.computeAdjoints();
    EXPECT_THAT(x.getAdjoint(), DoubleEq(9.0 + 2.0));
    EXPECT_THAT(xad::derivative(v[0]), DoubleEq(6.0));
    EXPECT_THAT(xad::derivative(v[1]), DoubleEq(6.0));
}

// ------------------- arithmetic, conversions, streams

namespace
{

XAD_TEST_FUNCTOR(PreIncrement, T y = x * 2.0; ++y; ++y; return y;);
XAD_TEST_FUNCTOR(PostIncrement, T y = x * 2.0; T z = y++; return T(y * z););
XAD_TEST_FUNCTOR(PreDecrement, T y = x * 2.0; --y; return y;);
XAD_TEST_FUNCTOR(PostDecrement, T y = x * 2.0; T z = y--; return T(y * z););
XAD_TEST_FUNCTOR(UnaryPlus, return T(+x * 3.0););
XAD_TEST_FUNCTOR(CompoundAssign, T y = x; y += 2; y -= 0.5; y *= x; y /= 4; y *= 2.0f;
                 y += T(x * 2.0); return y;);
XAD_TEST_FUNCTOR(IntLiterals, return T(2 * x + x * 3 - 1 + x / 2 - 4 / x););
XAD_TEST_FUNCTOR(AccumulateSort, std::vector<T> v; v.push_back(3.0 * x); v.push_back(x);
                 v.push_back(x * x + 1.0); std::sort(v.begin(), v.end());
                 return T(std::accumulate(v.begin(), v.end(), T(0.0)) + 10.0 * v[0]););
XAD_TEST_FUNCTOR(InnerProduct, std::vector<T> v(3, x); std::vector<T> w(3, T(2.0)); w[1] = x;
                 return T(std::inner_product(v.begin(), v.end(), w.begin(), T(0.0))););
XAD_TEST_FUNCTOR(Ternary, return T(x > 0.5 ? x : 0.5) + T(x < 0.1 ? 0.1 : x););
XAD_TEST_FUNCTOR(
    StdFunction, std::function<T(const T&)> g = [](const T& a) { return T(a * a); }; return g(x););

}  // namespace

TYPED_TEST(DirectModeCompatibility, IncrementDecrement)
{
    checkFunction<TypeParam>(PreIncrement(), 0.5, 3.0, 2.0);
    checkFunction<TypeParam>(PostIncrement(), 0.5, 2.0 * 1.0, 2.0 * 2.0 + 2.0 * 1.0);
    checkFunction<TypeParam>(PreDecrement(), 0.5, 0.0, 2.0);
    checkFunction<TypeParam>(PostDecrement(), 0.5, 0.0, 2.0 * 1.0 + 2.0 * 0.0);
}

TYPED_TEST(DirectModeCompatibility, ArithmeticWithLiterals)
{
    double x = 0.5;
    checkFunction<TypeParam>(UnaryPlus(), x, 3.0 * x, 3.0);
    checkFunction<TypeParam>(CompoundAssign(), x, ((x + 1.5) * x / 4.0 * 2.0) + 2.0 * x,
                             (2.0 * x + 1.5) / 4.0 * 2.0 + 2.0);
    checkFunction<TypeParam>(IntLiterals(), x, 2 * x + x * 3 - 1 + x / 2 - 4 / x,
                             2.0 + 3.0 + 0.5 + 4.0 / (x * x));
}

TYPED_TEST(DirectModeCompatibility, GenericAlgorithms)
{
    double x = 0.5;
    checkFunction<TypeParam>(AccumulateSort(), x, 3.0 * x + x + x * x + 1.0 + 10.0 * x,
                             3.0 + 1.0 + 2.0 * x + 10.0);
    checkFunction<TypeParam>(InnerProduct(), x, 2.0 * x + x * x + 2.0 * x, 2.0 + 2.0 * x + 2.0);
    checkFunction<TypeParam>(Ternary(), 0.7, 0.7 + 0.7, 2.0);
    checkFunction<TypeParam>(Ternary(), 0.3, 0.5 + 0.3, 1.0);
    checkFunction<TypeParam>(StdFunction(), x, x * x, 2.0 * x);
}

TYPED_TEST(DirectModeCompatibility, CompoundAssignmentReturnsDerivedType)
{
    typedef TypeParam T;
    T x = 1.0, y = 2.0;
    static_assert(std::is_same<decltype(x += y), T&>::value, "+= returns T&");
    static_assert(std::is_same<decltype(x -= y), T&>::value, "-= returns T&");
    static_assert(std::is_same<decltype(x *= y), T&>::value, "*= returns T&");
    static_assert(std::is_same<decltype(x /= y), T&>::value, "/= returns T&");
    static_assert(std::is_same<decltype(++x), T&>::value, "++ returns T&");
    static_assert(std::is_same<decltype(--x), T&>::value, "-- returns T&");
    (x += y) *= 2.0;
    EXPECT_THAT(xad::value(x), DoubleEq(6.0));
}

TYPED_TEST(DirectModeCompatibility, Conversions)
{
    typedef TypeParam T;
    static_assert(std::is_convertible<double, T>::value, "double converts implicitly");
    static_assert(std::is_convertible<int, T>::value, "int converts implicitly");
    static_assert(std::is_convertible<float, T>::value, "float converts implicitly");
    static_assert(!std::is_convertible<T, double>::value, "no implicit conversion to double");
    static_assert(!std::is_convertible<T, int>::value, "no implicit conversion to int");
    static_assert(!std::is_convertible<T, bool>::value, "no implicit conversion to bool");
    static_assert(!std::is_constructible<T, std::string>::value, "not constructible from string");
    static_assert(!std::is_convertible<std::string, T>::value, "string does not convert");
    static_assert(!std::is_convertible<const char*, T>::value, "const char* does not convert");
    static_assert(!std::is_convertible<std::vector<double>, T>::value, "vector does not convert");

    T x = 3.7, z = 0.0;
    EXPECT_TRUE(static_cast<bool>(x));
    EXPECT_FALSE(static_cast<bool>(z));
    EXPECT_TRUE(!z);
    bool b = false;
    if (x)
        b = true;
    EXPECT_TRUE(b);
#ifdef XAD_ALLOW_INT_CONVERSION
    EXPECT_EQ(static_cast<int>(x), 3);
    EXPECT_EQ(static_cast<long>(x), 3L);
    EXPECT_EQ(static_cast<long long>(x), 3LL);
    EXPECT_EQ(static_cast<unsigned>(x), 3u);
    EXPECT_EQ(static_cast<std::size_t>(x), std::size_t(3));
    EXPECT_EQ(static_cast<char>(x), char(3));
#endif
}

TYPED_TEST(DirectModeCompatibility, StreamsAndStrings)
{
    TypeParam x = 1.25;
    std::ostringstream os;
    os << x << " " << -x;
    EXPECT_EQ(os.str(), "1.25 -1.25");
    std::wostringstream wos;
    wos << x;
    EXPECT_TRUE(wos.str() == L"1.25");
    TypeParam y;
    std::istringstream is("2.5");
    is >> y;
    EXPECT_THAT(xad::value(y), DoubleEq(2.5));
    EXPECT_EQ(std::to_string(x), std::to_string(1.25));
}

TYPED_TEST(DirectModeCompatibility, ValueAndDerivativeAccess)
{
    typedef TypeParam T;
    // derivatives of adjoint types can only be accessed with an active tape
    std::unique_ptr<xad::Tape<double>> tape;
    if (xad::ExprTraits<T>::isReverse)
        tape.reset(new xad::Tape<double>());
    T x = 1.5;
    x.setDerivative(2.0);
    const T& cx = x;
    EXPECT_THAT(xad::value(cx), DoubleEq(1.5));
    EXPECT_THAT(xad::derivative(cx), DoubleEq(2.0));
    EXPECT_THAT(cx.getValue(), DoubleEq(1.5));
    EXPECT_THAT(cx.getDerivative(), DoubleEq(2.0));
    xad::value(x) = 2.5;
    xad::derivative(x) = 3.5;
    EXPECT_THAT(cx.value(), DoubleEq(2.5));
    EXPECT_THAT(cx.derivative(), DoubleEq(3.5));
}

TYPED_TEST(DirectModeCompatibility, Hashing)
{
    std::hash<TypeParam> h;
    TypeParam x = 42.5;
    EXPECT_EQ(h(x), std::hash<double>()(42.5));
}

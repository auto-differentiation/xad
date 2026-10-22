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

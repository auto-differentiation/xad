/*******************************************************************************

   Helpers for unit tests.

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

#pragma once

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399
#endif

inline void compareFinite(double xref, double xact, const std::string& msg = "")
{
    if (!(xad::isfinite)(xref))
        EXPECT_EQ(xad::fpclassify(xref), xad::fpclassify(xact)) << msg;
    else
        EXPECT_NEAR(xref, xact, 1e-10) << msg;
}

template <class F>
inline void mathTest_dbl(double x, double yref, F func)
{
    double y = func(x);
    EXPECT_DOUBLE_EQ(y, yref) << "dbl, yref";
}

template <class F>
inline void mathTest_adj(double x, double yref, double dref, F func)
{
    xad::Tape<double> s;
    xad::AD x1 = x;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = func(x1);
    s.registerOutput(y);
    EXPECT_EQ(2U, s.getNumVariables());
    y.setAdjoint(1.0);
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(yref, y.getValue()) << "adj, yref";
    compareFinite(dref, x1.getAdjoint(), "adj, dx");
}

template <class F>
inline void mathTest_fwd(double x, double yref, double dref, F func)
{
    xad::FAD x1(x, 1.0);
    xad::FAD y = func(x1);
    EXPECT_DOUBLE_EQ(yref, value(y)) << "fwd, y";
    compareFinite(dref, y.getDerivative(), "fwd, dx");
}

template <class F1, class F2>
inline void mathTest(double x, double yref, double dref, F1&& func, F2&& funcfwd)
{
    mathTest_adj(x, yref, dref, func);
    mathTest_fwd(x, yref, dref, funcfwd);
}

template <class F>
inline void mathTest_fwd_fwd(double x, double yref, double dref1, double dref2, F func)
{
    typedef xad::FReal<xad::FReal<double> > AD;
    AD x1 = x;
    derivative(value(x1)) = 1.0;
    value(derivative(x1)) = 1.0;
    AD y = func(x1);
    EXPECT_DOUBLE_EQ(yref, value(value(y))) << "fwd_fwd, y";
    compareFinite(dref1, derivative(value(y)), "fwd_fwd, dx1");
    compareFinite(dref2, derivative(derivative(y)), "fwd_fwd, dx2");
}

template <class F>
inline void mathTest_adj_adj(double x, double yref, double dref1, double dref2, F func)
{
    typedef xad::AReal<xad::AReal<double> > AD;
    xad::Tape<double> si;
    xad::Tape<xad::AReal<double> > so;

    AD x1 = x;
    so.registerInput(x1);
    so.newRecording();
    si.registerInput(value(x1));
    si.registerInput(derivative(x1));
    si.newRecording();
    AD y = func(x1);

    so.registerOutput(y);
    derivative(y) = 1.0;

    so.computeAdjoints();

    si.registerOutput(derivative(x1));
    derivative(derivative(x1)) = 1.0;
    si.computeAdjoints();

    EXPECT_DOUBLE_EQ(yref, value(value(y))) << "adj_adj, y";
    compareFinite(dref1, value(derivative(x1)), "adj_adj, dx1");
    compareFinite(dref2, derivative(value(x1)), "adj_adj, dx2");
}

template <class F>
inline void mathTest_fwd_adj(double x, double yref, double dref1, double dref2, F func)
{
    typedef xad::AReal<xad::FReal<double> > AD;
    xad::Tape<xad::FReal<double> > so;

    AD x1(x);
    derivative(value(x1)) = 1.0;
    so.registerInput(x1);
    so.newRecording();
    AD y = func(x1);
    so.registerOutput(y);
    value(derivative(y)) = 1.0;
    so.computeAdjoints();

    EXPECT_DOUBLE_EQ(yref, value(value(y))) << "fwd_adj, y";
    compareFinite(dref1, derivative(value(y)), "fwd_adj, dx1");
    compareFinite(dref2, derivative(derivative(x1)), "fwd_adj, dx2");
}

template <class F>
inline void mathTest_adj_fwd(double x, double yref, double dref1, double dref2, F func)
{
    typedef xad::FReal<xad::AReal<double> > AD;
    xad::Tape<double> si;

    AD x1(x);
    derivative(x1) = 1.0;
    si.registerInput(value(x1));
    si.registerInput(derivative(x1));
    si.newRecording();

    AD y = func(x1);
    auto yv = derivative(y);
    si.registerOutput(yv);
    derivative(yv) = 1.0;
    si.printStatus();
    si.computeAdjoints();

    EXPECT_DOUBLE_EQ(yref, value(value(y))) << "adj_fwd, y";
    compareFinite(dref1, value(yv), "adj_fwd, dx1");
    compareFinite(dref2, derivative(value(x1)), "adj_fwd, dx2");
}

template <class F>
inline void mathTest_all_aad(double x, double yref, double dref, double dref2, F func)
{
    mathTest_adj(x, yref, dref, func);
    mathTest_fwd(x, yref, dref, func);
    mathTest_fwd_fwd(x, yref, dref, dref2, func);
    mathTest_fwd_adj(x, yref, dref, dref2, func);
    mathTest_adj_fwd(x, yref, dref, dref2, func);
    mathTest_adj_adj(x, yref, dref, dref2, func);
}

template <class F>
inline void mathTest_all(double x, double yref, double dref, double dref2, F func)
{
    mathTest_dbl(x, yref, func);
    mathTest_all_aad(x, yref, dref, dref2, func);
}

template <class F>
inline void mathTest2_dbl(double x1, double x2, double yref, F func)
{
    double y = func(x1, x2);
    EXPECT_DOUBLE_EQ(y, yref);
}

template <class F>
inline void mathTest2_adj(double x1, double x2, double yref, double d1ref, double d2ref, F func)
{
    xad::Tape<double> s;
    xad::AD ax1 = x1;
    xad::AD ax2 = x2;
    s.registerInput(ax1);
    s.registerInput(ax2);
    s.newRecording();
    xad::AD y = func(ax1, ax2);
    s.registerOutput(y);
    EXPECT_EQ(3U, s.getNumVariables());
    y.setAdjoint(1.0);
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(yref, y.getValue()) << "adj, dy";
    compareFinite(d1ref, derivative(ax1), "adj, dx1");
    compareFinite(d2ref, derivative(ax2), "adj, dx2");
}

template <class F>
inline void mathTest2_fwd(double x1, double x2, double yref, double d1ref, double d2ref, F func)
{
    xad::FAD ax1 = x1;
    xad::FAD ax2 = x2;
    derivative(ax1) = 1.0;
    xad::FAD y1 = func(ax1, ax2);
    derivative(ax1) = 0.0;
    derivative(ax2) = 1.0;
    xad::FAD y2 = func(ax1, ax2);

    EXPECT_DOUBLE_EQ(yref, y1.getValue()) << "fwd, dy(1)";
    EXPECT_DOUBLE_EQ(yref, y2.getValue()) << "fwd, dy(2)";
    compareFinite(d1ref, derivative(y1), "fwd, dx1");
    compareFinite(d2ref, derivative(y2), "fwd, dx2");
}

template <class F>
inline void mathTest2_fwd_fwd(double x1, double x2, double yref, double d1ref, double d2ref,
                              double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    typedef xad::FReal<xad::FReal<double> > AD;
    AD ax1(x1);
    AD ax2(x2);
    derivative(value(ax1)) = 1.0;
    value(derivative(ax1)) = 1.0;
    AD y1 = func(ax1, ax2);

    derivative(value(ax1)) = 0.0;
    value(derivative(ax1)) = 0.0;
    derivative(value(ax2)) = 1.0;
    value(derivative(ax2)) = 1.0;
    AD y2 = func(ax1, ax2);

    derivative(value(ax1)) = 1.0;
    value(derivative(ax1)) = 0.0;
    derivative(value(ax2)) = 0.0;
    value(derivative(ax2)) = 1.0;
    AD y3 = func(ax1, ax2);

    derivative(value(ax1)) = 0.0;
    value(derivative(ax1)) = 1.0;
    derivative(value(ax2)) = 1.0;
    value(derivative(ax2)) = 0.0;
    AD y4 = func(ax1, ax2);

    EXPECT_DOUBLE_EQ(yref, value(value(y1))) << "fwd_fwd, dy(1)";
    EXPECT_DOUBLE_EQ(yref, value(value(y2))) << "fwd_fwd, dy(2)";
    EXPECT_DOUBLE_EQ(yref, value(value(y3))) << "fwd_fwd, dy(3)";
    EXPECT_DOUBLE_EQ(yref, value(value(y4))) << "fwd_fwd, dy(4)";
    compareFinite(d1ref, derivative(value(y1)), "fwd_fwd, dx1(1)");
    compareFinite(d1ref, derivative(value(y3)), "fwd_fwd, dx1(2)");
    compareFinite(d2ref, derivative(value(y2)), "fwd_fwd, dx2(1)");
    compareFinite(d2ref, derivative(value(y4)), "fwd_fwd, dx2(2)");

    compareFinite(d11ref, derivative(derivative(y1)), "fwd_fwd, d2x1");
    compareFinite(d22ref, derivative(derivative(y2)), "fwd_fwd, d2x2");
    compareFinite(d12ref, derivative(derivative(y3)), "fwd_fwd, dx1dx2");
    compareFinite(d21ref, derivative(derivative(y4)), "fwd_fwd, dx2dx1");
}

template <class F>
inline void mathTest2_fwd_adj(double x1, double x2, double yref, double d1ref, double d2ref,
                              double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    typedef xad::AReal<xad::FReal<double> > AD;
    xad::Tape<xad::FReal<double> > so;

    AD ax1(x1);
    AD ax2(x2);
    derivative(value(ax1)) = 1.0;
    so.registerInput(ax1);
    so.registerInput(ax2);
    so.newRecording();
    AD y1 = func(ax1, ax2);
    so.registerOutput(y1);
    value(derivative(y1)) = 1.0;
    // so.printStatus();
    so.computeAdjoints();

    double r_y1 = value(value(y1));
    double d1 = value(derivative(ax1));  // = derivative(value(y1))
    double d11 = derivative(derivative(ax1));
    double d12 = derivative(derivative(ax2));
    double d2 = value(derivative(ax2));

    /*
    std::cout << "y1: " <<  yref << " - " << r_y1 << "\n";
    std::cout << "d1: " <<  d1ref << " - " << d1 << "\n";
    std::cout << "d11: " << d11ref << " - " << d11 << "\n";
    std::cout << "d12: " << d12ref << " - " << d12 << "\n";
    std::cout << "d2: " << d2ref << " - " << d2 << "\n";
    std::cout << "d22: " << d22ref << "\n";
    std::cout << "d21: " << d21ref << "\n";
    std::cout << "vd1: " << value(derivative(ax1)) << "\n";
    std::cout << "vd2: " << value(derivative(ax2)) << "\n";
    std::cout << "dv2: " << derivative(value(ax2)) << "\n";
    */

    // we're missing d22 now, so we need to recompute the whole thing
    derivative(value(ax1)) = 0.0;
    derivative(value(ax2)) = 1.0;
    so.newRecording();
    AD y2 = func(ax1, ax2);
    so.registerOutput(y2);
    value(derivative(y2)) = 1.0;
    // so.printStatus();
    so.computeAdjoints();

    double r_y2 = value(value(y2));
    double d2_test = value(derivative(ax2));  // derivative(value(y2));
    double d21 = derivative(derivative(ax1));
    double d22 = derivative(derivative(ax2));
    double d1_test = value(derivative(ax1));

    EXPECT_DOUBLE_EQ(r_y1, yref) << "fwd_adj, y(1)";
    EXPECT_DOUBLE_EQ(r_y2, yref) << "fwd_adj, y(2)";
    compareFinite(d2, d2_test, "fwd_adj, dx2_t");
    compareFinite(d1, d1_test, "fwd_adj, dx1_t");
    compareFinite(d1ref, d1, "fwd_adj, dx1");
    compareFinite(d2ref, d2, "fwd_adj, dx2");
    compareFinite(d11ref, d11, "fwd_adj, d2x1");
    compareFinite(d12ref, d12, "fwd_adj, dx1dx2");
    compareFinite(d21ref, d21, "fwd_adj, dx2dx1");
    compareFinite(d22ref, d22, "fwd_adj, d2x2");
}

template <class F>
inline void mathTest2_adj_fwd(double x1, double x2, double yref, double d1ref, double d2ref,
                              double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    typedef xad::FReal<xad::AReal<double> > AD;
    xad::Tape<double> si;

    AD ax1(x1);
    AD ax2(x2);
    value(derivative(ax1)) = 1.0;
    si.registerInput(value(ax1));
    si.registerInput(derivative(ax1));
    si.registerInput(value(ax2));
    si.registerInput(derivative(ax2));
    si.newRecording();
    AD y1 = func(ax1, ax2);
    si.registerOutput(derivative(y1));
    si.registerOutput(value(y1));
    derivative(derivative(y1)) = 1.0;
    si.computeAdjoints();

    double r_y1 = value(value(y1));
    double d1 = derivative(derivative(ax1));
    double d11 = derivative(value(ax1));
    double d12 = derivative(value(ax2));
    double d2 = derivative(derivative(ax2));

    // si.printStatus();
    // std::cout << value(derivative(y1)) << "\n";
    // std::cout << derivative(derivative(ax1)) << "\n";
    // std::cout << derivative(derivative(ax2)) << "\n";
    // std::cout << derivative(derivative(ax1)) << "\n";
    // std::cout << derivative(derivative(ax2)) << "\n";

    value(derivative(ax1)) = 0.0;
    value(derivative(ax2)) = 1.0;
    si.newRecording();
    AD y2 = func(ax1, ax2);
    si.registerOutput(derivative(y2));
    si.registerOutput(value(y2));
    derivative(derivative(y2)) = 1.0;
    si.computeAdjoints();

    double r_y2 = value(value(y2));
    double d2_test = derivative(derivative(ax2));
    double d21 = derivative(value(ax1));
    double d22 = derivative(value(ax2));
    double d1_test = derivative(derivative(ax1));

    EXPECT_DOUBLE_EQ(r_y1, yref) << "adj_fwd, y(1)";
    EXPECT_DOUBLE_EQ(r_y2, yref) << "adj_fwd, y(2)";
    compareFinite(d1ref, d1, "adj_fwd, dx1");
    compareFinite(d2ref, d2, "adj_fwd, dx2");
    compareFinite(d11ref, d11, "adj_fwd, d2x1");
    compareFinite(d12ref, d12, "adj_fwd, dx1dx2");
    compareFinite(d21ref, d21, "adj_fwd, dx2dx1");
    compareFinite(d22ref, d22, "adj_fwd, d2x2");
    compareFinite(d1, d1_test, "adj_fwd, dx1_t");
    compareFinite(d2, d2_test, "adj_fwd, dx2_t");
}

template <class F>
inline void mathTest2_adj_adj(double x1, double x2, double yref, double d1ref, double d2ref,
                              double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    typedef xad::AReal<xad::AReal<double> > AD;
    xad::Tape<double> si;
    xad::Tape<xad::AReal<double> > so;

    AD ax1(x1);
    AD ax2(x2);
    so.registerInput(ax1);
    so.registerInput(ax2);
    so.newRecording();
    si.registerInput(value(ax1));
    si.registerInput(value(ax2));
    si.newRecording();
    AD y1 = func(ax1, ax2);
    so.registerOutput(y1);
    value(derivative(y1)) = 1.0;
    si.registerInput(derivative(ax1));
    si.registerInput(derivative(ax2));
    so.computeAdjoints();
    si.registerOutput(derivative(ax1));
    si.registerOutput(derivative(ax2));
    derivative(derivative(ax1)) = 1.0;
    si.computeAdjoints();

    double r_y1 = value(value(y1));
    double d1 = value(derivative(ax1));
    double d2 = value(derivative(ax2));
    double d11 = derivative(value(ax1));
    double d12 = derivative(value(ax2));

    // now the only unknown is d22, since d21 = d12 generally
    si.clearDerivatives();
    derivative(derivative(ax2)) = 1.0;
    si.computeAdjoints();

    double d22 = derivative(value(ax2));
    double d21 = derivative(value(ax1));

    EXPECT_DOUBLE_EQ(r_y1, yref) << "adj_adj, y";
    compareFinite(d1ref, d1, "adj_adj, dx1");
    compareFinite(d2ref, d2, "adj_adj, dx2");
    compareFinite(d11ref, d11, "adj_adj, d2x1");
    compareFinite(d12ref, d12, "adj_adj, dx1dx2");
    compareFinite(d21ref, d21, "adj_adj, dx2dx1");
    compareFinite(d22ref, d22, "adj_adj, d2x2");
}

template <class F>
inline void mathTest2_all_aad(double x1, double x2, double yref, double d1ref, double d2ref,
                              double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    mathTest2_adj(x1, x2, yref, d1ref, d2ref, func);
    mathTest2_fwd(x1, x2, yref, d1ref, d2ref, func);
    mathTest2_fwd_fwd(x1, x2, yref, d1ref, d2ref, d11ref, d12ref, d21ref, d22ref, func);
    mathTest2_fwd_adj(x1, x2, yref, d1ref, d2ref, d11ref, d12ref, d21ref, d22ref, func);
    mathTest2_adj_fwd(x1, x2, yref, d1ref, d2ref, d11ref, d12ref, d21ref, d22ref, func);
    mathTest2_adj_adj(x1, x2, yref, d1ref, d2ref, d11ref, d12ref, d21ref, d22ref, func);
}

template <class F>
inline void mathTest2_all(double x1, double x2, double yref, double d1ref, double d2ref,
                          double d11ref, double d12ref, double d21ref, double d22ref, F func)
{
    mathTest2_dbl(x1, x2, yref, func);
    mathTest2_all_aad(x1, x2, yref, d1ref, d2ref, d11ref, d12ref, d21ref, d22ref, func);
}

#define LOCAL_TEST_FUNCTOR1(name, val)                                                             \
    struct testFunctor_##name                                                                      \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x) const                                                             \
        {                                                                                          \
            using namespace std;                                                                   \
            return val;                                                                            \
        }                                                                                          \
    } name;

#define LOCAL_TEST_FUNCTOR2(name, val)                                                             \
    struct testFunctor_##name                                                                      \
    {                                                                                              \
        template <class T>                                                                         \
        T operator()(const T& x1, const T& x2) const                                               \
        {                                                                                          \
            using namespace std;                                                                   \
            return val;                                                                            \
        }                                                                                          \
    } name;

/*******************************************************************************

   Unit tests for derivatives of arithmetic and logical expressions.

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

#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>

using namespace ::testing;

TEST(Expressions, basic)
{
    xad::Tape<double> s;
    xad::AD a(1.0);
    xad::AD b(2.0);

    s.registerInput(a);
    s.registerInput(b);
    s.newRecording();

    xad::AD c = b;  // copy
    xad::AD ab = a + b;
    xad::AD abab = ab + ab + c;
    xad::AD big = a + b + a + b + a + a + a + 1.4;

    static_assert(xad::ExprTraits<decltype(c)>::numVariables == 1, "wrong number of variables");
    static_assert(xad::ExprTraits<decltype(a + b)>::numVariables == 2, "wrong number of variables");
    static_assert(xad::ExprTraits<decltype(ab + ab + c)>::numVariables == 3,
                  "wrong number of variables");
    static_assert(xad::ExprTraits<decltype(a + b + a + b + a + a + a + 1.4)>::numVariables == 7,
                  "wrong number of variables");

    EXPECT_DOUBLE_EQ(1.0, a.getValue());
    EXPECT_DOUBLE_EQ(2.0, b.getValue());

    EXPECT_DOUBLE_EQ(3.0, ab.getValue());
    EXPECT_DOUBLE_EQ(8.0, abab.getValue());

    EXPECT_DOUBLE_EQ(10.4, big.getValue());

    xad::AD res = big;  // construct from expression - puts it on tape
    
    s.registerOutput(res);
    derivative(res) = 1.0;
    s.computeAdjoints();
    EXPECT_DOUBLE_EQ(5.0, derivative(a));
    EXPECT_DOUBLE_EQ(2.0, derivative(b));
}

TEST(Expressions, basic_fwd)
{
    xad::FAD a = 1.0;
    xad::FAD b = 2.0;
    derivative(a) = 1.0;
    auto c = b;  // copy
    auto ab = a + b;
    auto abab = ab + ab + c;
    auto big = a + b + a + b + a + a + a + 1.4;

    EXPECT_DOUBLE_EQ(1.0, value(a));
    EXPECT_DOUBLE_EQ(1.0, derivative(a));
    EXPECT_DOUBLE_EQ(2.0, value(b));
    EXPECT_DOUBLE_EQ(0.0, derivative(b));
    EXPECT_DOUBLE_EQ(3.0, value(ab));
    EXPECT_DOUBLE_EQ(1.0, derivative(ab));
    EXPECT_DOUBLE_EQ(8.0, value(abab));
    EXPECT_DOUBLE_EQ(2.0, derivative(abab));
    EXPECT_DOUBLE_EQ(10.4, value(big));
    EXPECT_DOUBLE_EQ(5.0, derivative(big));

    xad::FAD res = big;  // construct from expression
    EXPECT_DOUBLE_EQ(5.0, derivative(res));
}

TEST(Expressions, basic_fwd_fwd)
{
    typedef xad::FReal<xad::FReal<double> > AD;

    AD x = 1.0;
    derivative(value(x)) = 1.0;
    value(derivative(x)) = 1.0;
    AD res = sin(x);

    EXPECT_DOUBLE_EQ(sin(1.0), value(value(res)));
    EXPECT_DOUBLE_EQ(cos(1.0), derivative(value(res)));
    EXPECT_DOUBLE_EQ(-sin(1.0), derivative(derivative(res)));
}

TEST(Expressions, basic_adj_adj)
{
    typedef xad::AReal<xad::AReal<double> > AD;

    xad::Tape<double> si;
    xad::Tape<xad::AReal<double> > so;

    AD x = 1.0;
    so.registerInput(x);
    si.registerInput(value(x));
    so.newRecording();
    si.newRecording();
    AD res = sin(x);
    so.registerOutput(res);
    
    value(derivative(res)) = 1.0;
    /*
    std::cout << "slot of x: " << x.getSlot() << "\n";
    std::cout << "slot of res: " << res.getSlot() << "\n";
    so.printStatus();
    std::cout << "------------\n";
  */
    so.computeAdjoints();
    /*
      so.printStatus();
      std::cout << "****------------\n";

      std::cout << "slot of x: " << value(x).getSlot() << "\n";
      std::cout << "slot of dx: " << derivative(x).getSlot() << "\n";
      std::cout << "slot of res: " << value(res).getSlot() << "\n";
      std::cout << "slot of dres: " << derivative(res).getSlot() << "\n";
      */

    // now we computed derivative(x) as an output, so we need to set its adjoint
    // to 1.0
    si.registerOutput(derivative(x));
    derivative(derivative(x)) = 1.0;

    // si.setDerivative(2, 1.0); // which variable is in slot 7??
    /*
      si.printStatus();
      std::cout << "------------\n";
      */
    si.computeAdjoints();
    /*
    si.printStatus();
    std::cout << "------------\n";
  */
    EXPECT_DOUBLE_EQ(sin(1.0), value(value(res)));
    EXPECT_DOUBLE_EQ(cos(1.0), value(derivative(x)));
    EXPECT_DOUBLE_EQ(-sin(1.0), derivative(value(x)));
}

TEST(Expressions, basic_fwd_adj)
{
    typedef xad::AReal<xad::FReal<double> > AD;

    xad::Tape<xad::FReal<double> > so;

    AD x = 1.0;
    derivative(value(x)) = 1.0;
    so.registerInput(x);
    so.newRecording();
    AD res = sin(x);
    so.registerOutput(res);
    value(derivative(res)) = 1.0;

    /*
    so.printStatus();
    std::cout << "------------\n";
    std::cout << res.getValue().getDerivative() << "\n";
    std::cout << res.getDerivative().getValue() << "\n";
    */
    so.computeAdjoints();
    /*
    std::cout << res.getValue().getDerivative() << "\n";
    std::cout << res.getDerivative().getValue() << "\n";
    so.printStatus();
    std::cout << "------------\n";
    */

    EXPECT_DOUBLE_EQ(sin(1.0), value(value(res)));
    EXPECT_DOUBLE_EQ(cos(1.0), derivative(value(res)));
    EXPECT_DOUBLE_EQ(-sin(1.0), derivative(derivative(x)));
}

TEST(Expressions, basic_adj_fwd)
{
    typedef xad::FReal<xad::AReal<double> > AD;

    xad::Tape<double> si;

    AD x(1.0);
    derivative(x) = 1.0;
    si.registerInput(value(x));
    si.newRecording();
    AD res = sin(x);
    si.registerOutput(derivative(res));
    // now we computed the derivative(res), so set its adjoint to one for reverse
    derivative(derivative(res)) = 1.0;

    /*
    si.printStatus();
    std::cout << "------------\n";
    std::cout << res.getValue().getDerivative() << "\n";
    std::cout << res.getDerivative().getValue() << "\n";
  */
    si.computeAdjoints();

    /*
    std::cout << res.getValue().getDerivative() << "\n";
    std::cout << res.getDerivative().getValue() << "\n";
    si.printStatus();
    std::cout << "------------\n";
  */

    EXPECT_DOUBLE_EQ(sin(1.0), value(value(res)));
    EXPECT_DOUBLE_EQ(cos(1.0), value(derivative(res)));
    EXPECT_DOUBLE_EQ(-sin(1.0), derivative(value(x)));
}

TEST(Expressions, wrapsAReal)
{
    xad::Tape<double> s;
    xad::AD x1 = 0.1;
    xad::AD x2 = 123.1;

    static_assert((std::is_same<decltype(x1 + x2),
                                xad::BinaryExpr<double, xad::add_op<double>, xad::ADVar<double>,
                                                xad::ADVar<double> > >::value),
                  "ad type not wrapped");
    static_assert(
        (std::is_same<
            decltype(x1 + x2 + x1 * x2),
            xad::BinaryExpr<double, xad::add_op<double>,
                            xad::BinaryExpr<double, xad::add_op<double>, xad::ADVar<double>,
                                            xad::ADVar<double> >,
                            xad::BinaryExpr<double, xad::prod_op<double>, xad::ADVar<double>,
                                            xad::ADVar<double> > > >::value),
        "ad type not wrapped");
    static_assert((std::is_same<decltype(max(x1, x1)),
                                xad::BinaryExpr<double, xad::max_op<double>, xad::ADVar<double>,
                                                xad::ADVar<double> > >::value),
                  "AD type not wrapped");
}

TEST(Expressions, canDeriveSimpleAdditions)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 + x2;
    s.registerOutput(y);
    derivative(y) = 1.0;
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue(), y.getValue());
    EXPECT_DOUBLE_EQ(1.0, derivative(x1));
    EXPECT_DOUBLE_EQ(1.0, derivative(x2));
}

TEST(Expressions, canDeriveSimpleAdditionsFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = x1 + x2;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = x1 + x2;

    EXPECT_DOUBLE_EQ(value(x1) + value(x2), value(y1));
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(1.0, derivative(y1));
    EXPECT_DOUBLE_EQ(1.0, derivative(y2));
}

TEST(Expressions, canDeriveSimpleScalarAdditions)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 + x2 + 4.12;
    s.registerOutput(y);
    derivative(y) = 1.0;
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue() + 4.12, y.getValue());
    EXPECT_DOUBLE_EQ(1.0, derivative(x1));
    EXPECT_DOUBLE_EQ(1.0, derivative(x2));
}

TEST(Expressions, canDeriveSimpleScalarAdditionsFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = x1 + x2 + 4.12;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = x1 + x2 + 4.12;

    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue() + 4.12, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), y2.getValue());
    EXPECT_DOUBLE_EQ(1.0, derivative(y1));
    EXPECT_DOUBLE_EQ(1.0, derivative(y2));
}

TEST(Expressions, canDeriveSimpleScalarIntAdditions)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 + x2 + 4;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue() + 4, y.getValue());
    EXPECT_DOUBLE_EQ(1.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(1.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleScalarIntAdditionsFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = x1 + x2 + 4;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = x1 + x2 + 4;

    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue() + 4, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), y2.getValue());
    EXPECT_DOUBLE_EQ(1.0, derivative(y1));
    EXPECT_DOUBLE_EQ(1.0, derivative(y2));
}

TEST(Expressions, canDeriveSimpleFactorAdditions)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (3.1 * x1 + 1.5 * (x2 + 3.2)) + 3.4;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(3.1 * x1.getValue() + 1.5 * (x2.getValue() + 3.2) + 3.4, y.getValue());
    EXPECT_DOUBLE_EQ(3.1, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(1.5, x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleFactorAdditionsFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = (3.1 * x1 + 1.5 * (x2 + 3.2)) + 3.4;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = (3.1 * x1 + 1.5 * (x2 + 3.2)) + 3.4;

    // s.printStatus();
    EXPECT_DOUBLE_EQ(3.1 * x1.getValue() + 1.5 * (x2.getValue() + 3.2) + 3.4, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(3.1, derivative(y1));
    EXPECT_DOUBLE_EQ(1.5, derivative(y2));
}

TEST(Expressions, canDeriveSimpleIntFactorAdditions)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (3.1 * x1 + 2 * (x2 + 3.2)) + 3.4;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(3.1 * x1.getValue() + 2 * (x2.getValue() + 3.2) + 3.4, y.getValue());
    EXPECT_DOUBLE_EQ(3.1, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(2.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleMultiplications)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 * x2;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(x1.getValue() * x2.getValue(), y.getValue());
    EXPECT_DOUBLE_EQ(x2.getValue(), x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1.getValue(), x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleMultiplicationsFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    x1.derivative() = 1.0;
    xad::FAD y1 = x1 * x2;
    x1.setDerivative(0.0);
    x2.setDerivative(1.0);
    xad::FAD y2 = x1 * x2;
    EXPECT_DOUBLE_EQ(x1.getValue() * x2.getValue(), y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(value(x2), derivative(y1));
    EXPECT_DOUBLE_EQ(value(x1), derivative(y2));
}

TEST(Expressions, canDeriveSimpleHigherPowers)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = 2.0 * x1 * x1 * x1 * x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(2.0 * x1.getValue() * x1.getValue() * x1.getValue() * x1.getValue(),
                     y.getValue());
    EXPECT_DOUBLE_EQ(2.0 * 4.0 * x1.getValue() * x1.getValue() * x1.getValue(), x1.getAdjoint());
}

TEST(Expressions, canDeriveSimpleHigherPowersFwd)
{
    xad::FAD x1(2.0, 1.0);
    xad::FAD y = 2.0 * x1 * x1 * x1 * x1;
    EXPECT_DOUBLE_EQ(2.0 * x1.getValue() * x1.getValue() * x1.getValue() * x1.getValue(),
                     y.getValue());
    EXPECT_DOUBLE_EQ(2.0 * 4.0 * x1.getValue() * x1.getValue() * x1.getValue(), y.getDerivative());
}

TEST(Expressions, canDeriveSimpleIntHigherPowers)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = 2 * x1 * x1 * x1 * x1;

    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(2 * x1.getValue() * x1.getValue() * x1.getValue() * x1.getValue(),
                     y.getValue());
    EXPECT_DOUBLE_EQ(2 * 4.0 * x1.getValue() * x1.getValue() * x1.getValue(), x1.getAdjoint());
}

TEST(Expressions, canDerive2StatementsAdd)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD v = 1.5 * x1 + x1 * x2;
    xad::AD y = v + x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(2.5 * x1.getValue() + x1.getValue() * x2.getValue(), y.getValue());
    EXPECT_DOUBLE_EQ(2.5 + x2.getValue(), x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1.getValue(), x2.getAdjoint());
}

TEST(Expressions, canDerive2StatementsAddFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD v1 = 1.5 * x1 + x1 * x2;
    xad::FAD y1 = v1 + x1;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD v2 = 1.5 * x1 + x1 * x2;
    xad::FAD y2 = v2 + x1;

    EXPECT_DOUBLE_EQ(2.5 * x1.getValue() + x1.getValue() * x2.getValue(), y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(2.5 + x2.getValue(), derivative(y1));
    EXPECT_DOUBLE_EQ(x1.getValue(), derivative(y2));
}

TEST(Expressions, canDerive2StatementsAddInt)
{
    xad::Tape<double> s;
    xad::AD x1 = 2;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD v = 2 * x1 + x1 * x2;
    xad::AD y = v + x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(3.0 * x1.getValue() + x1.getValue() * x2.getValue(), y.getValue());
    EXPECT_DOUBLE_EQ(3.0 + x2.getValue(), x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1.getValue(), x2.getAdjoint());
}

TEST(Expressions, canDerive2StatementsMul)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD v = 1.5 * x1 + 1.3 * (x1 * x2);
    xad::AD y = v * x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(
        1.5 * x1.getValue() * x1.getValue() + 1.3 * x1.getValue() * x1.getValue() * x2.getValue(),
        y.getValue());
    EXPECT_DOUBLE_EQ(3.0 * x1.getValue() + 2.0 * 1.3 * x1.getValue() * x2.getValue(),
                     x1.getAdjoint());
    EXPECT_DOUBLE_EQ(1.3 * x1.getValue() * x1.getValue(), x2.getAdjoint());
}

TEST(Expressions, canDerive2StatementsMulFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD v1 = 1.5 * x1 + 1.3 * (x1 * x2);
    xad::FAD y1 = v1 * x1;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD v2 = 1.5 * x1 + 1.3 * (x1 * x2);
    xad::FAD y2 = v2 * x1;

    EXPECT_DOUBLE_EQ(
        1.5 * x1.getValue() * x1.getValue() + 1.3 * x1.getValue() * x1.getValue() * x2.getValue(),
        y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));

    EXPECT_DOUBLE_EQ(3.0 * x1.getValue() + 2.0 * 1.3 * x1.getValue() * x2.getValue(),
                     derivative(y1));
    EXPECT_DOUBLE_EQ(1.3 * x1.getValue() * x1.getValue(), derivative(y2));
}

TEST(Expressions, canDerive2StatementsSqr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD v = 1.5 * x1 + x1 * x2;
    xad::AD y = v * v;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    // s.printStatus();
    // 2*v*inner
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    auto vd = 1.5 * x1d + x1d * x2d;
    EXPECT_DOUBLE_EQ(vd * vd, y.getValue());
    EXPECT_DOUBLE_EQ(2 * vd * (1.5 + x2d), x1.getAdjoint());
    EXPECT_DOUBLE_EQ(2 * vd * x1d, x2.getAdjoint());
}

TEST(Expressions, canDerive2StatementsSqrFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD v1 = 1.5 * x1 + x1 * x2;
    xad::FAD y1 = v1 * v1;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD v2 = 1.5 * x1 + x1 * x2;
    xad::FAD y2 = v2 * v2;

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    auto vd = 1.5 * x1d + x1d * x2d;
    EXPECT_DOUBLE_EQ(vd * vd, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(2 * vd * (1.5 + x2d), derivative(y1));
    EXPECT_DOUBLE_EQ(2 * vd * x1d, derivative(y2));
}

namespace
{
template <class ADT>
ADT complexAddMul(const ADT& x1, const ADT& x2, const ADT& x3)
{
    ADT z1 = 3.0 * x1 * x2 + 2.0 * x3 + x3 * x1;
    ADT z2 = x1 * z1;
    return z2;
}
}  // namespace

TEST(Expressions, canDeriveComplexAddMulExpression)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 3.0;
    xad::AD x3 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.registerInput(x3);
    s.newRecording();
    xad::AD z2 = complexAddMul(x1, x2, x3);
    s.registerOutput(z2);

    // --> z2 = 3*x2*x1^2 + 2*x3*x1 + x3*x1^2
    z2.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    auto x3d = x3.getValue();
    EXPECT_DOUBLE_EQ(3. * x2d * x1d * x1d + 2. * x3d * x1d + x3d * x1d * x1d, z2.getValue());
    // 2*3*x2*x1 + 2*x3 + 2*x3*x1
    EXPECT_DOUBLE_EQ(6. * x2d * x1d + 2. * x3d + 2. * x3d * x1d, x1.getAdjoint());
    // 3*x1^2
    EXPECT_DOUBLE_EQ(3 * x1d * x1d, x2.getAdjoint());
    // 2*x1 + x1^2
    EXPECT_DOUBLE_EQ(2 * x1d + x1d * x1d, x3.getAdjoint());
}

TEST(Expressions, canDeriveComplexAddMulExpressionFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 3.0;
    xad::FAD x3 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD z1 = complexAddMul(x1, x2, x3);
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD z2 = complexAddMul(x1, x2, x3);
    derivative(x2) = 0.0;
    derivative(x3) = 1.0;
    xad::FAD z3 = complexAddMul(x1, x2, x3);

    // --> z2 = 3*x2*x1^2 + 2*x3*x1 + x3*x1^2
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    auto x3d = x3.getValue();
    EXPECT_DOUBLE_EQ(3. * x2d * x1d * x1d + 2. * x3d * x1d + x3d * x1d * x1d, z1.getValue());
    EXPECT_DOUBLE_EQ(value(z1), value(z2));
    EXPECT_DOUBLE_EQ(value(z1), value(z3));

    // 2*3*x2*x1 + 2*x3 + 2*x3*x1
    EXPECT_DOUBLE_EQ(6. * x2d * x1d + 2. * x3d + 2. * x3d * x1d, derivative(z1));
    // 3*x1^2
    EXPECT_DOUBLE_EQ(3 * x1d * x1d, derivative(z2));
    // 2*x1 + x1^2
    EXPECT_DOUBLE_EQ(2 * x1d + x1d * x1d, derivative(z3));
}

TEST(Expressions, canDeriveSimpleDiv)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 / x2;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1.getValue() / x2.getValue(), y.getValue());
    EXPECT_DOUBLE_EQ(1. / x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-x1d / (x2d * x2d), x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleDivFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = x1 / x2;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = x1 / x2;

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1.getValue() / x2.getValue(), y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(1. / x2d, derivative(y1));
    EXPECT_DOUBLE_EQ(-x1d / (x2d * x2d), derivative(y2));
}

TEST(Expressions, canDeriveScalarDiv)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = 1.3 / x2 + x1 / 12.4;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(1.3 / x2d + x1d / 12.4, y.getValue());
    EXPECT_DOUBLE_EQ(1. / 12.4, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-1.3 / (x2d * x2d), x2.getAdjoint());
}

TEST(Expressions, canDeriveScalarDivFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = 1.3 / x2 + x1 / 12.4;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = 1.3 / x2 + x1 / 12.4;

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(1.3 / x2d + x1d / 12.4, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(1. / 12.4, y1.getDerivative());
    EXPECT_DOUBLE_EQ(-1.3 / (x2d * x2d), y2.getDerivative());
}

TEST(Expressions, canDeriveScalarIntDiv)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = 2 / x2 + x1 / 12;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(2 / x2d + x1d / 12, y.getValue());
    EXPECT_DOUBLE_EQ(1. / 12, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-2. / (x2d * x2d), x2.getAdjoint());
}

TEST(Expressions, canDeriveScalarDivExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = 1.3 / (1.3 * x2 + x1) + (x1 * x2) / 12.4;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(1.3 / (1.3 * x2d + x1d) + (x1d * x2d) / 12.4, y.getValue());
    // -1.3/(1.3*x2+x1)^2*1 + x2/12.4
    EXPECT_DOUBLE_EQ(-1.3 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) + x2d / 12.4, x1.getAdjoint());
    // -1.3/(1.3*x2+x1)^2*1.3 + x1/12.4
    EXPECT_DOUBLE_EQ(-1.3 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) * 1.3 + x1d / 12.4,
                     x2.getAdjoint());
}

TEST(Expressions, canDeriveScalarDivExprFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = 1.3 / (1.3 * x2 + x1) + (x1 * x2) / 12.4;
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = 1.3 / (1.3 * x2 + x1) + (x1 * x2) / 12.4;

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(1.3 / (1.3 * x2d + x1d) + (x1d * x2d) / 12.4, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    // -1.3/(1.3*x2+x1)^2*1 + x2/12.4
    EXPECT_DOUBLE_EQ(-1.3 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) + x2d / 12.4, y1.getDerivative());
    // -1.3/(1.3*x2+x1)^2*1.3 + x1/12.4
    EXPECT_DOUBLE_EQ(-1.3 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) * 1.3 + x1d / 12.4,
                     y2.getDerivative());
}

TEST(Expressions, canDeriveScalarDivIntExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = 2 / (1.3 * x2 + x1) + (x1 * x2) / 12;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(2 / (1.3 * x2d + x1d) + (x1d * x2d) / 12, y.getValue());
    // -1.3/(1.3*x2+x1)^2*1 + x2/12.4
    EXPECT_DOUBLE_EQ(-2 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) + x2d / 12, x1.getAdjoint());
    // -1.3/(1.3*x2+x1)^2*1.3 + x1/12.4
    EXPECT_DOUBLE_EQ(-2 / (1.3 * x2d + x1d) / (1.3 * x2d + x1d) * 1.3 + x1d / 12, x2.getAdjoint());
}

TEST(Expressions, canDeriveDivExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 / (1.3 * x2 + x1);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1d / (1.3 * x2d + x1d), y.getValue());
    // 130x2/(10x1+13x2)^2
    EXPECT_DOUBLE_EQ(130 * x2d / (10 * x1d + 13 * x2d) / (10 * x1d + 13 * x2d), x1.getAdjoint());
    // -130x1/(13x2+10*x1)^2
    EXPECT_DOUBLE_EQ(-130 * x1d / (13 * x2d + 10 * x1d) / (13 * x2d + 10 * x1d), x2.getAdjoint());
}

TEST(Expressions, canDeriveDivExprFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = x1 / (1.3 * x2 + x1);
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = x1 / (1.3 * x2 + x1);

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1d / (1.3 * x2d + x1d), y1.getValue());
    // 130x2/(10x1+13x2)^2
    EXPECT_DOUBLE_EQ(130 * x2d / (10 * x1d + 13 * x2d) / (10 * x1d + 13 * x2d), y1.getDerivative());
    // -130x1/(13x2+10*x1)^2
    EXPECT_DOUBLE_EQ(-130 * x1d / (13 * x2d + 10 * x1d) / (13 * x2d + 10 * x1d),
                     y2.getDerivative());
}

TEST(Expressions, canDeriveUnaryPlus)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (+x1) * (+x2);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1d * x2d, y.getValue());
    EXPECT_DOUBLE_EQ(x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveUnaryPlusExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = +(x1 + x1) * +(x2 * x1);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(2.0 * x1d * x2d * x1d, y.getValue());
    EXPECT_DOUBLE_EQ(4.0 * x1d * x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(2.0 * x1d * x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveUnaryPlusFullExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = +(x1 + x1 * 2.0);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    // auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(2.0 * x1d + x1d, y.getValue());
    EXPECT_DOUBLE_EQ(3.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(0.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveUnaryMinus)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (-x1) * (x2);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(-x1d * x2d, y.getValue());
    EXPECT_DOUBLE_EQ(-x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveUnaryMinusExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (x1 + x1) * -(x2 * x1);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(-2.0 * x1d * x2d * x1d, y.getValue());
    EXPECT_DOUBLE_EQ(-4.0 * x1d * x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-2.0 * x1d * x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveUnaryMinusExprFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = (x1 + x1) * -(x2 * x1);
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = (x1 + x1) * -(x2 * x1);

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(-2.0 * x1d * x2d * x1d, y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(-4.0 * x1d * x2d, y1.getDerivative());
    EXPECT_DOUBLE_EQ(-2.0 * x1d * x1d, y2.getDerivative());
}

TEST(Expressions, canDeriveUnaryMinusFullExpr)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = -(x1 + x1 * 2.0);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    // auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(-3.0 * x1d, y.getValue());
    EXPECT_DOUBLE_EQ(-3.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(0.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveScalarSubtract)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (x1 * x2 - 1.2) + (2.1 - (x1 + 1.0));
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x2d * x1d - 1.2 + (2.1 - (x1d + 1.0)), y.getValue());
    EXPECT_DOUBLE_EQ(x2d - 1.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveScalarSubtractFwd)
{
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = (x1 * x2 - 1.2) + (2.1 - (x1 + 1.0));
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = (x1 * x2 - 1.2) + (2.1 - (x1 + 1.0));

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x2d * x1d - 1.2 + (2.1 - (x1d + 1.0)), y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(x2d - 1.0, y1.getDerivative());
    EXPECT_DOUBLE_EQ(x1d, y2.getDerivative());
}

TEST(Expressions, canDeriveScalarIntSubtract)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (x1 * x2 - 2) + (2.1 - (x1 + 1.0));
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x2d * x1d - 2 + (2.1 - (x1d + 1.0)), y.getValue());
    EXPECT_DOUBLE_EQ(x2d - 1.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveSimpleSubtract)
{
    // AD - AD
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 - x2;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1d - x2d, y.getValue());
    EXPECT_DOUBLE_EQ(1.0, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-1.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveAdExprSubtract)
{
    // AD - Expr
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = x1 - (x2 * 2.0 + 1.2 * x1);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ(x1d - (x2d * 2.0 + 1.2 * x1d), y.getValue());
    EXPECT_DOUBLE_EQ(-0.2, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(-2.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveExprAdSubtract)
{
    // Expr - AD
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (x2 * 2.0 + 1.2 * x1) - x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ((x2d * 2.0 + 1.2 * x1d) - x1d, y.getValue());
    EXPECT_DOUBLE_EQ(0.2, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(2.0, x2.getAdjoint());
}

TEST(Expressions, canDeriveExprExprSubtract)
{
    // Expr - Expr
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    xad::AD y = (x2 * 2.0 + 1.2 * x1) - (x1 * x2);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ((x2d * 2.0 + 1.2 * x1d) - (x1d * x2d), y.getValue());
    EXPECT_DOUBLE_EQ(1.2 - x2d, x1.getAdjoint());
    EXPECT_DOUBLE_EQ(2.0 - x1d, x2.getAdjoint());
}

TEST(Expressions, canDeriveExprExprSubtractFwd)
{
    // Expr - Expr
    xad::FAD x1 = 2.0;
    xad::FAD x2 = 5.0;
    derivative(x1) = 1.0;
    xad::FAD y1 = (x2 * 2.0 + 1.2 * x1) - (x1 * x2);
    derivative(x1) = 0.0;
    derivative(x2) = 1.0;
    xad::FAD y2 = (x2 * 2.0 + 1.2 * x1) - (x1 * x2);

    auto x1d = x1.getValue();
    auto x2d = x2.getValue();
    EXPECT_DOUBLE_EQ((x2d * 2.0 + 1.2 * x1d) - (x1d * x2d), y1.getValue());
    EXPECT_DOUBLE_EQ(value(y1), value(y2));
    EXPECT_DOUBLE_EQ(1.2 - x2d, y1.getDerivative());
    EXPECT_DOUBLE_EQ(2.0 - x1d, y2.getDerivative());
}

TEST(Expressions, canDeriveADScalarSubtract)
{
    // AD - scalar
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = x1 - 1.0;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(x1d - 1.0, y.getValue());
    EXPECT_DOUBLE_EQ(1.0, x1.getAdjoint());
}

TEST(Expressions, canDeriveADScalarIntSubtract)
{
    // AD - scalar
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = x1 - 1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(x1d - 1, y.getValue());
    EXPECT_DOUBLE_EQ(1.0, x1.getAdjoint());
}

TEST(Expressions, canDeriveScalarADSubtract)
{
    // scalar - AD
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = 2.0 - x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(2.0 - x1d, y.getValue());
    EXPECT_DOUBLE_EQ(-1.0, x1.getAdjoint());
}

TEST(Expressions, canDeriveScalarIntADSubtract)
{
    // scalar - AD
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = 2 - x1;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(2 - x1d, y.getValue());
    EXPECT_DOUBLE_EQ(-1.0, x1.getAdjoint());
}

TEST(Expressions, canDeriveExprScalarSubtract)
{
    // Expr - scalar
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = (x1 * x1 * 3.0) - 1.0;
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(x1d * x1d * 3.0 - 1.0, y.getValue());
    EXPECT_DOUBLE_EQ(6.0 * x1d, x1.getAdjoint());
}

TEST(Expressions, canDeriveScalarExprSubtract)
{
    // scalar - Expr
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    s.registerInput(x1);
    s.newRecording();
    xad::AD y = 2.0 - (x1 * x1 * 3.0);
    s.registerOutput(y);
    y.setAdjoint(1.0);
    s.computeAdjoints();
    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(2.0 - x1d * x1d * 3.0, y.getValue());
    EXPECT_DOUBLE_EQ(-6.0 * x1d, x1.getAdjoint());
}

TEST(Expressions, canDeriveScalarExprSubtractFwd)
{
    // scalar - Expr
    xad::FAD x1 = 2.0;
    derivative(x1) = 1.0;
    xad::FAD y = 2.0 - (x1 * x1 * 3.0);

    auto x1d = x1.getValue();
    EXPECT_DOUBLE_EQ(2.0 - x1d * x1d * 3.0, y.getValue());
    EXPECT_DOUBLE_EQ(-6.0 * x1d, y.getDerivative());
}

TEST(Expressions, canScalarCompare)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.1;
    x1 *= 1.0;

    EXPECT_TRUE(x1 == 2.1);
    EXPECT_TRUE(x1 != 2.0);
    EXPECT_TRUE(x1 < 2.2);
    EXPECT_TRUE(x1 <= 2.1);
    EXPECT_TRUE(x1 > 1.1);
    EXPECT_TRUE(x1 >= 2.1);

    EXPECT_TRUE(2.1 == x1);
    EXPECT_TRUE(2.0 != x1);
    EXPECT_TRUE(2.2 > x1);
    EXPECT_TRUE(2.1 >= x1);
    EXPECT_TRUE(1.1 < x1);
    EXPECT_TRUE(2.1 <= x1);

    static_assert((std::is_base_of<xad::Expression<double, xad::AD>, xad::AD>::value),
                  "should be same type");
}

TEST(Expressions, canScalarCompareFwd)
{
    xad::FAD x1 = 2.1;

    EXPECT_TRUE(x1 == 2.1);
    EXPECT_TRUE(x1 != 2.0);
    EXPECT_TRUE(x1 < 2.2);
    EXPECT_TRUE(x1 <= 2.1);
    EXPECT_TRUE(x1 > 1.1);
    EXPECT_TRUE(x1 >= 2.1);

    EXPECT_TRUE(2.1 == x1);
    EXPECT_TRUE(2.0 != x1);
    EXPECT_TRUE(2.2 > x1);
    EXPECT_TRUE(2.1 >= x1);
    EXPECT_TRUE(1.1 < x1);
    EXPECT_TRUE(2.1 <= x1);

    static_assert((std::is_base_of<xad::Expression<double, xad::FAD>, xad::FAD>::value),
                  "should be same type");
}

TEST(Expressions, canScalarIntCompare)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;

    EXPECT_TRUE(x1 == 2);
    EXPECT_TRUE(x1 != 3);
    EXPECT_TRUE(x1 < 3);
    EXPECT_TRUE(x1 <= 2);
    EXPECT_TRUE(x1 > 1);
    EXPECT_TRUE(x1 >= 2);

    EXPECT_TRUE(2 == x1);
    EXPECT_TRUE(3 != x1);
    EXPECT_TRUE(3 > x1);
    EXPECT_TRUE(2 >= x1);
    EXPECT_TRUE(1 < x1);
    EXPECT_TRUE(2 <= x1);
}

TEST(Expressions, canScalarIntCompareFwd)
{
    xad::FAD x1 = 2.0;

    EXPECT_TRUE(x1 == 2);
    EXPECT_TRUE(x1 != 3);
    EXPECT_TRUE(x1 < 3);
    EXPECT_TRUE(x1 <= 2);
    EXPECT_TRUE(x1 > 1);
    EXPECT_TRUE(x1 >= 2);

    EXPECT_TRUE(2 == x1);
    EXPECT_TRUE(3 != x1);
    EXPECT_TRUE(3 > x1);
    EXPECT_TRUE(2 >= x1);
    EXPECT_TRUE(1 < x1);
    EXPECT_TRUE(2 <= x1);
}

TEST(Expressions, canExprCompare)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.1;
    auto x2 = (0.5 * x1);

    EXPECT_TRUE(x1 == x1);
    EXPECT_TRUE(x1 != x2);
    EXPECT_TRUE(x2 < x1);
    EXPECT_TRUE(x2 <= x1);
    EXPECT_TRUE(x2 <= x2);
    EXPECT_TRUE(x1 > x2);
    EXPECT_TRUE(x1 >= x2);
    EXPECT_TRUE(x1 >= x1);
}

TEST(Expressions, canExprCompareFwd)
{
    xad::FAD x1 = 2.1;
    auto x2 = (0.5 * x1);

    EXPECT_TRUE(x1 == x1);
    EXPECT_TRUE(x1 != x2);
    EXPECT_TRUE(x2 < x1);
    EXPECT_TRUE(x2 <= x1);
    EXPECT_TRUE(x2 <= x2);
    EXPECT_TRUE(x1 > x2);
    EXPECT_TRUE(x1 >= x2);
    EXPECT_TRUE(x1 >= x1);
}

TEST(Expressions, canDerivePreIncrementFwd)
{
    xad::FAD x1 = 2.1;
    derivative(x1) = 1.0;
    xad::FAD x2 = x1;
    ++x2;

    EXPECT_DOUBLE_EQ(value(x2), 3.1);
    EXPECT_DOUBLE_EQ(derivative(x2), 1.0);
}

TEST(Expressions, canDerivePostIncrementFwd)
{
    xad::FAD x1 = 2.1;
    derivative(x1) = 1.0;
    xad::FAD x2 = x1;
    xad::FAD x3 = x2++;

    EXPECT_DOUBLE_EQ(value(x2), 3.1);
    EXPECT_DOUBLE_EQ(derivative(x2), 1.0);
    EXPECT_DOUBLE_EQ(value(x3), 2.1);
}

TEST(Expressions, canDerivePreIncrementAAD)
{
    xad::Tape<double> tape;
    xad::AD x1 = 2.1;
    tape.registerInput(x1);
    tape.newRecording();
    xad::AD x2 = x1;
    ++x2;
    tape.registerOutput(x2);
    derivative(x2) = 1.0;
    tape.computeAdjoints();

    EXPECT_DOUBLE_EQ(value(x2), 3.1);
    EXPECT_DOUBLE_EQ(derivative(x1), 1.0);
}

TEST(Expressions, canDerivePostIncrementAAD)
{
    xad::Tape<double> tape;
    xad::AD x1 = 2.1;
    tape.registerInput(x1);
    tape.newRecording();
    xad::AD x2 = x1;
    xad::AD x3 = x2++;
    tape.registerOutput(x3);
    derivative(x3) = 1.0;
    tape.computeAdjoints();

    EXPECT_DOUBLE_EQ(value(x2), 3.1);
    EXPECT_DOUBLE_EQ(derivative(x1), 1.0);
    EXPECT_DOUBLE_EQ(value(x3), 2.1);
}

TEST(Expressions, canDerivePostDecrementFwd)
{
    xad::FAD x1 = 2.1;
    derivative(x1) = 1.0;
    xad::FAD x2 = x1;
    xad::FAD x3 = x2--;

    EXPECT_DOUBLE_EQ(value(x2), 1.1);
    EXPECT_DOUBLE_EQ(derivative(x2), 1.0);
    EXPECT_DOUBLE_EQ(value(x3), 2.1);
}

TEST(Expressions, canDerivePreDecrementAAD)
{
    xad::Tape<double> tape;
    xad::AD x1 = 2.1;
    tape.registerInput(x1);
    tape.newRecording();
    xad::AD x2 = x1;
    --x2;
    tape.registerOutput(x2);
    derivative(x2) = 1.0;
    tape.computeAdjoints();

    EXPECT_DOUBLE_EQ(value(x2), 1.1);
    EXPECT_DOUBLE_EQ(derivative(x1), 1.0);
}

TEST(Expressions, canDerivePostDecrementAAD)
{
    xad::Tape<double> tape;
    xad::AD x1 = 2.1;
    tape.registerInput(x1);
    tape.newRecording();
    xad::AD x2 = x1;
    xad::AD x3 = x2--;
    tape.registerOutput(x3);
    derivative(x3) = 1.0;
    tape.computeAdjoints();

    EXPECT_DOUBLE_EQ(value(x2), 1.1);
    EXPECT_DOUBLE_EQ(derivative(x1), 1.0);
    EXPECT_DOUBLE_EQ(value(x3), 2.1);
}

TEST(Expressions, canDeriveLongExpressionFromLambdaReturnAdjoint)
{
    std::vector<xad::AD> tmp(3, 1.0);
    tmp[2] = 0.0;
    auto lbd = [&](xad::AD in) -> xad::AD {
        // we make this function really long with loads of temporaries in the expression
        // to trigger problems with overwriting temp refs
        return tmp[0] * (xad::AD(in * in) * xad::AD(tmp[1]) + exp(in)) +
               0.0 * (xad::AD(in * in * in) * xad::AD(tmp[1]) + exp(sin(in * in)));
    };
    // we capture this lambda in a std::function object to make sure that it doesn't get inlined
    // and optimised away (std::function is a virtual polymorphic class)
    using ret_t = std::decay<decltype(lbd(tmp[0]))>::type;
    std::function<ret_t(xad::AD)> func = lbd;

    xad::Tape<double> tape;
    std::vector<xad::AD> xv(10, 2.1);
    std::vector<xad::AD> yv(10);
    tape.registerInputs(xv);
    tape.newRecording();
    std::transform(xv.begin(), xv.end(), yv.begin(), func);
    xad::AD y = std::accumulate(yv.begin(), yv.end(), xad::AD(0.0));
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();

    EXPECT_DOUBLE_EQ(value(y), 10 * (std::exp(2.1) + 2.1 * 2.1));
    EXPECT_DOUBLE_EQ(derivative(xv[0]), std::exp(2.1) + 2.0 * 2.1);
    EXPECT_DOUBLE_EQ(derivative(xv[1]), std::exp(2.1) + 2.0 * 2.1);
}

TEST(Expressions, canDeriveLongExpressionFromLambdaReturnForward)
{
    std::vector<xad::FAD> tmp(3, 1.0);
    tmp[2] = 0.0;
    auto lbd = [&](xad::FAD in) -> xad::FAD {
        // we make this function really long with loads of temporaries in the
        // expression to trigger problems with overwriting temp refs
        return tmp[0] * (xad::FAD(in * in) * xad::FAD(tmp[1]) + exp(in)) +
               0.0 * (xad::FAD(in * in * in) * xad::FAD(tmp[1]) + exp(sin(in * in)));
    };
    // we capture this lambda in a std::function object to make sure that it
    // doesn't get inlined and optimised away (std::function is a virtual
    // polymorphic class)
    using ret_t = std::decay<decltype(lbd(tmp[0]))>::type;
    std::function<ret_t(xad::FAD)> func = lbd;

    std::vector<xad::FAD> xv(10, 2.1);
    std::vector<xad::FAD> yv(10);
    derivative(xv[0]) = 1.0;
    derivative(xv[1]) = 1.0;
    std::transform(xv.begin(), xv.end(), yv.begin(), func);

    EXPECT_DOUBLE_EQ(value(yv[0]), (std::exp(2.1) + 2.1 * 2.1));
    EXPECT_DOUBLE_EQ(value(yv[1]), (std::exp(2.1) + 2.1 * 2.1));
    EXPECT_DOUBLE_EQ(derivative(yv[0]), std::exp(2.1) + 2.0 * 2.1);
    EXPECT_DOUBLE_EQ(derivative(yv[1]), std::exp(2.1) + 2.0 * 2.1);
}

// this is an insanely long expression taken from QuantLib's COSHestonEngine,
// where a bug was detected with XAD in Ubuntu Groovy
template <class T>
struct TestHeston
{
    T kappa_ = 0.4;
    T rho_ = 0.8;
    T theta_ = 1.2;
    T sigma_ = 0.8;
    T v0_ = 12.0;

    T c4(T t) const
    {
        const T sigma2 = sigma_ * sigma_;
        const T sigma3 = sigma2 * sigma_;
        const T sigma4 = sigma2 * sigma2;
        const T kappa2 = kappa_ * kappa_;
        const T kappa3 = kappa2 * kappa_;
        const T kappa4 = kappa2 * kappa2;
        const T kappa5 = kappa2 * kappa3;
        const T kappa6 = kappa3 * kappa3;
        const T kappa7 = kappa4 * kappa3;
        const T rho2 = rho_ * rho_;
        const T rho3 = rho2 * rho_;
        const T t2 = t * t;
        const T t3 = t2 * t;

        return (sigma2 *
                (3 * sigma4 * (theta_ - 4 * v0_) +
                 3 * exp(4 * kappa_ * t) *
                     ((-93 * sigma4 + 64 * kappa5 * (t + 4 * rho2 * t) +
                       4 * kappa_ * sigma3 * (176 * rho_ + 5 * sigma_ * t) -
                       32 * kappa2 * sigma2 * (11 + 50 * rho2 + 5 * rho_ * sigma_ * t) +
                       32 * kappa3 * sigma_ *
                           (3 * sigma_ * t + 4 * rho_ * (10 + 8 * rho2 + 3 * rho_ * sigma_ * t)) -
                       32 * kappa4 * (5 + 4 * rho_ * (6 * rho_ + (3 + 2 * rho2) * sigma_ * t))) *
                          theta_ +
                      4 * (4 * kappa2 - 4 * kappa_ * rho_ * sigma_ + sigma2) *
                          (4 * kappa2 * (1 + 4 * rho2) - 20 * kappa_ * rho_ * sigma_ + 5 * sigma2) *
                          v0_) +
                 24 * exp(kappa_ * t) * sigma2 *
                     (-2 * kappa2 * (-1 + rho_ * sigma_ * t) * (theta_ - 3 * v0_) +
                      sigma2 * (theta_ - 2 * v0_) +
                      kappa_ * sigma_ *
                          (-4 * rho_ * theta_ + sigma_ * t * theta_ + 10 * rho_ * v0_ -
                           3 * sigma_ * t * v0_)) +
                 12 * exp(2 * kappa_ * t) *
                     (sigma4 * (7 * theta_ - 4 * v0_) +
                      8 * kappa4 * (1 + 2 * rho_ * sigma_ * t * (-2 + rho_ * sigma_ * t)) *
                          (theta_ - 2 * v0_) +
                      2 * kappa_ * sigma3 *
                          (-24 * rho_ * theta_ + 5 * sigma_ * t * theta_ + 20 * rho_ * v0_ -
                           6 * sigma_ * t * v0_) +
                      4 * kappa2 * sigma2 *
                          ((6 + 20 * rho2 - 14 * rho_ * sigma_ * t + sigma2 * t2) * theta_ -
                           2 * (3 + 12 * rho2 - 10 * rho_ * sigma_ * t + sigma2 * t2) * v0_) +
                      8 * kappa3 * sigma_ *
                          ((3 * sigma_ * t +
                            2 * rho_ * (-4 + sigma_ * t * (4 * rho_ - sigma_ * t))) *
                               theta_ +
                           2 *
                               (-3 * sigma_ * t +
                                2 * rho_ * (3 + sigma_ * t * (-3 * rho_ + sigma_ * t))) *
                               v0_)) -
                 8 * exp(3 * kappa_ * t) *
                     (16 * kappa6 * rho2 * t2 * (-3 + rho_ * sigma_ * t) * (theta_ - v0_) -
                      3 * sigma4 * (7 * theta_ + 2 * v0_) +
                      2 * kappa3 * sigma_ *
                          ((192 * (rho_ + rho3) - 6 * (9 + 40 * rho2) * sigma_ * t +
                            42 * rho_ * sigma2 * t2 - sigma3 * t3) *
                               theta_ +
                           (-48 * rho3 + 18 * (1 + 4 * rho2) * sigma_ * t -
                            24 * rho_ * sigma2 * t2 + sigma3 * t3) *
                               v0_) +
                      12 * kappa4 *
                          ((-4 - 24 * rho2 + 8 * rho_ * (4 + 3 * rho2) * sigma_ * t -
                            (3 + 14 * rho2) * sigma2 * t2 + rho_ * sigma3 * t3) *
                               theta_ +
                           (8 * rho2 - 8 * rho_ * (2 + rho2) * sigma_ * t +
                            (3 + 8 * rho2) * sigma2 * t2 - rho_ * sigma3 * t3) *
                               v0_) -
                      6 * kappa2 * sigma2 *
                          ((15 + 80 * rho2 - 35 * rho_ * sigma_ * t + 2 * sigma2 * t2) * theta_ +
                           (3 + sigma_ * t * (7 * rho_ - sigma_ * t)) * v0_) +
                      24 * kappa5 * t *
                          ((-2 + rho_ * (4 * sigma_ * t +
                                         rho_ * (-8 + sigma_ * t * (4 * rho_ - sigma_ * t)))) *
                               theta_ +
                           (2 + rho_ * (-4 * sigma_ * t +
                                        rho_ * (4 + sigma_ * t * (-2 * rho_ + sigma_ * t)))) *
                               v0_) +
                      3 * kappa_ * sigma3 *
                          (sigma_ * t * (-9 * theta_ + v0_) + 10 * rho_ * (6 * theta_ + v0_))))) /
               (64. * exp(4 * kappa_ * t) * kappa7);
    }
};

TEST(Expressions, canEvaluateLongExpressionsLikeHestonAdjoint)
{
    xad::Tape<double> tape;

    TestHeston<xad::AD> tester;
    xad::AD x = 0.8;
    tape.registerInput(x);
    tape.newRecording();
    xad::AD y = tester.c4(x);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();
    double dx = derivative(x);

    // same with double + bumping
    TestHeston<double> testerd;
    double yd = testerd.c4(value(x));
    double eps = 1e-6;
    double yd_eps = testerd.c4(value(x) + eps);
    double dxd = (yd_eps - yd) / eps;

    EXPECT_DOUBLE_EQ(value(y), yd);
    EXPECT_THAT(dx, DoubleNear(dxd, 1e-5));
}

TEST(Expressions, canEvaluateLongExpressionsLikeHestonForward)
{
    TestHeston<xad::FAD> tester;
    xad::FAD x = 0.8;
    derivative(x) = 1.0;
    xad::FAD y = tester.c4(x);
    double dx = derivative(y);

    // same with double
    TestHeston<double> testerd;
    double yd = testerd.c4(value(x));
    double eps = 1e-6;
    double yd_eps = testerd.c4(value(x) + eps);
    double dxd = (yd_eps - yd) / eps;

    EXPECT_DOUBLE_EQ(value(y), yd);
    EXPECT_THAT(dx, DoubleNear(dxd, 1e-5));
}

namespace
{
template <class Scalar>
inline xad::AReal<Scalar> calc(xad::AReal<Scalar> a, xad::AReal<Scalar> b)
{
    return a * b;
}
class ConstexprTest
{
  public:
    xad::AReal<double> test_func(xad::AReal<double> x)
    {
        xad::AReal<double> z = x;
        x *= 1.0;
        z = a1_ * z;
        z = a1_ * z;
        z = a2_ + z;
        z = a3_ - z;
        z = a4_ / z;
        z = z * b1_;
        z = z + b2_;
        z = z - b3_;
        z = z / b4_;

        using std::min;
        using std::max;

        z = min(z, a1_);
        z = max(z, a2_);
        z = min(b1_, z);
        z = max(b2_, z);

        z = min<xad::AReal<double>>(z, c1_);
        z = min<xad::AReal<double>>(c2_, z);
        z = max<xad::AReal<double>>(z, c3_);
        z = max<xad::AReal<double>>(c4_, z);

        if (z >= c1_ || (z < d3_ && !(z == d4_))) {
            z = min<xad::AReal<double>>(z, d1_);
            z = min<xad::AReal<double>>(d2_, z);
            z = max<xad::AReal<double>>(z, d3_);
            z = max<xad::AReal<double>>(d4_, z);
        }

        return z;
    }

  private:
    static constexpr double a1_ = -3.969683028665376e+01;
    static constexpr double a2_ = 2.209460984245205e+02;
    static constexpr double a3_ = -2.759285104469687e+02;
    static constexpr double a4_ = 1.383577518672690e+02;
    static constexpr double b1_ = -5.447609879822406e+01;
    static constexpr double b2_ = 1.615858368580409e+02;
    static constexpr double b3_ = -1.556989798598866e+02;
    static constexpr double b4_ = 6.680131188771972e+01;
    static constexpr int c1_ = 1;
    static constexpr int c2_ = 2;
    static constexpr int c3_ = 3;
    static constexpr int c4_ = 4;
    static constexpr long long d1_ = 1;
    static constexpr long long d2_ = 2;
    static constexpr long long d3_ = 3;
    static constexpr long long d4_ = 4;
};
}  // namespace

TEST(Expressions, doesNotCaptureConstexprByRef)
{
    // this function should simply compile, without complaining about
    // undefined references in the linker in Linux/Debug
    auto c = ConstexprTest();
    xad::AReal<double> result = c.test_func(1.2);

    EXPECT_THAT(result, Gt(0.0));
}


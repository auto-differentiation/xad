/*******************************************************************************

   Unit tests for derivatives of arithmetic and logical expressions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, version 3 of the License only.

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

TEST(Expressions, manyVariablesInOneStatement)
{
    xad::Tape<double> s;
    xad::AD a(1.0), b(2.0), c(3.0), d(4.0);

    s.registerInput(a);
    s.registerInput(b);
    s.registerInput(c);
    s.registerInput(d);
    s.newRecording();

    xad::AD y = a * b + c * d + a * c + b * d + a * d + b * c;

    static_assert(
        xad::ExprTraits<decltype(a * b + c * d + a * c + b * d + a * d + b * c)>::numVariables ==
            12,
        "wrong number of variables");

    s.registerOutput(y);
    derivative(y) = 1.0;
    s.computeAdjoints();

    EXPECT_DOUBLE_EQ(35.0, value(y));
    EXPECT_DOUBLE_EQ(9.0, derivative(a));
    EXPECT_DOUBLE_EQ(8.0, derivative(b));
    EXPECT_DOUBLE_EQ(7.0, derivative(c));
    EXPECT_DOUBLE_EQ(6.0, derivative(d));
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
    typedef xad::FReal<xad::FReal<double>> AD;

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
    typedef xad::AReal<xad::AReal<double>> AD;

    xad::Tape<double> si;
    xad::Tape<xad::AReal<double>> so;

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
    typedef xad::AReal<xad::FReal<double>> AD;

    xad::Tape<xad::FReal<double>> so;

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
    typedef xad::FReal<xad::AReal<double>> AD;

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
                                                xad::ADVar<double>>>::value),
                  "ad type not wrapped");
    static_assert(
        (std::is_same<
            decltype(x1 + x2 + x1 * x2),
            xad::BinaryExpr<double, xad::add_op<double>,
                            xad::BinaryExpr<double, xad::add_op<double>, xad::ADVar<double>,
                                            xad::ADVar<double>>,
                            xad::BinaryExpr<double, xad::prod_op<double>, xad::ADVar<double>,
                                            xad::ADVar<double>>>>::value),
        "ad type not wrapped");
    static_assert((std::is_same<decltype(max(x1, x1)),
                                xad::BinaryExpr<double, xad::max_op<double>, xad::ADVar<double>,
                                                xad::ADVar<double>>>::value),
                  "AD type not wrapped");
}

TEST(Expressions, supportsRegisteringVectorOutputs)
{
    xad::Tape<double> s;
    xad::AD x1 = 2.0;
    xad::AD x2 = 5.0;
    s.registerInput(x1);
    s.registerInput(x2);
    s.newRecording();
    std::vector<xad::AD> yv;
    yv.emplace_back(x1 + x2);
    s.registerOutputs(yv);
    derivative(yv[0]) = 1.0;
    s.computeAdjoints();
    // s.printStatus();
    EXPECT_DOUBLE_EQ(x1.getValue() + x2.getValue(), yv[0].getValue());
    EXPECT_DOUBLE_EQ(1.0, derivative(x1));
    EXPECT_DOUBLE_EQ(1.0, derivative(x2));
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

namespace
{
template <class Scalar, std::size_t M = 1>
inline xad::AReal<Scalar, M> calc(xad::AReal<Scalar, M> a, xad::AReal<Scalar, M> b)
{
    return a * b;
}
}  // namespace

TEST(Expressions, FmaFunctionTest)
{
    xad::FReal<double> a(1, 1), b(2, 2), c(3, 3), d(1, 1), f(1, 1), r(1, 1);
    f = fma(a, r, b);
    xad::FReal<double> s = a * r + b;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    f = fma(f, r, c);
    s = s * r + c;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    f = fma(f, r, d);
    s = s * r + d;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    xad::FReal<double> df = fma(3 * a, r, 2 * b);
    s = ((3 * a) * r + (2 * b));
    EXPECT_THAT(value(df), value(s));
    EXPECT_THAT(derivative(df), derivative(s));
    auto df2 = fma(3 * a, r * 3, b);
    auto df3 = fma(3., r * 3., 2 * b);
    df = fma(df, df2, df3);
    auto df4 = fma(a, r * 3, b);
    auto df5 = fma(a, r, 2 * b);
    auto df6 = fma(a * 2, r, b);
    auto df7 = fma(df4, df5, df6);
    auto df8 = fma(a * 2, r * 2, b * 2);
    df = fma(df8, df7, df6);
    s = df8 * df7 + df6;
    EXPECT_THAT(value(df), value(s));
    EXPECT_THAT(derivative(df), derivative(s));
}

TEST(Expressions, FmaFunctionTestWithAReal)
{
    xad::Tape<double> tape;
    xad::AReal<double> a = 1, b = 2, c = 3;
    tape.registerInput(a);
    tape.registerInput(b);
    tape.registerInput(c);
    tape.newRecording();
    xad::AReal<double> y = fma(a, b, c);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();
    EXPECT_THAT(derivative(a), DoubleEq(value(b)));
    EXPECT_THAT(derivative(b), DoubleEq(value(a)));
    EXPECT_THAT(derivative(c), DoubleEq(1.0));
}

TEST(Expressions, FmaFunctionTestWithFRealDirect)
{
    xad::FRealDirect<double> a(1, 1), b(2, 2), c(3, 3), d(1, 1), f(1, 1), r(1, 1);
    f = fma(a, r, b);
    xad::FRealDirect<double> s = a * r + b;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    f = fma(f, r, c);
    s = s * r + c;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    f = fma(f, r, d);
    s = s * r + d;
    EXPECT_THAT(value(f), value(s));
    EXPECT_THAT(derivative(f), derivative(s));
    xad::FRealDirect<double> df = fma(3 * a, r, 2 * b);
    s = ((3 * a) * r + (2 * b));
    EXPECT_THAT(value(df), value(s));
    EXPECT_THAT(derivative(df), derivative(s));
    auto df2 = fma(3 * a, r * 3, b);
    auto df3 = fma(3., r * 3., 2 * b);
    df = fma(df, df2, df3);
    auto df4 = fma(a, r * 3, b);
    auto df5 = fma(a, r, 2 * b);
    auto df6 = fma(a * 2, r, b);
    auto df7 = fma(df4, df5, df6);
    auto df8 = fma(a * 2, r * 2, b * 2);
    df = fma(df8, df7, df6);
    s = df8 * df7 + df6;
    EXPECT_THAT(value(df), value(s));
    EXPECT_THAT(derivative(df), derivative(s));
}

TEST(Expressions, FmaFunctionTestWithFwd_Adj)
{
    xad::Tape<double> tape;
    xad::FReal<xad::AReal<double>> a = 1, b = 2, c = 3;
    tape.registerInput(value(a));
    tape.registerInput(derivative(a));
    tape.registerInput(value(b));
    tape.registerInput(derivative(b));
    tape.registerInput(value(c));
    tape.registerInput(derivative(c));
    tape.newRecording();
    xad::FReal<xad::AReal<double>> y = fma(a, b, c);
    tape.registerOutput(derivative(y));
    auto yv = derivative(y);
    derivative(yv) = 1.0;
    tape.computeAdjoints();
    EXPECT_THAT(derivative(derivative(a)), value(b));
    EXPECT_THAT(derivative(derivative(b)), value(a));
    EXPECT_THAT(derivative(derivative(c)), DoubleEq(1.0));
}

/*******************************************************************************

   Unit tests for Meta information of expressions.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <XAD/XAD.hpp>

#include <type_traits>

using namespace ::testing;

/*
   NOTE: None of these tests are doing anything at runtime - they are
   purely compile-time assertions. If this file compiles, we're good.
*/

TEST(ExpressionMeta, IdentifiesNoneExpressions)
{
    static_assert(xad::ExprTraits<double>::isExpr == false,
                  "double expression is not XAD expression type");
    static_assert(xad::ExprTraits<int>::isExpr == false,
                  "int expression is not XAD expression type");
    struct XX;
    static_assert(xad::ExprTraits<XX>::isExpr == false,
                  "custom struct expression is not XAD expression type");
}

TEST(ExpressionMeta, IdentifiesExpressions)
{
    struct op;
    static_assert((xad::ExprTraits<xad::UnaryExpr<double, op, double>>::isExpr == true),
                  "Expression Not Identified");
}

TEST(ExpressionMeta, DeterminesUnderlyingTypeForScalar)
{
    static_assert(
        (std::is_same<xad::ExprTraits<xad::AReal<double>>::value_type, xad::AReal<double>>::value),
        "Expression Not Identified");
    static_assert(
        (std::is_same<xad::ExprTraits<xad::AReal<float>>::value_type, xad::AReal<float>>::value),
        "Expression Not Identified");
    static_assert(
        (std::is_same<xad::ExprTraits<xad::FReal<double>>::value_type, xad::FReal<double>>::value),
        "Expression Not Identified");
    static_assert(
        (std::is_same<xad::ExprTraits<xad::FReal<float>>::value_type, xad::FReal<float>>::value),
        "Expression Not Identified");
    static_assert((std::is_same<xad::ExprTraits<double>::value_type, double>::value),
                  "Expression Not Identified");
    static_assert((std::is_same<xad::ExprTraits<float>::value_type, float>::value),
                  "Expression Not Identified");
    static_assert((std::is_same<xad::ExprTraits<xad::AReal<xad::AReal<double>>>::value_type,
                                xad::AReal<xad::AReal<double>>>::value),
                  "Expression Not Identified");
}

TEST(ExpressionMeta, DeterminesUnderlyingTypeForUnaryExpr)
{
    xad::Tape<double> t;  // need this for AD instantiation
    xad::AReal<double> a;
    xad::FReal<double> f;
    // auto captures the expression here
    auto minus_a = -a;
    auto minus_f = -f;
    auto sin_a = sin(a);
    auto sin_f = sin(f);
    auto cos_sin_a = cos(sin_a);
    auto cos_sin_f = cos(sin_f);

    static_assert(
        (std::is_same<xad::ExprTraits<decltype(minus_a)>::value_type, xad::AReal<double>>::value),
        "Unary plus expr of AReal value_type should be AReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(minus_f)>::value_type, xad::FReal<double>>::value),
        "Unary plus expr of FReal value_type should be FReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(sin_a)>::value_type, xad::AReal<double>>::value),
        "Sin expr of AReal value_type should be AReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(sin_f)>::value_type, xad::FReal<double>>::value),
        "Sin expr of FReal value_type should be FReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(cos_sin_a)>::value_type, xad::AReal<double>>::value),
        "Sin of cos expr of AReal value_type should be AReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(cos_sin_f)>::value_type, xad::FReal<double>>::value),
        "Sin of cos expr of FReal value_type should be FReal");
}

TEST(ExpressionMeta, DeterminesUnderlyingTypeForBinaryExpr)
{
    xad::Tape<double> t;  // need this for AD instantiation
    xad::AReal<double> a;
    xad::FReal<double> f;
    // auto captures the expression here
    auto a_plus = a + a;
    auto f_plus = f + f;
    auto a_plus_scalar = a + 1.0;
    auto f_plus_scalar = f + 1.0;
    auto a_scalar_plus = 1.0 + a;
    auto f_scalar_plus = 1.0 + f;
    auto a_plus_unary = a + (-a);
    auto f_plus_unary = f + (-f);
    auto a_pow = pow(a, a);
    auto f_pow = pow(f, f);
    auto a_pow_scalar = pow(a, 1.0);
    auto f_pow_scalar = pow(f, 1.0);
    auto a_scalar_pow = pow(1.0, a);
    auto f_scalar_pow = pow(1.0, f);
    auto a_long_expr = sqrt(a * 2.0 + a * a + sin(a) / cos(a)) + 2.4;
    auto f_long_expr = sqrt(f * 2.0 + f * f + sin(f) / cos(f)) + 2.4;

    // adjoint mode
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(a_plus)>::value_type, xad::AReal<double>>::value),
        "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_plus_scalar)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_scalar_plus)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_plus_unary)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(a_pow)>::value_type, xad::AReal<double>>::value),
        "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_pow_scalar)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_scalar_pow)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_long_expr)>::value_type,
                                xad::AReal<double>>::value),
                  "Binary expr of AReal value_type should be AReal");

    // forward mode
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(f_plus)>::value_type, xad::FReal<double>>::value),
        "Binary expr of FReal value_type should be FReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_plus_scalar)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be FReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_scalar_plus)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be FReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_plus_unary)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be FReal");
    static_assert(
        (std::is_same<xad::ExprTraits<decltype(f_pow)>::value_type, xad::FReal<double>>::value),
        "Binary expr of AReal value_type should be FReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_pow_scalar)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be FReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_scalar_pow)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be AReal");
    static_assert((std::is_same<xad::ExprTraits<decltype(f_long_expr)>::value_type,
                                xad::FReal<double>>::value),
                  "Binary expr of FReal value_type should be FReal");
}

TEST(ExpressionMeta, DeterminesUnderlyingTypeForNestedBinaryExpr)
{
    xad::Tape<xad::FReal<double>> t;  // need this for AD instantiation
    xad::AReal<xad::FReal<double>> a;

    // auto captures the expression here
    auto a_plus = a + a;
    auto a_plus_scalar = a + 1.0;

    static_assert((std::is_same<xad::ExprTraits<decltype(a_plus)>::value_type,
                                xad::AReal<xad::FReal<double>>>::value),
                  "Expr of nested base type should have the base type as value_type");
    static_assert((std::is_same<xad::ExprTraits<decltype(a_plus_scalar)>::value_type,
                                xad::AReal<xad::FReal<double>>>::value),
                  "Expr of nested base type should have the base type as value_type");
}

TEST(ExpressionMeta, plainDoubleTraits)
{
    static_assert(!xad::ExprTraits<double>::isExpr, "not an expression");
    static_assert(!xad::ExprTraits<double>::isLiteral, "not a literal");
    static_assert(xad::ExprTraits<double>::direction == xad::Direction::DIR_NONE,
                  "direction should be none");
}

TEST(ExpressionMeta, forwardLiteralTraits)
{
    typedef xad::FReal<double> type;
    static_assert(xad::ExprTraits<type>::isExpr, "should be an expression");
    static_assert(xad::ExprTraits<type>::isLiteral, "should be a literal");
    static_assert(xad::ExprTraits<type>::isForward, "should be forward mode");
    static_assert(!xad::ExprTraits<type>::isReverse, "should not be reverse mode");
    static_assert(xad::ExprTraits<type>::numVariables == 1, "should be one variable");
    static_assert(xad::ExprTraits<type>::direction == xad::Direction::DIR_FORWARD,
                  "should be forward");
}

TEST(ExpressionMeta, forwardExprTraits)
{
    xad::FReal<double> x, y;
    auto binx = x * x;
    auto binx2 = binx + 2. * y;
    typedef decltype(binx2) type;

    static_assert(xad::ExprTraits<type>::isExpr, "should be an expression");
    static_assert(!xad::ExprTraits<type>::isLiteral, "is not a literal");
    static_assert(xad::ExprTraits<type>::isForward, "should be forward");
    static_assert(!xad::ExprTraits<type>::isReverse, "should not be reverse");
    static_assert(xad::ExprTraits<type>::numVariables == 3, "should be 3 variables");
    static_assert(xad::ExprTraits<type>::direction == xad::Direction::DIR_FORWARD,
                  "should be forward direction");
}

TEST(ExpressionMeta, reverseLiteralTraits)
{
    typedef xad::AReal<double> type;
    static_assert(xad::ExprTraits<type>::isExpr, "should be an expression");
    static_assert(xad::ExprTraits<type>::isLiteral, "should be a literal");
    static_assert(!xad::ExprTraits<type>::isForward, "should not be forward");
    static_assert(xad::ExprTraits<type>::isReverse, "should be reverse");
    static_assert(xad::ExprTraits<type>::numVariables == 1, "should be 1 variable");
    static_assert(xad::ExprTraits<type>::direction == xad::Direction::DIR_REVERSE,
                  "should be reverse direction");
}

TEST(ExpressionMeta, reverseExprTraits)
{
    xad::AReal<double> x, y;
    auto binx = x * x;
    auto binx2 = binx + 2. * y;
    typedef decltype(binx2) type;

    static_assert(xad::ExprTraits<type>::isExpr, "should be an expression");
    static_assert(!xad::ExprTraits<type>::isLiteral, "should not be a literal");
    static_assert(!xad::ExprTraits<type>::isForward, "should not be forward");
    static_assert(xad::ExprTraits<type>::isReverse, "should be reverse");
    static_assert(xad::ExprTraits<type>::numVariables == 3, "should be 3 variables");
    static_assert(xad::ExprTraits<type>::direction == xad::Direction::DIR_REVERSE,
                  "should be reverse");
}

TEST(ExpressionMeta, LongExpression) {
    using complex_expr = xad::BinaryExpr<
        double, xad::prod_op<double>,
        xad::UnaryExpr<double, xad::scalar_prod_op<double, double>, xad::ADVar<double>>,
        xad::UnaryExpr<double, xad::exp_op<double>,
                       xad::BinaryExpr<double, xad::prod_op<double>,
                                       xad::UnaryExpr<double, xad::scalar_prod_op<double, double>,
                                                      xad::ADVar<double>>,
                                       xad::ADVar<double>>>>;

    static_assert(xad::ExprTraits<complex_expr>::isExpr, "should be an expression");
    static_assert(!xad::ExprTraits<complex_expr>::isLiteral, "should not be a literal");
    static_assert(!xad::ExprTraits<complex_expr>::isForward, "should not be forward");
    static_assert(xad::ExprTraits<complex_expr>::isReverse, "should be reverse");
    static_assert(xad::ExprTraits<complex_expr>::numVariables == 3, "should be 3 variables");
    static_assert(xad::ExprTraits<complex_expr>::direction == xad::Direction::DIR_REVERSE,
                  "should be reverse");
    
}
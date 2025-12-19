/*******************************************************************************

   Unit tests for JITGraphInterpreter

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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
#include <cmath>
#include <memory>
#include <utility>

#ifdef XAD_ENABLE_JIT
TEST(JITGraphInterpreter, executeBasicOperations)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    // Test add
    {
        AD a = 2.0, b = 3.0;
        jit.registerInput(a);
        jit.registerInput(b);
        AD c = a + b;
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_DOUBLE_EQ(5.0, output);
        jit.newRecording();
    }

    // Test subtract
    {
        AD a = 5.0, b = 3.0;
        jit.registerInput(a);
        jit.registerInput(b);
        AD c = a - b;
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_DOUBLE_EQ(2.0, output);
        jit.newRecording();
    }

    // Test multiply
    {
        AD a = 4.0, b = 3.0;
        jit.registerInput(a);
        jit.registerInput(b);
        AD c = a * b;
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_DOUBLE_EQ(12.0, output);
        jit.newRecording();
    }

    // Test divide
    {
        AD a = 12.0, b = 3.0;
        jit.registerInput(a);
        jit.registerInput(b);
        AD c = a / b;
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_DOUBLE_EQ(4.0, output);
        jit.newRecording();
    }
}

TEST(JITGraphInterpreter, executeUnaryMathFunctions)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    // Test sin
    {
        AD a = 1.0;
        jit.registerInput(a);
        AD c = sin(a);
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_NEAR(std::sin(1.0), output, 1e-10);
        jit.newRecording();
    }

    // Test cos
    {
        AD a = 1.0;
        jit.registerInput(a);
        AD c = cos(a);
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_NEAR(std::cos(1.0), output, 1e-10);
        jit.newRecording();
    }

    // Test exp
    {
        AD a = 2.0;
        jit.registerInput(a);
        AD c = exp(a);
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_NEAR(std::exp(2.0), output, 1e-10);
        jit.newRecording();
    }

    // Test log
    {
        AD a = 2.0;
        jit.registerInput(a);
        AD c = log(a);
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_NEAR(std::log(2.0), output, 1e-10);
        jit.newRecording();
    }

    // Test sqrt
    {
        AD a = 4.0;
        jit.registerInput(a);
        AD c = sqrt(a);
        jit.registerOutput(c);
        jit.compile();
        double output;
        jit.forward(&output, 1);
        EXPECT_NEAR(2.0, output, 1e-10);
    }
}

TEST(JITGraphInterpreter, executeNegation)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 5.0;

    jit.registerInput(a);
    AD c = -a;
    jit.registerOutput(c);
    jit.compile();

    double output;
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(-5.0, output);
}

TEST(JITGraphInterpreter, complexExpressionWorks)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD x = 2.0;
    AD y = 3.0;

    jit.registerInput(x);
    jit.registerInput(y);

    // (x^2 + y) * sin(x) / y
    AD result = (x * x + y) * sin(x) / y;
    jit.registerOutput(result);

    jit.compile();

    double output;
    jit.forward(&output, 1);

    double expected = (2.0 * 2.0 + 3.0) * std::sin(2.0) / 3.0;
    EXPECT_NEAR(expected, output, 1e-10);
}

TEST(JITGraphInterpreter, adjointsForComplexExpression)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD x = 2.0;
    AD y = 3.0;

    jit.registerInput(x);
    jit.registerInput(y);

    // f(x,y) = x^2 + y^2
    // df/dx = 2x = 4, df/dy = 2y = 6
    AD result = x * x + y * y;
    jit.registerOutput(result);

    jit.compile();
    jit.setDerivative(result.getSlot(), 1.0);
    jit.computeAdjoints();

    EXPECT_NEAR(4.0, jit.getDerivative(x.getSlot()), 1e-10);
    EXPECT_NEAR(6.0, jit.getDerivative(y.getSlot()), 1e-10);
}

// =============================================================================
// ABool tests
// =============================================================================

TEST(JITGraphInterpreter, squareOpCode)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t sq = graph.addUnary(xad::JITOpCode::Square, inp);
    graph.markOutput(sq);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 3.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    EXPECT_DOUBLE_EQ(9.0, output);
}

TEST(JITGraphInterpreter, squareAdjoint)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t sq = graph.addUnary(xad::JITOpCode::Square, inp);
    graph.markOutput(sq);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 3.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // d(x^2)/dx = 2x = 6
    EXPECT_DOUBLE_EQ(9.0, output);
    EXPECT_DOUBLE_EQ(6.0, inputAdjoint);
}

TEST(JITGraphInterpreter, recipOpCode)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t rec = graph.addUnary(xad::JITOpCode::Recip, inp);
    graph.markOutput(rec);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 4.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    EXPECT_DOUBLE_EQ(0.25, output);
}

TEST(JITGraphInterpreter, recipAdjoint)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t rec = graph.addUnary(xad::JITOpCode::Recip, inp);
    graph.markOutput(rec);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 2.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // d(1/x)/dx = -1/x^2 = -1/4 = -0.25
    EXPECT_DOUBLE_EQ(0.5, output);
    EXPECT_DOUBLE_EQ(-0.25, inputAdjoint);
}

TEST(JITGraphInterpreter, smoothAbsOpCode)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(0.5);  // smoothing parameter
    uint32_t sa = graph.addBinary(xad::JITOpCode::SmoothAbs, inp, c);
    graph.markOutput(sa);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    // Test in the smooth region (|x| < c)
    double input = 0.3;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    // For x > 0 and |x| < c: x^2 * (2/c - x/c^2)
    double c_val = 0.5;
    double expected = input * input * (2.0 / c_val - input / (c_val * c_val));
    EXPECT_NEAR(expected, output, 1e-10);

    // Test outside smooth region (|x| > c)
    input = 1.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // Should be |x|
}

TEST(JITGraphInterpreter, smoothAbsNegative)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(0.5);
    uint32_t sa = graph.addBinary(xad::JITOpCode::SmoothAbs, inp, c);
    graph.markOutput(sa);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    // Test negative value in smooth region
    double input = -0.3;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    // For x < 0 and |x| < c: x^2 * (2/c + x/c^2)
    double c_val = 0.5;
    double expected = input * input * (2.0 / c_val + input / (c_val * c_val));
    EXPECT_NEAR(expected, output, 1e-10);

    // Test negative outside smooth region
    input = -1.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // Should be |x|
}

TEST(JITGraphInterpreter, smoothAbsAdjoint)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(0.5);
    uint32_t sa = graph.addBinary(xad::JITOpCode::SmoothAbs, inp, c);
    graph.markOutput(sa);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    // Test adjoint in smooth region (positive x)
    double input = 0.3;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // For positive x in smooth region: derivative is -x/(c^2) * (3x - 4c)
    double c_val = 0.5;
    double expected_deriv = -input / (c_val * c_val) * (3.0 * input - 4.0 * c_val);
    EXPECT_NEAR(expected_deriv, inputAdjoint, 1e-10);

    // Test adjoint in smooth region (negative x)
    input = -0.3;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // For negative x in smooth region: derivative is x/(c^2) * (3x + 4c)
    expected_deriv = input / (c_val * c_val) * (3.0 * input + 4.0 * c_val);
    EXPECT_NEAR(expected_deriv, inputAdjoint, 1e-10);

    // Test adjoint outside smooth region (positive)
    input = 1.0;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);
    EXPECT_DOUBLE_EQ(1.0, inputAdjoint);  // d|x|/dx = 1 for x > 0

    // Test adjoint outside smooth region (negative)
    input = -1.0;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);
    EXPECT_DOUBLE_EQ(-1.0, inputAdjoint);  // d|x|/dx = -1 for x < 0
}

// =============================================================================
// Comparison OpCode tests
// =============================================================================

TEST(JITGraphInterpreter, cmpLTOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpLT, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 3.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 3 < 5 is true

    input = 7.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 7 < 5 is false
}

TEST(JITGraphInterpreter, cmpLEOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpLE, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 <= 5 is true

    input = 6.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 6 <= 5 is false
}

TEST(JITGraphInterpreter, cmpGTOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpGT, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 7.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 7 > 5 is true

    input = 3.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 3 > 5 is false
}

TEST(JITGraphInterpreter, cmpGEOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpGE, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 >= 5 is true

    input = 4.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 4 >= 5 is false
}

TEST(JITGraphInterpreter, cmpEQOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpEQ, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 == 5 is true

    input = 4.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 4 == 5 is false
}

TEST(JITGraphInterpreter, cmpNEOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::JITOpCode::CmpNE, a, b);
    graph.markOutput(cmp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 4.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(1.0, output);  // 4 != 5 is true

    input = 5.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(0.0, output);  // 5 != 5 is false
}

// =============================================================================
// If OpCode tests
// =============================================================================

TEST(JITGraphInterpreter, ifOpCodeTrueBranch)
{
    xad::JITGraph graph;
    uint32_t cond = graph.addConstant(1.0);  // true
    uint32_t t = graph.addConstant(10.0);
    uint32_t f = graph.addConstant(20.0);
    uint32_t result = graph.addTernary(xad::JITOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double output;
    interp.forward(graph, nullptr, 0, &output, 1);
    EXPECT_DOUBLE_EQ(10.0, output);  // condition is true, return trueVal
}

TEST(JITGraphInterpreter, ifOpCodeFalseBranch)
{
    xad::JITGraph graph;
    uint32_t cond = graph.addConstant(0.0);  // false
    uint32_t t = graph.addConstant(10.0);
    uint32_t f = graph.addConstant(20.0);
    uint32_t result = graph.addTernary(xad::JITOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double output;
    interp.forward(graph, nullptr, 0, &output, 1);
    EXPECT_DOUBLE_EQ(20.0, output);  // condition is false, return falseVal
}

TEST(JITGraphInterpreter, ifOpCodeAdjointTrueBranch)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t cond = graph.addConstant(1.0);  // true
    uint32_t t = graph.addBinary(xad::JITOpCode::Mul, inp, graph.addConstant(2.0));  // 2*x
    uint32_t f = graph.addBinary(xad::JITOpCode::Mul, inp, graph.addConstant(3.0));  // 3*x
    uint32_t result = graph.addTernary(xad::JITOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(10.0, output);  // 2*5
    EXPECT_DOUBLE_EQ(2.0, inputAdjoint);  // d(2x)/dx = 2
}

TEST(JITGraphInterpreter, ifOpCodeAdjointFalseBranch)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t cond = graph.addConstant(0.0);  // false
    uint32_t t = graph.addBinary(xad::JITOpCode::Mul, inp, graph.addConstant(2.0));  // 2*x
    uint32_t f = graph.addBinary(xad::JITOpCode::Mul, inp, graph.addConstant(3.0));  // 3*x
    uint32_t result = graph.addTernary(xad::JITOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(15.0, output);  // 3*5
    EXPECT_DOUBLE_EQ(3.0, inputAdjoint);  // d(3x)/dx = 3
}

// =============================================================================
// Additional OpCode tests for coverage
// =============================================================================

TEST(JITGraphInterpreter, modOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(3.0);
    uint32_t result = graph.addBinary(xad::JITOpCode::Mod, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 7.5;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(std::fmod(7.5, 3.0), output);
}

TEST(JITGraphInterpreter, modAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Mod, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double inputs[2] = {7.5, 3.0};
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    // d(fmod(a,b))/da = 1, d(fmod(a,b))/db = -floor(a/b)
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    EXPECT_DOUBLE_EQ(-std::floor(7.5 / 3.0), inputAdjoints[1]);
}

TEST(JITGraphInterpreter, copysignOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(-1.0);
    uint32_t result = graph.addBinary(xad::JITOpCode::Copysign, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(-5.0, output);  // copysign(5, -1) = -5

    input = -3.0;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(-3.0, output);  // copysign(-3, -1) = -3
}

TEST(JITGraphInterpreter, copysignAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Copysign, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    // Test with positive b
    double inputs[2] = {5.0, 1.0};
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    // d/da copysign(a, b) = sign(b) = 1
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    // d/db copysign(a, b) = 0
    EXPECT_DOUBLE_EQ(0.0, inputAdjoints[1]);

    // Test with negative b
    inputs[1] = -1.0;
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);
    EXPECT_DOUBLE_EQ(-1.0, inputAdjoints[0]);  // sign(b) = -1
}

TEST(JITGraphInterpreter, frexpOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::JITOpCode::Frexp, a);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 8.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    int exp;
    double expected = std::frexp(8.0, &exp);
    EXPECT_DOUBLE_EQ(expected, output);  // frexp(8) = 0.5, exp=4
}

TEST(JITGraphInterpreter, frexpAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::JITOpCode::Frexp, a);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 8.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // Derivative of frexp mantissa is 1 / 2^exp
    int exp;
    std::frexp(input, &exp);
    EXPECT_DOUBLE_EQ(1.0 / (1 << exp), inputAdjoint);
}

TEST(JITGraphInterpreter, modfOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::JITOpCode::Modf, a);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 3.75;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    double intpart;
    double expected = std::modf(3.75, &intpart);
    EXPECT_DOUBLE_EQ(expected, output);  // fractional part = 0.75
}

TEST(JITGraphInterpreter, modfAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::JITOpCode::Modf, a);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 3.75;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    // Derivative of fractional part is 1
    EXPECT_DOUBLE_EQ(1.0, inputAdjoint);
}

TEST(JITGraphInterpreter, remquoOpCode)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(3.0);
    uint32_t result = graph.addBinary(xad::JITOpCode::Remquo, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 7.5;
    double output;
    interp.forward(graph, &input, 1, &output, 1);

    int quo;
    double expected = std::remquo(7.5, 3.0, &quo);
    EXPECT_DOUBLE_EQ(expected, output);
}

TEST(JITGraphInterpreter, remquoAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Remquo, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double inputs[2] = {7.5, 3.0};
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    int quo;
    const double rem = std::remquo(7.5, 3.0, &quo);
    XAD_UNUSED_VARIABLE(rem);
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    EXPECT_DOUBLE_EQ(-static_cast<double>(quo), inputAdjoints[1]);
}

TEST(JITGraphInterpreter, smoothAbsCDerivative)
{
    // Test the derivative w.r.t. the c parameter (smoothing width)
    // We need two inputs: x and c
    xad::JITGraph graph;
    uint32_t x = graph.addInput();
    uint32_t c = graph.addInput();
    uint32_t sa = graph.addBinary(xad::JITOpCode::SmoothAbs, x, c);
    graph.markOutput(sa);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    // Test in smooth region (positive x)
    double inputs[2] = {0.3, 0.5};
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    // dc derivative for positive x in smooth region: -2*x^2*(c-x)/(c^3)
    double x_val = 0.3, c_val = 0.5;
    double expected_dc = -2.0 * x_val * x_val * (c_val - x_val) / (c_val * c_val * c_val);
    EXPECT_NEAR(expected_dc, inputAdjoints[1], 1e-10);

    // Test in smooth region (negative x)
    inputs[0] = -0.3;
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    // dc derivative for negative x in smooth region: -2*x^2*(c+x)/(c^3)
    x_val = -0.3;
    expected_dc = -2.0 * x_val * x_val * (c_val + x_val) / (c_val * c_val * c_val);
    EXPECT_NEAR(expected_dc, inputAdjoints[1], 1e-10);

    // Test outside smooth region - dc should be 0
    inputs[0] = 1.0;  // |x| > c
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);
    EXPECT_DOUBLE_EQ(0.0, inputAdjoints[1]);
}

TEST(JITGraphInterpreter, minEqualValues)
{
    // Test min adjoint when values are equal (splits 50/50)
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Min, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double inputs[2] = {5.0, 5.0};  // Equal values
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    EXPECT_DOUBLE_EQ(5.0, output);
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[0]);  // Split 50/50
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[1]);
}

TEST(JITGraphInterpreter, maxEqualValues)
{
    // Test max adjoint when values are equal (splits 50/50)
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Max, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double inputs[2] = {5.0, 5.0};  // Equal values
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    EXPECT_DOUBLE_EQ(5.0, output);
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[0]);  // Split 50/50
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[1]);
}

TEST(JITGraphInterpreter, ldexpAdjoint)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    // ldexp uses immediate for exponent
    uint32_t result = graph.addNode(xad::JITOpCode::Ldexp, a, 0, 0, 3.0);  // ldexp(a, 3) = a * 8
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 2.0;
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(graph, &input, 1, &outputAdjoint, 1, &output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(16.0, output);  // 2 * 2^3 = 16
    EXPECT_DOUBLE_EQ(8.0, inputAdjoint);  // d(a*8)/da = 8
}

TEST(JITGraphInterpreter, powAdjointWithZeroBase)
{
    // Test pow adjoint when base > 0 (log path is taken)
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::JITOpCode::Pow, a, b);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double inputs[2] = {2.0, 3.0};  // 2^3 = 8
    double outputAdjoint = 1.0;
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(graph, inputs, 2, &outputAdjoint, 1, &output, inputAdjoints);

    // d(a^b)/da = b * a^(b-1) = 3 * 4 = 12
    EXPECT_DOUBLE_EQ(12.0, inputAdjoints[0]);
    // d(a^b)/db = a^b * log(a) = 8 * log(2)
    EXPECT_NEAR(8.0 * std::log(2.0), inputAdjoints[1], 1e-10);
}

TEST(JITGraphInterpreter, reset)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t result = graph.addUnary(xad::JITOpCode::Neg, inp);
    graph.markOutput(result);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(-5.0, output);

    // Reset and verify we can still use it after recompiling
    interp.reset();
    interp.compile(graph);
    interp.forward(graph, &input, 1, &output, 1);
    EXPECT_DOUBLE_EQ(-5.0, output);
}

TEST(JITGraphInterpreter, forwardInputCountMismatch)
{
    xad::JITGraph graph;
    graph.addInput();
    graph.addInput();
    uint32_t c = graph.addConstant(1.0);
    graph.markOutput(c);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double output;
    // Graph expects 2 inputs but we provide 1
    EXPECT_THROW(interp.forward(graph, &input, 1, &output, 1), std::runtime_error);
}

TEST(JITGraphInterpreter, forwardOutputCountMismatch)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    graph.markOutput(inp);

    xad::JITGraphInterpreter interp;
    interp.compile(graph);

    double input = 5.0;
    double outputs[2];
    // Graph has 1 output but we request 2
    EXPECT_THROW(interp.forward(graph, &input, 1, outputs, 2), std::runtime_error);
}


#endif  // XAD_ENABLE_JIT

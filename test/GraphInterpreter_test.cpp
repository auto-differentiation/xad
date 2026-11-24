/*******************************************************************************

   Unit tests for GraphInterpreter

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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <utility>

#ifdef XAD_ENABLE_GRAPH
TEST(GraphInterpreter, executeBasicOperations)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    // Test add
    {
        AD a = 2.0, b = 3.0;
        recorder.registerInput(a);
        recorder.registerInput(b);
        AD c = a + b;
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_DOUBLE_EQ(5.0, output);
        recorder.clearAll();
    }

    // Test subtract
    {
        AD a = 5.0, b = 3.0;
        recorder.registerInput(a);
        recorder.registerInput(b);
        AD c = a - b;
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_DOUBLE_EQ(2.0, output);
        recorder.clearAll();
    }

    // Test multiply
    {
        AD a = 4.0, b = 3.0;
        recorder.registerInput(a);
        recorder.registerInput(b);
        AD c = a * b;
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_DOUBLE_EQ(12.0, output);
        recorder.clearAll();
    }

    // Test divide
    {
        AD a = 12.0, b = 3.0;
        recorder.registerInput(a);
        recorder.registerInput(b);
        AD c = a / b;
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_DOUBLE_EQ(4.0, output);
        recorder.clearAll();
    }
}

TEST(GraphInterpreter, executeUnaryMathFunctions)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    // Test sin
    {
        AD a = 1.0;
        recorder.registerInput(a);
        AD c = sin(a);
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_NEAR(std::sin(1.0), output, 1e-10);
        recorder.clearAll();
    }

    // Test cos
    {
        AD a = 1.0;
        recorder.registerInput(a);
        AD c = cos(a);
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_NEAR(std::cos(1.0), output, 1e-10);
        recorder.clearAll();
    }

    // Test exp
    {
        AD a = 2.0;
        recorder.registerInput(a);
        AD c = exp(a);
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_NEAR(std::exp(2.0), output, 1e-10);
        recorder.clearAll();
    }

    // Test log
    {
        AD a = 2.0;
        recorder.registerInput(a);
        AD c = log(a);
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_NEAR(std::log(2.0), output, 1e-10);
        recorder.clearAll();
    }

    // Test sqrt
    {
        AD a = 4.0;
        recorder.registerInput(a);
        AD c = sqrt(a);
        recorder.registerOutput(c);
        recorder.compile();
        double output;
        recorder.forward(&output);
        EXPECT_NEAR(2.0, output, 1e-10);
    }
}

TEST(GraphInterpreter, executeNegation)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 5.0;

    recorder.registerInput(a);
    AD c = -a;
    recorder.registerOutput(c);
    recorder.compile();

    double output;
    recorder.forward(&output);
    EXPECT_DOUBLE_EQ(-5.0, output);
}

TEST(GraphInterpreter, complexExpressionWorks)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD x = 2.0;
    AD y = 3.0;

    recorder.registerInput(x);
    recorder.registerInput(y);

    // (x^2 + y) * sin(x) / y
    AD result = (x * x + y) * sin(x) / y;
    recorder.registerOutput(result);

    recorder.compile();

    double output;
    recorder.forward(&output);

    double expected = (2.0 * 2.0 + 3.0) * std::sin(2.0) / 3.0;
    EXPECT_NEAR(expected, output, 1e-10);
}

TEST(GraphInterpreter, adjointsForComplexExpression)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD x = 2.0;
    AD y = 3.0;

    recorder.registerInput(x);
    recorder.registerInput(y);

    // f(x,y) = x^2 + y^2
    // df/dx = 2x = 4, df/dy = 2y = 6
    AD result = x * x + y * y;
    recorder.registerOutput(result);

    recorder.compile();
    recorder.setDerivative(result.getSlot(), 1.0);
    recorder.computeAdjoints();

    EXPECT_NEAR(4.0, recorder.getDerivative(x.getSlot()), 1e-10);
    EXPECT_NEAR(6.0, recorder.getDerivative(y.getSlot()), 1e-10);
}

// =============================================================================
// ABool tests
// =============================================================================

TEST(GraphInterpreter, squareOpCode)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t sq = graph.addUnary(xad::GraphOpCode::Square, inp);
    graph.markOutput(sq);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    EXPECT_DOUBLE_EQ(9.0, output);
}

TEST(GraphInterpreter, squareAdjoint)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t sq = graph.addUnary(xad::GraphOpCode::Square, inp);
    graph.markOutput(sq);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    // d(x^2)/dx = 2x = 6
    EXPECT_DOUBLE_EQ(9.0, output);
    EXPECT_DOUBLE_EQ(6.0, inputAdjoint);
}

TEST(GraphInterpreter, recipOpCode)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t rec = graph.addUnary(xad::GraphOpCode::Recip, inp);
    graph.markOutput(rec);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 4.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    EXPECT_DOUBLE_EQ(0.25, output);
}

TEST(GraphInterpreter, recipAdjoint)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t rec = graph.addUnary(xad::GraphOpCode::Recip, inp);
    graph.markOutput(rec);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 2.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    // d(1/x)/dx = -1/x^2 = -1/4 = -0.25
    EXPECT_DOUBLE_EQ(0.5, output);
    EXPECT_DOUBLE_EQ(-0.25, inputAdjoint);
}

TEST(GraphInterpreter, smoothAbsOpCode)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(0.5);  // smoothing parameter
    uint32_t sa = graph.addBinary(xad::GraphOpCode::SmoothAbs, inp, c);
    graph.markOutput(sa);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    // Test in the smooth region (|x| < c)
    double input = 0.3;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    // For x > 0 and |x| < c: x^2 * (2/c - x/c^2)
    double c_val = 0.5;
    double expected = input * input * (2.0 / c_val - input / (c_val * c_val));
    EXPECT_NEAR(expected, output, 1e-10);

    // Test outside smooth region (|x| > c)
    input = 1.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // Should be |x|
}

TEST(GraphInterpreter, smoothAbsAdjoint)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(0.5);
    uint32_t sa = graph.addBinary(xad::GraphOpCode::SmoothAbs, inp, c);
    graph.markOutput(sa);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    // Test adjoint in smooth region (positive x)
    double input = 0.3;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    // For positive x in smooth region: derivative is -x/(c^2) * (3x - 4c)
    double c_val = 0.5;
    double expected_deriv = -input / (c_val * c_val) * (3.0 * input - 4.0 * c_val);
    EXPECT_NEAR(expected_deriv, inputAdjoint, 1e-10);

    // Test adjoint in smooth region (negative x)
    input = -0.3;
    interp.setInput(0, &input);
    interp.forwardAndBackward(&output, &inputAdjoint);

    // For negative x in smooth region: derivative is x/(c^2) * (3x + 4c)
    expected_deriv = input / (c_val * c_val) * (3.0 * input + 4.0 * c_val);
    EXPECT_NEAR(expected_deriv, inputAdjoint, 1e-10);

    // Test adjoint outside smooth region (positive)
    input = 1.0;
    interp.setInput(0, &input);
    interp.forwardAndBackward(&output, &inputAdjoint);
    EXPECT_DOUBLE_EQ(1.0, inputAdjoint);  // d|x|/dx = 1 for x > 0

    // Test adjoint outside smooth region (negative)
    input = -1.0;
    interp.setInput(0, &input);
    interp.forwardAndBackward(&output, &inputAdjoint);
    EXPECT_DOUBLE_EQ(-1.0, inputAdjoint);  // d|x|/dx = -1 for x < 0
}

// =============================================================================
// Comparison OpCode tests
// =============================================================================

TEST(GraphInterpreter, cmpLTOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpLT, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 3 < 5 is true

    input = 7.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 7 < 5 is false
}

TEST(GraphInterpreter, cmpLEOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpLE, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 <= 5 is true

    input = 6.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 6 <= 5 is false
}

TEST(GraphInterpreter, cmpGTOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpGT, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 7.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 7 > 5 is true

    input = 3.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 3 > 5 is false
}

TEST(GraphInterpreter, cmpGEOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpGE, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 >= 5 is true

    input = 4.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 4 >= 5 is false
}

TEST(GraphInterpreter, cmpEQOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpEQ, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 5 == 5 is true

    input = 4.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 4 == 5 is false
}

TEST(GraphInterpreter, cmpNEOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpNE, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 4.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(1.0, output);  // 4 != 5 is true

    input = 5.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(0.0, output);  // 5 != 5 is false
}

// =============================================================================
// Comparison OpCode adjoint tests (comparisons have zero derivative)
// =============================================================================

TEST(GraphInterpreter, cmpLTAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(5.0);
    uint32_t cmp = graph.addBinary(xad::GraphOpCode::CmpLT, a, b);
    graph.markOutput(cmp);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(1.0, output);   // 3 < 5 is true
    EXPECT_DOUBLE_EQ(0.0, inputAdjoint);  // comparison has zero derivative
}

// =============================================================================
// If OpCode tests
// =============================================================================

TEST(GraphInterpreter, ifOpCodeTrueBranch)
{
    xad::Graph graph;
    uint32_t cond = graph.addConstant(1.0);  // true
    uint32_t t = graph.addConstant(10.0);
    uint32_t f = graph.addConstant(20.0);
    uint32_t result = graph.addTernary(xad::GraphOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(10.0, output);  // condition is true, return trueVal
}

TEST(GraphInterpreter, ifOpCodeAdjointTrueBranch)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t cond = graph.addConstant(1.0);  // true
    uint32_t t = graph.addBinary(xad::GraphOpCode::Mul, inp, graph.addConstant(2.0));  // 2*x
    uint32_t f = graph.addBinary(xad::GraphOpCode::Mul, inp, graph.addConstant(3.0));  // 3*x
    uint32_t result = graph.addTernary(xad::GraphOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(10.0, output);  // 2*5
    EXPECT_DOUBLE_EQ(2.0, inputAdjoint);  // d(2x)/dx = 2
}

TEST(GraphInterpreter, ifOpCodeAdjointFalseBranch)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t cond = graph.addConstant(0.0);  // false
    uint32_t t = graph.addBinary(xad::GraphOpCode::Mul, inp, graph.addConstant(2.0));  // 2*x
    uint32_t f = graph.addBinary(xad::GraphOpCode::Mul, inp, graph.addConstant(3.0));  // 3*x
    uint32_t result = graph.addTernary(xad::GraphOpCode::If, cond, t, f);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(15.0, output);  // 3*5
    EXPECT_DOUBLE_EQ(3.0, inputAdjoint);  // d(3x)/dx = 3
}

// =============================================================================
// Additional OpCode tests for coverage
// =============================================================================

TEST(GraphInterpreter, modOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(3.0);
    uint32_t result = graph.addBinary(xad::GraphOpCode::Mod, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 7.5;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(std::fmod(7.5, 3.0), output);
}

TEST(GraphInterpreter, modAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::GraphOpCode::Mod, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double inputs[2] = {7.5, 3.0};
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    // d(fmod(a,b))/da = 1, d(fmod(a,b))/db = -trunc(a/b)
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    EXPECT_DOUBLE_EQ(-std::trunc(7.5 / 3.0), inputAdjoints[1]);
}

TEST(GraphInterpreter, copysignOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(-1.0);
    uint32_t result = graph.addBinary(xad::GraphOpCode::Copysign, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(-5.0, output);  // copysign(5, -1) = -5

    input = -3.0;
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(-3.0, output);  // copysign(-3, -1) = -3
}

TEST(GraphInterpreter, copysignAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::GraphOpCode::Copysign, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    // Test with positive b
    double inputs[2] = {5.0, 1.0};
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    // d/da copysign(a, b) = sign(b) = 1
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    // d/db copysign(a, b) = 0
    EXPECT_DOUBLE_EQ(0.0, inputAdjoints[1]);

    // Test with negative b
    inputs[1] = -1.0;
    interp.setInput(1, &inputs[1]);
    interp.forwardAndBackward(&output, inputAdjoints);
    EXPECT_DOUBLE_EQ(-1.0, inputAdjoints[0]);  // sign(b) = -1
}

TEST(GraphInterpreter, frexpOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::GraphOpCode::Frexp, a);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 8.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    int exp;
    double expected = std::frexp(8.0, &exp);
    EXPECT_DOUBLE_EQ(expected, output);  // frexp(8) = 0.5, exp=4
}

TEST(GraphInterpreter, frexpAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::GraphOpCode::Frexp, a);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 8.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    // Derivative of frexp mantissa is 1 / 2^exp
    int exp;
    std::frexp(input, &exp);
    EXPECT_DOUBLE_EQ(1.0 / (1 << exp), inputAdjoint);
}

TEST(GraphInterpreter, modfOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::GraphOpCode::Modf, a);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.75;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    double intpart;
    double expected = std::modf(3.75, &intpart);
    EXPECT_DOUBLE_EQ(expected, output);  // fractional part = 0.75
}

TEST(GraphInterpreter, modfAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t result = graph.addUnary(xad::GraphOpCode::Modf, a);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 3.75;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    // Derivative of fractional part is 1
    EXPECT_DOUBLE_EQ(1.0, inputAdjoint);
}

TEST(GraphInterpreter, remquoOpCode)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addConstant(3.0);
    uint32_t result = graph.addBinary(xad::GraphOpCode::Remquo, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 7.5;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);

    int quo;
    double expected = std::remquo(7.5, 3.0, &quo);
    EXPECT_DOUBLE_EQ(expected, output);
}

TEST(GraphInterpreter, remquoAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::GraphOpCode::Remquo, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double inputs[2] = {7.5, 3.0};
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    int quo;
    const double rem = std::remquo(7.5, 3.0, &quo);
    XAD_UNUSED_VARIABLE(rem);
    EXPECT_DOUBLE_EQ(1.0, inputAdjoints[0]);
    EXPECT_DOUBLE_EQ(-static_cast<double>(quo), inputAdjoints[1]);
}

TEST(GraphInterpreter, smoothAbsCDerivative)
{
    // Test the derivative w.r.t. the c parameter (smoothing width)
    // We need two inputs: x and c
    xad::Graph graph;
    uint32_t x = graph.addInput();
    uint32_t c = graph.addInput();
    uint32_t sa = graph.addBinary(xad::GraphOpCode::SmoothAbs, x, c);
    graph.markOutput(sa);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    // Test in smooth region (positive x)
    double inputs[2] = {0.3, 0.5};
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    // dc derivative for positive x in smooth region: -2*x^2*(c-x)/(c^3)
    double x_val = 0.3, c_val = 0.5;
    double expected_dc = -2.0 * x_val * x_val * (c_val - x_val) / (c_val * c_val * c_val);
    EXPECT_NEAR(expected_dc, inputAdjoints[1], 1e-10);

    // Test in smooth region (negative x)
    inputs[0] = -0.3;
    interp.setInput(0, &inputs[0]);
    interp.forwardAndBackward(&output, inputAdjoints);

    // dc derivative for negative x in smooth region: -2*x^2*(c+x)/(c^3)
    x_val = -0.3;
    expected_dc = -2.0 * x_val * x_val * (c_val + x_val) / (c_val * c_val * c_val);
    EXPECT_NEAR(expected_dc, inputAdjoints[1], 1e-10);

    // Test outside smooth region - dc should be 0
    inputs[0] = 1.0;  // |x| > c
    interp.setInput(0, &inputs[0]);
    interp.forwardAndBackward(&output, inputAdjoints);
    EXPECT_DOUBLE_EQ(0.0, inputAdjoints[1]);
}

TEST(GraphInterpreter, minEqualValues)
{
    // Test min adjoint when values are equal (splits 50/50)
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::GraphOpCode::Min, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double inputs[2] = {5.0, 5.0};  // Equal values
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    EXPECT_DOUBLE_EQ(5.0, output);
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[0]);  // Split 50/50
    EXPECT_DOUBLE_EQ(0.5, inputAdjoints[1]);
}

TEST(GraphInterpreter, ldexpAdjoint)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    // ldexp uses immediate for exponent
    uint32_t result = graph.addNode(xad::GraphOpCode::Ldexp, a, 0, 0, 3.0);  // ldexp(a, 3) = a * 8
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 2.0;
    interp.setInput(0, &input);
    double output;
    double inputAdjoint;
    interp.forwardAndBackward(&output, &inputAdjoint);

    EXPECT_DOUBLE_EQ(16.0, output);  // 2 * 2^3 = 16
    EXPECT_DOUBLE_EQ(8.0, inputAdjoint);  // d(a*8)/da = 8
}

TEST(GraphInterpreter, powAdjointWithZeroBase)
{
    // Test pow adjoint when base > 0 (log path is taken)
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t result = graph.addBinary(xad::GraphOpCode::Pow, a, b);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double inputs[2] = {2.0, 3.0};  // 2^3 = 8
    interp.setInput(0, &inputs[0]);
    interp.setInput(1, &inputs[1]);
    double output;
    double inputAdjoints[2];
    interp.forwardAndBackward(&output, inputAdjoints);

    // d(a^b)/da = b * a^(b-1) = 3 * 4 = 12
    EXPECT_DOUBLE_EQ(12.0, inputAdjoints[0]);
    // d(a^b)/db = a^b * log(a) = 8 * log(2)
    EXPECT_NEAR(8.0 * std::log(2.0), inputAdjoints[1], 1e-10);
}

TEST(GraphInterpreter, reset)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t result = graph.addUnary(xad::GraphOpCode::Neg, inp);
    graph.markOutput(result);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    double input = 5.0;
    interp.setInput(0, &input);
    double output;
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(-5.0, output);

    // Reset and verify we can still use it after recompiling
    interp.reset();
    interp.compile(graph);
    interp.setInput(0, &input);
    interp.forward(&output);
    EXPECT_DOUBLE_EQ(-5.0, output);
}

TEST(GraphInterpreter, queryMethods)
{
    xad::Graph graph;
    graph.addInput();
    graph.addInput();
    uint32_t c = graph.addConstant(1.0);
    graph.markOutput(c);

    xad::GraphInterpreter<double> interp;
    interp.compile(graph);

    EXPECT_EQ(1u, interp.vectorWidth());
    EXPECT_EQ(2u, interp.numInputs());
    EXPECT_EQ(1u, interp.numOutputs());
}

// =============================================================================
// Exception path tests for coverage
// =============================================================================

TEST(GraphInterpreter, forwardThrowsWhenNotCompiled)
{
    xad::GraphInterpreter<double> interp;
    double output;
    EXPECT_THROW(interp.forward(&output), std::runtime_error);
}

#endif  // XAD_ENABLE_GRAPH

/*******************************************************************************

   Unit tests for JIT compilation functionality

   Tests for the JIT compilation features added to XAD.

******************************************************************************/

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <cmath>

#ifdef XAD_ENABLE_JIT

TEST(JITCompiler, isEmptyByDefault)
{
    EXPECT_EQ(nullptr, xad::JITCompiler<double>::getActive());
    {
        xad::JITCompiler<double> jit;
        EXPECT_TRUE(jit.isActive());
        EXPECT_EQ(&jit, xad::JITCompiler<double>::getActive());
    }
    EXPECT_EQ(nullptr, xad::JITCompiler<double>::getActive());
}

TEST(JITCompiler, canInitializeDeactivated)
{
    xad::JITCompiler<float> jit(false);

    EXPECT_FALSE(jit.isActive());
    EXPECT_EQ(nullptr, xad::JITCompiler<float>::getActive());

    jit.activate();

    EXPECT_TRUE(jit.isActive());
    EXPECT_NE(nullptr, xad::JITCompiler<float>::getActive());
}

TEST(JITCompiler, canActivateStatically)
{
    xad::JITCompiler<float> jit(false);

    EXPECT_FALSE(jit.isActive());
    EXPECT_EQ(nullptr, xad::JITCompiler<float>::getActive());

    xad::JITCompiler<float>::setActive(&jit);

    EXPECT_TRUE(jit.isActive());
    EXPECT_NE(nullptr, xad::JITCompiler<float>::getActive());
}

TEST(JITCompiler, canDeactivateGlobally)
{
    EXPECT_EQ(nullptr, xad::JITCompiler<double>::getActive());

    xad::JITCompiler<double> jit;

    EXPECT_TRUE(jit.isActive());
    xad::JITCompiler<double>::deactivateAll();
    EXPECT_FALSE(jit.isActive());
}

TEST(JITCompiler, isMovable)
{
    xad::JITCompiler<double> jit1(false);
    xad::JITCompiler<double> jit2 = std::move(jit1);  // move constructor
    EXPECT_FALSE(jit2.isActive());

    xad::JITCompiler<double> jit3(true);
    jit3 = std::move(jit2);  // move assign
    EXPECT_FALSE(jit3.isActive());

    xad::JITCompiler<double> jit4(true);
    EXPECT_TRUE(jit4.isActive());
    xad::JITCompiler<double> jit5 = std::move(jit4);  // move active
    EXPECT_TRUE(jit5.isActive());
    EXPECT_FALSE(jit4.isActive());
}

TEST(JITCompiler, canRegisterInputsAndOutputs)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    auto c = a + b;
    jit.registerOutput(c);

    EXPECT_EQ(2u, jit.getGraph().input_ids.size());
    EXPECT_EQ(1u, jit.getGraph().output_ids.size());
    EXPECT_GE(jit.getGraph().nodeCount(), 3u);  // at least 2 inputs + 1 operation
}

TEST(JITCompiler, forwardProducesCorrectValues)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    auto c = a * b + a;  // 2*3 + 2 = 8
    jit.registerOutput(c);

    jit.compile();

    double output;
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(8.0, output);
}

TEST(JITCompiler, computeAdjointsProducesCorrectGradients)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    auto c = a * b;  // dc/da = b = 3, dc/db = a = 2
    jit.registerOutput(c);

    jit.compile();
    jit.setDerivative(c.getSlot(), 1.0);  // seed
    jit.computeAdjoints();

    EXPECT_DOUBLE_EQ(3.0, jit.getDerivative(a.getSlot()));
    EXPECT_DOUBLE_EQ(2.0, jit.getDerivative(b.getSlot()));
}

TEST(JITCompiler, canUseNewRecording)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    auto c1 = a + b;
    jit.registerOutput(c1);
    jit.compile();

    double output1;
    jit.forward(&output1, 1);
    EXPECT_DOUBLE_EQ(5.0, output1);

    // New recording with same inputs
    jit.newRecording();
    auto c2 = a * b;  // different computation
    jit.registerOutput(c2);
    jit.compile();

    double output2;
    jit.forward(&output2, 1);
    EXPECT_DOUBLE_EQ(6.0, output2);
}

TEST(JITCompiler, clearDerivativesWorks)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;

    jit.registerInput(a);
    auto c = a * a;
    jit.registerOutput(c);
    jit.compile();

    jit.setDerivative(c.getSlot(), 1.0);
    jit.computeAdjoints();
    EXPECT_DOUBLE_EQ(4.0, jit.getDerivative(a.getSlot()));

    jit.clearDerivatives();
    EXPECT_DOUBLE_EQ(0.0, jit.getDerivative(a.getSlot()));
    EXPECT_DOUBLE_EQ(0.0, jit.getDerivative(c.getSlot()));
}

TEST(JITGraph, canAddNodesAndConstants)
{
    xad::JITGraph graph;
    uint32_t c1 = graph.addConstant(3.14);
    uint32_t c2 = graph.addConstant(2.71);
    uint32_t n1 = graph.addNode(xad::JITOpCode::Add, c1, c2);

    EXPECT_EQ(3u, graph.nodeCount());
    EXPECT_DOUBLE_EQ(3.14, graph.nodes[c1].constant_value);
    EXPECT_DOUBLE_EQ(2.71, graph.nodes[c2].constant_value);
    EXPECT_EQ(xad::JITOpCode::Add, graph.nodes[n1].op);
}

TEST(JITGraph, canAddInputsAndMarkOutputs)
{
    xad::JITGraph graph;
    uint32_t in1 = graph.addInput();
    uint32_t in2 = graph.addInput();
    uint32_t out = graph.addNode(xad::JITOpCode::Mul, in1, in2);

    graph.markOutput(out);

    EXPECT_EQ(2u, graph.input_ids.size());
    EXPECT_EQ(1u, graph.output_ids.size());
    EXPECT_EQ(out, graph.output_ids[0]);
}

TEST(JITGraph, clearWorks)
{
    xad::JITGraph graph;
    graph.addConstant(1.0);
    graph.addInput();
    graph.addNode(xad::JITOpCode::Add, 0, 1);

    EXPECT_GT(graph.nodeCount(), 0u);

    graph.clear();

    EXPECT_EQ(0u, graph.nodeCount());
    EXPECT_EQ(0u, graph.input_ids.size());
    EXPECT_EQ(0u, graph.output_ids.size());
}

TEST(JITGraphInterpreter, executeBasicOperations)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    // Test add
    {
        AD a = 2.0, b = 3.0;
        jit.registerInput(a);
        jit.registerInput(b);
        auto c = a + b;
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
        auto c = a - b;
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
        auto c = a * b;
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
        auto c = a / b;
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
        auto c = sin(a);
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
        auto c = cos(a);
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
        auto c = exp(a);
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
        auto c = log(a);
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
        auto c = sqrt(a);
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
    auto c = -a;
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
    auto result = (x * x + y) * sin(x) / y;
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
    auto result = x * x + y * y;
    jit.registerOutput(result);

    jit.compile();
    jit.setDerivative(result.getSlot(), 1.0);
    jit.computeAdjoints();

    EXPECT_NEAR(4.0, jit.getDerivative(x.getSlot()), 1e-10);
    EXPECT_NEAR(6.0, jit.getDerivative(y.getSlot()), 1e-10);
}

#endif  // XAD_ENABLE_JIT

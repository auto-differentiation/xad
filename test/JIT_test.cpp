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

TEST(JITCompiler, canSetBackend)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    AD c = a * b;
    jit.registerOutput(c);
    jit.compile();

    double output;
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(6.0, output);

    // Replace backend with a new interpreter
    jit.setBackend(std::unique_ptr<xad::IJITBackend>(new xad::JITGraphInterpreter()));

    // After setBackend, need to recompile since backend was reset
    jit.compile();
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(6.0, output);
}

TEST(JITCompiler, constructorWithExplicitBackend)
{
    auto backend = std::unique_ptr<xad::IJITBackend>(new xad::JITGraphInterpreter());
    xad::JITCompiler<double> jit(std::move(backend), true);

    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    AD c = a + b;
    jit.registerOutput(c);
    jit.compile();

    double output;
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(5.0, output);
}

TEST(JITCompiler, canRegisterInputsAndOutputs)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    jit.registerInput(a);
    jit.registerInput(b);

    AD c = a + b;
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

    AD c = a * b + a;  // 2*3 + 2 = 8
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

    AD c = a * b;  // dc/da = b = 3, dc/db = a = 2
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

    AD c1 = a + b;
    jit.registerOutput(c1);
    jit.compile();

    double output1;
    jit.forward(&output1, 1);
    EXPECT_DOUBLE_EQ(5.0, output1);

    // New recording with same inputs
    jit.newRecording();
    AD c2 = a * b;  // different computation
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
    AD c = a * a;
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
    EXPECT_DOUBLE_EQ(3.14, graph.getConstantValue(c1));
    EXPECT_DOUBLE_EQ(2.71, graph.getConstantValue(c2));
    EXPECT_EQ(xad::JITOpCode::Add, graph.getOpCode(n1));
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

TEST(ABool, defaultConstructor)
{
    xad::ABool<double> ab;
    EXPECT_FALSE(ab.passive());
    EXPECT_FALSE(ab.hasSlot());
    // Compare with local copy to avoid ODR-use of static constexpr member
    constexpr auto invalid_slot = xad::ABool<double>::INVALID_SLOT;
    EXPECT_EQ(invalid_slot, ab.slot());
}

TEST(ABool, constructorFromBool)
{
    xad::ABool<double> ab_true(true);
    xad::ABool<double> ab_false(false);

    EXPECT_TRUE(ab_true.passive());
    EXPECT_FALSE(ab_false.passive());
    EXPECT_FALSE(ab_true.hasSlot());
    EXPECT_FALSE(ab_false.hasSlot());
}

TEST(ABool, constructorWithSlot)
{
    xad::ABool<double> ab(42, true);

    EXPECT_TRUE(ab.passive());
    EXPECT_TRUE(ab.hasSlot());
    EXPECT_EQ(42u, ab.slot());
}

TEST(ABool, implicitBoolConversion)
{
    xad::ABool<double> ab_true(true);
    xad::ABool<double> ab_false(false);

    // Test implicit conversion
    if (ab_true)
        EXPECT_TRUE(true);
    else
        FAIL() << "ABool(true) should convert to true";

    if (ab_false)
        FAIL() << "ABool(false) should convert to false";
    else
        EXPECT_TRUE(true);
}

TEST(ABool, ifWithoutJIT)
{
    // Without JIT active, If should use passive value
    using AD = xad::AReal<double, 1>;
    AD trueVal(10.0);
    AD falseVal(20.0);

    xad::ABool<double> cond_true(true);
    xad::ABool<double> cond_false(false);

    AD result_true = cond_true.If(trueVal, falseVal);
    AD result_false = cond_false.If(trueVal, falseVal);

    EXPECT_DOUBLE_EQ(10.0, xad::value(result_true));
    EXPECT_DOUBLE_EQ(20.0, xad::value(result_false));
}

TEST(ABool, staticIfWithoutJIT)
{
    using AD = xad::AReal<double, 1>;
    AD trueVal(10.0);
    AD falseVal(20.0);

    xad::ABool<double> cond_true(true);
    xad::ABool<double> cond_false(false);

    AD result_true = xad::ABool<double>::If(cond_true, trueVal, falseVal);
    AD result_false = xad::ABool<double>::If(cond_false, trueVal, falseVal);

    EXPECT_DOUBLE_EQ(10.0, xad::value(result_true));
    EXPECT_DOUBLE_EQ(20.0, xad::value(result_false));
}

TEST(ABool, lessComparison)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    AD b = 3.0;
    jit.registerInput(a);
    jit.registerInput(b);

    auto cond = xad::less(a, b);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_TRUE(cond.hasSlot());  // JIT is active, so slot should be set
}

TEST(ABool, lessComparisonWithScalar)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    jit.registerInput(a);

    auto cond = xad::less(a, 3.0);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, greaterComparison)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    AD b = 3.0;
    jit.registerInput(a);
    jit.registerInput(b);

    auto cond = xad::greater(a, b);
    EXPECT_TRUE(cond.passive());  // 5 > 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, greaterComparisonWithScalar)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    jit.registerInput(a);

    auto cond = xad::greater(a, 3.0);
    EXPECT_TRUE(cond.passive());
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, lessEqualComparison)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 3.0;
    AD b = 3.0;
    jit.registerInput(a);
    jit.registerInput(b);

    auto cond = xad::lessEqual(a, b);
    EXPECT_TRUE(cond.passive());  // 3 <= 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, lessEqualComparisonWithScalar)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 3.0;
    jit.registerInput(a);

    auto cond = xad::lessEqual(a, 3.0);
    EXPECT_TRUE(cond.passive());
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, greaterEqualComparison)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    AD b = 3.0;
    jit.registerInput(a);
    jit.registerInput(b);

    auto cond = xad::greaterEqual(a, b);
    EXPECT_TRUE(cond.passive());  // 5 >= 3 is true
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, greaterEqualComparisonWithScalar)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 5.0;
    jit.registerInput(a);

    auto cond = xad::greaterEqual(a, 3.0);
    EXPECT_TRUE(cond.passive());
    EXPECT_TRUE(cond.hasSlot());
}

TEST(ABool, ifWithJITRecording)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    jit.registerInput(x);
    jit.newRecording();

    AD trueVal = x * 2.0;   // 4.0
    AD falseVal = x * 3.0;  // 6.0

    auto cond = xad::less(x, 5.0);  // true for x=2
    AD result = cond.If(trueVal, falseVal);
    jit.registerOutput(result);

    jit.compile();
    double output;
    jit.forward(&output, 1);

    EXPECT_DOUBLE_EQ(4.0, output);  // x < 5, so trueVal = 2*2 = 4
}

TEST(ABool, ifWithJITRecordingFalseBranch)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 10.0;
    jit.registerInput(x);
    jit.newRecording();

    AD trueVal = x * 2.0;   // 20.0
    AD falseVal = x * 3.0;  // 30.0

    auto cond = xad::less(x, 5.0);  // false for x=10
    AD result = cond.If(trueVal, falseVal);
    jit.registerOutput(result);

    jit.compile();
    double output;
    jit.forward(&output, 1);

    EXPECT_DOUBLE_EQ(30.0, output);  // x >= 5, so falseVal = 10*3 = 30
}

TEST(ABool, ifDerivativeTrueBranch)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    jit.registerInput(x);
    jit.newRecording();

    AD trueVal = x * x;     // x^2, derivative = 2x
    AD falseVal = x * 3.0;  // 3x, derivative = 3

    auto cond = xad::less(x, 5.0);  // true for x=2
    AD result = cond.If(trueVal, falseVal);
    jit.registerOutput(result);

    jit.compile();
    jit.setDerivative(result.getSlot(), 1.0);
    jit.computeAdjoints();

    // Since x=2 < 5, we take the true branch (x^2)
    // d(x^2)/dx = 2x = 4
    EXPECT_NEAR(4.0, jit.getDerivative(x.getSlot()), 1e-10);
}

TEST(ABool, ifDerivativeFalseBranch)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 10.0;
    jit.registerInput(x);
    jit.newRecording();

    AD trueVal = x * x;     // x^2, derivative = 2x
    AD falseVal = x * 3.0;  // 3x, derivative = 3

    auto cond = xad::less(x, 5.0);  // false for x=10
    AD result = cond.If(trueVal, falseVal);
    jit.registerOutput(result);

    jit.compile();
    jit.setDerivative(result.getSlot(), 1.0);
    jit.computeAdjoints();

    // Since x=10 >= 5, we take the false branch (3x)
    // d(3x)/dx = 3
    EXPECT_NEAR(3.0, jit.getDerivative(x.getSlot()), 1e-10);
}

TEST(ABool, ifWithConstantOperands)
{
    // Test ABool::If when operands don't have slots (need to be recorded as constants)
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    jit.registerInput(x);

    // Create a condition that has a slot
    auto cond = xad::less(x, 5.0);  // true for x=2

    // Use constant values (not from graph operations) for branches
    // These AD values won't have slots, so ABool::If should record them as constants
    AD trueVal(100.0);   // No slot - just a constant
    AD falseVal(200.0);  // No slot - just a constant

    AD result = cond.If(trueVal, falseVal);
    jit.registerOutput(result);

    jit.compile();
    double output;
    jit.forward(&output, 1);

    EXPECT_DOUBLE_EQ(100.0, output);  // x < 5, so trueVal = 100
}

TEST(ABool, comparisonWithoutJIT)
{
    // Test comparison functions when JIT is NOT active
    // Should return ABool with passive value but no slot
    using AD = xad::AReal<double, 1>;

    AD a(2.0);
    AD b(3.0);

    // No JIT active - comparisons should work but not have slots
    auto cond = xad::less(a, b);
    EXPECT_TRUE(cond.passive());  // 2 < 3 is true
    EXPECT_FALSE(cond.hasSlot()); // No JIT, so no slot

    auto cond2 = xad::greater(a, 1.0);
    EXPECT_TRUE(cond2.passive());  // 2 > 1 is true
    EXPECT_FALSE(cond2.hasSlot());
}

TEST(ABool, comparisonWithInvalidSlotOperands)
{
    // Test comparison when AReal operands don't have slots yet
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    // Create AD values that are NOT registered as inputs (no slots)
    AD a(2.0);  // No slot
    AD b(3.0);  // No slot

    // Compare should still work - should record constants for the operands
    auto cond = xad::less(a, b);
    EXPECT_TRUE(cond.passive());
    EXPECT_TRUE(cond.hasSlot());  // JIT is active, so slot should be created
}

// =============================================================================
// Additional JITCompiler tests
// =============================================================================

TEST(JITCompiler, registerInputsVector)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    std::vector<AD> inputs = {1.0, 2.0, 3.0};
    jit.registerInputs(inputs);

    EXPECT_EQ(3u, jit.getGraph().input_ids.size());
}

TEST(JITCompiler, registerInputsRange)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    std::vector<AD> inputs = {1.0, 2.0, 3.0, 4.0};
    jit.registerInputs(inputs.begin(), inputs.begin() + 2);

    EXPECT_EQ(2u, jit.getGraph().input_ids.size());
}

TEST(JITCompiler, registerOutputsVector)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 1.0, b = 2.0;
    jit.registerInput(a);
    jit.registerInput(b);

    std::vector<AD> outputs;
    outputs.push_back(a + b);
    outputs.push_back(a * b);

    jit.registerOutputs(outputs);

    EXPECT_EQ(2u, jit.getGraph().output_ids.size());
}

TEST(JITCompiler, registerOutputsRange)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 1.0, b = 2.0;
    jit.registerInput(a);
    jit.registerInput(b);

    std::vector<AD> outputs;
    outputs.push_back(a + b);
    outputs.push_back(a * b);
    outputs.push_back(a - b);

    jit.registerOutputs(outputs.begin(), outputs.begin() + 2);

    EXPECT_EQ(2u, jit.getGraph().output_ids.size());
}

TEST(JITCompiler, recordNodeAndConstant)
{
    xad::JITCompiler<double> jit;

    uint32_t c1 = jit.recordConstant(5.0);
    uint32_t c2 = jit.recordConstant(3.0);
    uint32_t n = jit.recordNode(xad::JITOpCode::Add, c1, c2);

    EXPECT_EQ(xad::JITOpCode::Constant, jit.getGraph().getOpCode(c1));
    EXPECT_EQ(xad::JITOpCode::Constant, jit.getGraph().getOpCode(c2));
    EXPECT_EQ(xad::JITOpCode::Add, jit.getGraph().getOpCode(n));
}

TEST(JITCompiler, registerVariable)
{
    xad::JITCompiler<double> jit;

    auto slot1 = jit.registerVariable();
    EXPECT_EQ(0u, slot1);  // First variable gets slot 0

    jit.recordConstant(1.0);  // Add a node
    auto slot2 = jit.registerVariable();
    EXPECT_EQ(1u, slot2);  // Second variable gets slot 1
}

TEST(JITCompiler, setActiveThrowsWhenAlreadyActive)
{
    xad::JITCompiler<double> jit1;  // Activates itself

    // Trying to activate another JIT should throw
    EXPECT_THROW(
        {
            xad::JITCompiler<double> jit2;  // Tries to activate, should throw
        },
        xad::OutOfRange);
}

TEST(JITCompiler, forwardThrowsOnOutputMismatch)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD x = 2.0;
    jit.registerInput(x);
    AD y = x * x;
    jit.registerOutput(y);
    jit.compile();

    double outputs[2];  // Wrong size - we only have 1 output
    EXPECT_THROW(jit.forward(outputs, 2), xad::OutOfRange);
}

TEST(JITCompiler, clearAll)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    jit.registerInput(a);
    AD c = a * a;
    jit.registerOutput(c);

    EXPECT_GT(jit.getGraph().nodeCount(), 0u);

    jit.clearAll();

    EXPECT_EQ(0u, jit.getGraph().nodeCount());
    EXPECT_EQ(0u, jit.getGraph().input_ids.size());
    EXPECT_EQ(0u, jit.getGraph().output_ids.size());
}

TEST(JITCompiler, getMemory)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    jit.registerInput(a);
    AD c = a * a;
    jit.registerOutput(c);

    std::size_t mem = jit.getMemory();
    EXPECT_GT(mem, 0u);
}

TEST(JITCompiler, getPosition)
{
    xad::JITCompiler<double> jit;
    using AD = xad::AReal<double, 1>;

    auto pos0 = jit.getPosition();
    EXPECT_EQ(0u, pos0);

    AD a = 2.0;
    jit.registerInput(a);

    auto pos1 = jit.getPosition();
    EXPECT_GT(pos1, pos0);

    AD c = a * a;
    auto pos2 = jit.getPosition();
    EXPECT_GT(pos2, pos1);
}

TEST(JITCompiler, derivativeNonConstAccess)
{
    xad::JITCompiler<double> jit;

    // Access derivative for a slot that doesn't exist yet
    auto& deriv = jit.derivative(10);
    deriv = 42.0;

    EXPECT_DOUBLE_EQ(42.0, jit.getDerivative(10));
}

TEST(JITCompiler, derivativeConstAccessOutOfRange)
{
    xad::JITCompiler<double> jit;

    // Const access to out-of-range slot should return zero
    const auto& jit_const = jit;
    const auto& deriv = jit_const.derivative(999);

    EXPECT_DOUBLE_EQ(0.0, deriv);
}

TEST(JITCompiler, floatScalarOperations)
{
    // Test JIT with float scalar type to exercise getNestedDoubleValue(float)
    xad::JITCompiler<float> jit;
    xad::AReal<float> x = 2.0f;
    jit.registerInput(x);

    // Scalar multiplication: uses scalar_prod_op which calls getScalarConstant -> getNestedDoubleValue(float)
    xad::AReal<float> y = x * 3.0f;
    jit.registerOutput(y);

    jit.compile();

    double output;  // JIT always uses double internally
    jit.forward(&output, 1);

    EXPECT_DOUBLE_EQ(6.0, output);
}


// =============================================================================
// Additional JITGraph tests
// =============================================================================

TEST(JITGraph, empty)
{
    xad::JITGraph graph;
    EXPECT_TRUE(graph.empty());

    graph.addInput();
    EXPECT_FALSE(graph.empty());
}

TEST(JITGraph, reserve)
{
    xad::JITGraph graph;
    graph.reserve(100);
    // Just verify it doesn't crash - capacity is implementation detail
    EXPECT_TRUE(graph.empty());
}

TEST(JITGraph, addUnary)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t neg = graph.addUnary(xad::JITOpCode::Neg, inp);

    EXPECT_EQ(xad::JITOpCode::Neg, graph.getOpCode(neg));
}

TEST(JITGraph, addBinary)
{
    xad::JITGraph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t sum = graph.addBinary(xad::JITOpCode::Add, a, b);

    EXPECT_EQ(xad::JITOpCode::Add, graph.getOpCode(sum));
}

TEST(JITGraph, addTernary)
{
    xad::JITGraph graph;
    uint32_t cond = graph.addInput();
    uint32_t t = graph.addInput();
    uint32_t f = graph.addInput();
    uint32_t result = graph.addTernary(xad::JITOpCode::If, cond, t, f);

    EXPECT_EQ(xad::JITOpCode::If, graph.getOpCode(result));
}

TEST(JITGraph, isInput)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(1.0);

    EXPECT_TRUE(graph.isInput(inp));
    EXPECT_FALSE(graph.isInput(c));
}

TEST(JITGraph, isConstant)
{
    xad::JITGraph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(1.0);

    EXPECT_FALSE(graph.isConstant(inp));
    EXPECT_TRUE(graph.isConstant(c));
}

TEST(JITGraph, constantPoolDeduplication)
{
    xad::JITGraph graph;
    uint32_t c1 = graph.addConstant(3.14);
    uint32_t c2 = graph.addConstant(3.14);  // Same value - should reuse pool entry

    // Both should give the same constant value
    EXPECT_DOUBLE_EQ(graph.getConstantValue(c1), graph.getConstantValue(c2));
    // Pool should have only one entry since we added the same value twice
    EXPECT_EQ(1u, graph.const_pool.size());

    // Adding a different value should add to the pool
    uint32_t c3 = graph.addConstant(2.71);
    EXPECT_DOUBLE_EQ(2.71, graph.getConstantValue(c3));
    EXPECT_EQ(2u, graph.const_pool.size());  // Now two constants in pool
}

// =============================================================================
// OpCode tests for Square, Recip, SmoothAbs
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
    std::remquo(7.5, 3.0, &quo);
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

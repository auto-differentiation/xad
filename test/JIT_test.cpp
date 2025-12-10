#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Test Functions
// ============================================================================

// f1: Simple linear function
// f(x) = x * 3 + 2, f'(x) = 3
template <class T>
T f1(const T& x)
{
    return x * 3.0 + 2.0;
}

// f2: Function with supported math operations
// Uses: sin, cos, exp, log, sqrt, abs (Forge-compatible)
template <class T>
T f2(const T& x)
{
    using std::sin; using std::cos; using std::exp; using std::log;
    using std::sqrt; using std::abs;

    T result = sin(x) + cos(x) * 2.0;
    result = result + exp(x / 10.0) + log(x + 5.0);
    result = result + sqrt(x + 1.0);
    result = result + abs(x - 1.0) + x * x;
    result = result + 1.0 / (x + 2.0);
    return result;
}

// f3: Branching function to demonstrate JIT graph reuse
// if (x < 2) return 2*x else return 10*x
// When recorded with x=1, JIT captures the first branch (2*x)
// and will use it even for x=3, showing graph reuse behavior

// Helper to get value for both double and AD types
inline double getValue(double x) { return x; }
template <class T>
double getValue(const T& x) { return value(x); }

template <class T>
T f3(const T& x)
{
    if (getValue(x) < 2.0)
        return 2.0 * x;
    else
        return 10.0 * x;
}

// f3ABool: Same as f3 but using ABool::If for trackable branches
// This allows JIT to record both branches and select at runtime
// f(x) = if (x < 2) 2*x else 10*x
// For x=1: f(1)=2, f'(1)=2
// For x=3: f(3)=30, f'(3)=10
xad::AD f3ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

// Plain double version for comparison
double f3ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

// ============================================================================
// Test Infrastructure
// ============================================================================

struct TestCase
{
    std::string name;
    std::string formula;
    std::function<double(double)> func_double;
    std::function<xad::AD(const xad::AD&)> func_ad;
    std::vector<double> inputs;
    bool expectJITMatch = true;  // false for branching functions where JIT intentionally differs
};

// ============================================================================
// Tests
// ============================================================================

class JITTest : public ::testing::Test
{
  protected:
    std::vector<TestCase> testCases;

    void SetUp() override
    {
        testCases = {
            {"f1", "x * 3 + 2", f1<double>, f1<xad::AD>, {2.0, 0.5, -1.0}, true},
            {"f2", "sin(x) + cos(x)*2 + exp(x/10) + log(x+5) + sqrt(x+1) + abs(x-1) + x*x + 1/(x+2)", f2<double>, f2<xad::AD>, {2.0, 0.5}, true},
            {"f3", "if (x < 2) 2*x else 10*x  [JIT uses recorded branch - EXPECT MISMATCH]", f3<double>, f3<xad::AD>, {1.0, 3.0}, false},
            {"f3ABool", "ABool::If(x < 2, 2*x, 10*x)  [JIT tracks branches - SHOULD MATCH]", f3ABool_double, f3ABool, {1.0, 3.0}, true},
        };
    }
};

TEST_F(JITTest, TapeVsJIT)
{
    for (const auto& tc : testCases)
    {
        std::cout << tc.name << "(x) = " << tc.formula << std::endl;

        std::vector<double> tapeOutputs, tapeDerivatives;
        std::vector<double> jitOutputs, jitDerivatives;

        // Compute all with Tape
        {
            xad::Tape<double> tape;
            for (double input : tc.inputs)
            {
                xad::AD x(input);
                tape.registerInput(x);
                tape.newRecording();
                xad::AD y = tc.func_ad(x);
                tape.registerOutput(y);
                derivative(y) = 1.0;
                tape.computeAdjoints();
                tapeOutputs.push_back(value(y));
                tapeDerivatives.push_back(derivative(x));
                tape.clearAll();
            }
        }

        // Compute all with JIT (record once, reuse for all inputs)
        {
            xad::JITCompiler<double> jit;

            // Record graph with first input
            xad::AD x(tc.inputs[0]);
            jit.registerInput(x);
            jit.newRecording();
            xad::AD y = tc.func_ad(x);
            jit.registerOutput(y);

            // Reuse graph for all inputs
            for (double input : tc.inputs)
            {
                value(x) = input;

                // Forward pass
                double output;
                jit.forward(&output, 1);
                jitOutputs.push_back(output);

                // Backward pass
                jit.clearDerivatives();
                derivative(y) = 1.0;
                jit.computeAdjoints();
                jitDerivatives.push_back(derivative(x));
            }
        }

        // Compare and print results
        for (std::size_t i = 0; i < tc.inputs.size(); ++i)
        {
            double input = tc.inputs[i];
            double expectedOutput = tc.func_double(input);

            std::cout << "  x=" << input << ": "
                      << "outTape=" << tapeOutputs[i] << ", "
                      << "outJIT=" << jitOutputs[i] << ", "
                      << "derivTape=" << tapeDerivatives[i] << ", "
                      << "derivJIT=" << jitDerivatives[i] << std::endl;

            EXPECT_NEAR(expectedOutput, tapeOutputs[i], 1e-10) << tc.name << " tape output at x=" << input;
            if (tc.expectJITMatch)
            {
                EXPECT_NEAR(expectedOutput, jitOutputs[i], 1e-10) << tc.name << " JIT output at x=" << input;
                EXPECT_NEAR(tapeDerivatives[i], jitDerivatives[i], 1e-10) << tc.name << " derivatives at x=" << input;
            }
        }
        std::cout << std::endl;
    }
}

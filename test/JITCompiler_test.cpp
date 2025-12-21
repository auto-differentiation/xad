/*******************************************************************************

   Unit tests for JITCompiler

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
    xad::JITCompiler<double> jit(false);

    EXPECT_FALSE(jit.isActive());
    EXPECT_EQ(nullptr, xad::JITCompiler<double>::getActive());

    jit.activate();

    EXPECT_TRUE(jit.isActive());
    EXPECT_NE(nullptr, xad::JITCompiler<double>::getActive());
}

TEST(JITCompiler, canActivateStatically)
{
    xad::JITCompiler<double> jit(false);

    EXPECT_FALSE(jit.isActive());
    EXPECT_EQ(nullptr, xad::JITCompiler<double>::getActive());

    xad::JITCompiler<double>::setActive(&jit);

    EXPECT_TRUE(jit.isActive());
    EXPECT_NE(nullptr, xad::JITCompiler<double>::getActive());
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

TEST(JITCompiler, moveAssignFromActiveTransfersActivePointer)
{
    // Cover the move-assignment branch where the source compiler is active.
    xad::JITCompiler<double> src(true);
    EXPECT_TRUE(src.isActive());
    EXPECT_EQ(&src, xad::JITCompiler<double>::getActive());

    xad::JITCompiler<double> dst(false);
    EXPECT_FALSE(dst.isActive());

    dst = std::move(src);
    EXPECT_TRUE(dst.isActive());
    EXPECT_EQ(&dst, xad::JITCompiler<double>::getActive());
    EXPECT_FALSE(src.isActive());
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
    jit.setBackend(std::unique_ptr<xad::JITBackend>(new xad::JITGraphInterpreter()));

    // After setBackend, need to recompile since backend was reset
    jit.compile();
    jit.forward(&output, 1);
    EXPECT_DOUBLE_EQ(6.0, output);
}

TEST(JITCompiler, constructorWithExplicitBackend)
{
    auto backend = std::unique_ptr<xad::JITBackend>(new xad::JITGraphInterpreter());
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


TEST(JITAReal, derivativeFallbackUsesJITWhenNoTape)
{
    // Cover Literals.hpp JIT-derivative fallback paths:
    // - const derivative() returns a zero ref when slot is INVALID and JIT is active
    // - non-const derivative() allocates a slot when slot is INVALID and JIT is active
    xad::JITCompiler<double, 1> jit;
    using AD = xad::AReal<double, 1>;

    const AD c(2.0);  // no tape, no slot
    EXPECT_EQ(AD::INVALID_SLOT, c.getSlot());
    EXPECT_DOUBLE_EQ(0.0, c.derivative());
    EXPECT_EQ(AD::INVALID_SLOT, c.getSlot()); // const path must not allocate a slot

    AD x(3.0);  // no tape, no slot
    EXPECT_EQ(AD::INVALID_SLOT, x.getSlot());
    auto& dx = x.derivative(); // allocates slot in JIT
    dx = 7.0;
    EXPECT_NE(AD::INVALID_SLOT, x.getSlot());
    EXPECT_DOUBLE_EQ(7.0, jit.getDerivative(x.getSlot()));
}

TEST(JITAReal, copyCtorCopiesSlotWhenNoTapeAndJitActive)
{
    xad::JITCompiler<double, 1> jit;
    using AD = xad::AReal<double, 1>;

    AD x(3.0);
    (void)x.derivative();  // allocate slot in JIT
    EXPECT_NE(AD::INVALID_SLOT, x.getSlot());

    AD y(x);  // copy-ctor should copy the slot when JIT is active and no tape is active
    EXPECT_EQ(x.getSlot(), y.getSlot());
    EXPECT_DOUBLE_EQ(x.getValue(), y.getValue());
}

TEST(JITAReal, copyAssignCopiesSlotWhenNoTapeAndJitActive)
{
    xad::JITCompiler<double, 1> jit;
    using AD = xad::AReal<double, 1>;

    AD x(4.0);
    (void)x.derivative();  // allocate slot in JIT
    EXPECT_NE(AD::INVALID_SLOT, x.getSlot());

    AD y(1.0);
    EXPECT_EQ(AD::INVALID_SLOT, y.getSlot());

    y = x;  // copy-assign should copy the slot when JIT is active and no tape is active
    EXPECT_EQ(x.getSlot(), y.getSlot());
    EXPECT_DOUBLE_EQ(x.getValue(), y.getValue());
}

TEST(JITAReal, constantExpressionDoesNotRecordWhenNoTapeAndJitActive)
{
    // Exercise the "expr.shouldRecord() == false" branch in the JIT recording path.
    xad::JITCompiler<double, 1> jit;
    using AD = xad::AReal<double, 1>;

    AD c(2.0);  // no slot
    EXPECT_EQ(AD::INVALID_SLOT, c.getSlot());

    AD y = c + 1.0;  // pure constant expression => should not record any JIT node
    EXPECT_EQ(AD::INVALID_SLOT, y.getSlot());
    EXPECT_DOUBLE_EQ(3.0, y.getValue());
}

TEST(JITAReal, vectorModeDoesNotUseJitFallback)
{
    // JIT is intentionally scalar-only; vector AD should not attempt JIT fallback.
    xad::JITCompiler<double, 1> jit;

    using ADV = xad::AReal<double, 2>;
    const ADV v;  // no tape

    EXPECT_THROW((void)v.derivative(), xad::NoTapeException);
}


// =============================================================================
// Additional JITGraph tests
// =============================================================================


#endif  // XAD_ENABLE_JIT

/*******************************************************************************

   Unit tests for GraphRecorder

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

TEST(GraphRecorder, isEmptyByDefault)
{
    EXPECT_EQ(nullptr, xad::GraphRecorder<double>::getActive());
    {
        xad::GraphRecorder<double> recorder;
        EXPECT_TRUE(recorder.isActive());
        EXPECT_EQ(&recorder, xad::GraphRecorder<double>::getActive());
    }
    EXPECT_EQ(nullptr, xad::GraphRecorder<double>::getActive());
}

TEST(GraphRecorder, canInitializeDeactivated)
{
    xad::GraphRecorder<float> recorder(false);

    EXPECT_FALSE(recorder.isActive());
    EXPECT_EQ(nullptr, xad::GraphRecorder<float>::getActive());

    recorder.activate();

    EXPECT_TRUE(recorder.isActive());
    EXPECT_NE(nullptr, xad::GraphRecorder<float>::getActive());
}

TEST(GraphRecorder, canActivateStatically)
{
    xad::GraphRecorder<float> recorder(false);

    EXPECT_FALSE(recorder.isActive());
    EXPECT_EQ(nullptr, xad::GraphRecorder<float>::getActive());

    xad::GraphRecorder<float>::setActive(&recorder);

    EXPECT_TRUE(recorder.isActive());
    EXPECT_NE(nullptr, xad::GraphRecorder<float>::getActive());
}

TEST(GraphRecorder, canDeactivateGlobally)
{
    EXPECT_EQ(nullptr, xad::GraphRecorder<double>::getActive());

    xad::GraphRecorder<double> recorder;

    EXPECT_TRUE(recorder.isActive());
    xad::GraphRecorder<double>::deactivateAll();
    EXPECT_FALSE(recorder.isActive());
}

TEST(GraphRecorder, isMovable)
{
    xad::GraphRecorder<double> recorder1(false);
    xad::GraphRecorder<double> recorder2 = std::move(recorder1);  // move constructor
    EXPECT_FALSE(recorder2.isActive());

    xad::GraphRecorder<double> recorder3(true);
    recorder3 = std::move(recorder2);  // move assign
    EXPECT_FALSE(recorder3.isActive());

    xad::GraphRecorder<double> recorder4(true);
    EXPECT_TRUE(recorder4.isActive());
    xad::GraphRecorder<double> recorder5 = std::move(recorder4);  // move active
    EXPECT_TRUE(recorder5.isActive());
    EXPECT_FALSE(recorder4.isActive());
}

TEST(GraphRecorder, canSetBackend)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c = a * b;
    recorder.registerOutput(c);
    recorder.compile();

    double output;
    recorder.forward(&output);
    EXPECT_DOUBLE_EQ(6.0, output);

    // Replace backend with a new interpreter
    recorder.setBackend(std::unique_ptr<xad::GraphBackend<double>>(new xad::GraphInterpreter<double>()));

    // After setBackend, need to recompile since backend was reset
    recorder.compile();
    recorder.forward(&output);
    EXPECT_DOUBLE_EQ(6.0, output);
}

TEST(GraphRecorder, constructorWithExplicitBackend)
{
    auto backend = std::unique_ptr<xad::GraphBackend<double>>(new xad::GraphInterpreter<double>());
    xad::GraphRecorder<double> recorder(std::move(backend), true);

    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c = a + b;
    recorder.registerOutput(c);
    recorder.compile();

    double output;
    recorder.forward(&output);
    EXPECT_DOUBLE_EQ(5.0, output);
}

TEST(GraphRecorder, canRegisterInputsAndOutputs)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c = a + b;
    recorder.registerOutput(c);

    EXPECT_EQ(2u, recorder.getGraph().input_ids.size());
    EXPECT_EQ(1u, recorder.getGraph().output_ids.size());
    EXPECT_GE(recorder.getGraph().nodeCount(), 3u);  // at least 2 inputs + 1 operation
}

TEST(GraphRecorder, forwardProducesCorrectValues)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c = a * b + a;  // 2*3 + 2 = 8
    recorder.registerOutput(c);

    recorder.compile();

    double output;
    recorder.forward(&output);
    EXPECT_DOUBLE_EQ(8.0, output);
}

TEST(GraphRecorder, computeAdjointsProducesCorrectGradients)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c = a * b;  // dc/da = b = 3, dc/db = a = 2
    recorder.registerOutput(c);

    recorder.compile();
    recorder.setDerivative(c.getSlot(), 1.0);  // seed
    recorder.computeAdjoints();

    EXPECT_DOUBLE_EQ(3.0, recorder.getDerivative(a.getSlot()));
    EXPECT_DOUBLE_EQ(2.0, recorder.getDerivative(b.getSlot()));
}

TEST(GraphRecorder, canUseNewRecording)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;
    AD b = 3.0;

    recorder.registerInput(a);
    recorder.registerInput(b);

    AD c1 = a + b;
    recorder.registerOutput(c1);
    recorder.compile();

    double output1;
    recorder.forward(&output1);
    EXPECT_DOUBLE_EQ(5.0, output1);

    // New recording with same inputs
    recorder.newRecording();
    AD c2 = a * b;  // different computation
    recorder.registerOutput(c2);
    recorder.compile();

    double output2;
    recorder.forward(&output2);
    EXPECT_DOUBLE_EQ(6.0, output2);
}

TEST(GraphRecorder, clearDerivativesWorks)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;
    AD a = 2.0;

    recorder.registerInput(a);
    AD c = a * a;
    recorder.registerOutput(c);
    recorder.compile();

    recorder.setDerivative(c.getSlot(), 1.0);
    recorder.computeAdjoints();
    EXPECT_DOUBLE_EQ(4.0, recorder.getDerivative(a.getSlot()));

    recorder.clearDerivatives();
    EXPECT_DOUBLE_EQ(0.0, recorder.getDerivative(a.getSlot()));
    EXPECT_DOUBLE_EQ(0.0, recorder.getDerivative(c.getSlot()));
}

TEST(GraphRecorder, registerInputsVector)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    std::vector<AD> inputs = {1.0, 2.0, 3.0};
    recorder.registerInputs(inputs);

    EXPECT_EQ(3u, recorder.getGraph().input_ids.size());
}

TEST(GraphRecorder, registerInputsRange)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    std::vector<AD> inputs = {1.0, 2.0, 3.0, 4.0};
    recorder.registerInputs(inputs.begin(), inputs.begin() + 2);

    EXPECT_EQ(2u, recorder.getGraph().input_ids.size());
}

TEST(GraphRecorder, registerOutputsVector)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 1.0, b = 2.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    std::vector<AD> outputs;
    outputs.push_back(a + b);
    outputs.push_back(a * b);

    recorder.registerOutputs(outputs);

    EXPECT_EQ(2u, recorder.getGraph().output_ids.size());
}

TEST(GraphRecorder, registerOutputsRange)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 1.0, b = 2.0;
    recorder.registerInput(a);
    recorder.registerInput(b);

    std::vector<AD> outputs;
    outputs.push_back(a + b);
    outputs.push_back(a * b);
    outputs.push_back(a - b);

    recorder.registerOutputs(outputs.begin(), outputs.begin() + 2);

    EXPECT_EQ(2u, recorder.getGraph().output_ids.size());
}

TEST(GraphRecorder, recordNodeAndConstant)
{
    xad::GraphRecorder<double> recorder;

    uint32_t c1 = recorder.recordConstant(5.0);
    uint32_t c2 = recorder.recordConstant(3.0);
    uint32_t n = recorder.recordNode(xad::GraphOpCode::Add, c1, c2);

    EXPECT_EQ(xad::GraphOpCode::Constant, recorder.getGraph().getOpCode(c1));
    EXPECT_EQ(xad::GraphOpCode::Constant, recorder.getGraph().getOpCode(c2));
    EXPECT_EQ(xad::GraphOpCode::Add, recorder.getGraph().getOpCode(n));
}

TEST(GraphRecorder, registerVariable)
{
    xad::GraphRecorder<double> recorder;

    auto slot1 = recorder.registerVariable();
    EXPECT_EQ(0u, slot1);  // First variable gets slot 0

    recorder.recordConstant(1.0);  // Add a node
    auto slot2 = recorder.registerVariable();
    EXPECT_EQ(1u, slot2);  // Second variable gets slot 1
}

TEST(GraphRecorder, setActiveThrowsWhenAlreadyActive)
{
    xad::GraphRecorder<double> recorder1;  // Activates itself

    // Trying to activate another graph recorder should throw
    EXPECT_THROW(
        {
            xad::GraphRecorder<double> recorder2;  // Tries to activate, should throw
        },
        xad::OutOfRange);
}

TEST(GraphRecorder, clearAll)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    recorder.registerInput(a);
    AD c = a * a;
    recorder.registerOutput(c);

    EXPECT_GT(recorder.getGraph().nodeCount(), 0u);

    recorder.clearAll();

    EXPECT_EQ(0u, recorder.getGraph().nodeCount());
    EXPECT_EQ(0u, recorder.getGraph().input_ids.size());
    EXPECT_EQ(0u, recorder.getGraph().output_ids.size());
}

TEST(GraphRecorder, getMemory)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    recorder.registerInput(a);
    AD c = a * a;
    recorder.registerOutput(c);

    std::size_t mem = recorder.getMemory();
    EXPECT_GT(mem, 0u);
}

TEST(GraphRecorder, getPosition)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    auto pos0 = recorder.getPosition();
    EXPECT_EQ(0u, pos0);

    AD a = 2.0;
    recorder.registerInput(a);

    auto pos1 = recorder.getPosition();
    EXPECT_GT(pos1, pos0);

    AD c = a * a;
    auto pos2 = recorder.getPosition();
    EXPECT_GT(pos2, pos1);
}

TEST(GraphRecorder, floatScalarOperations)
{
    // Test graph recording with float scalar type to exercise getNestedDoubleValue(float)
    xad::GraphRecorder<float> recorder;
    xad::AReal<float> x = 2.0f;
    recorder.registerInput(x);

    // Scalar multiplication: uses scalar_prod_op which calls getScalarConstant -> getNestedDoubleValue(float)
    xad::AReal<float> y = x * 3.0f;
    recorder.registerOutput(y);

    recorder.compile();

    float output;
    recorder.forward(&output);

    EXPECT_FLOAT_EQ(6.0f, output);
}

// =============================================================================
// Pass-through method tests for coverage
// =============================================================================

TEST(GraphRecorder, passThroughMethods)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0, b = 3.0;
    recorder.registerInput(a);
    recorder.registerInput(b);
    AD c = a * b;
    recorder.registerOutput(c);
    recorder.compile();

    // Test pass-through accessors
    EXPECT_EQ(1u, recorder.vectorWidth());
    EXPECT_EQ(2u, recorder.numInputs());
    EXPECT_EQ(1u, recorder.numOutputs());

    // Test setInput pass-through (must set ALL inputs before forwardAndBackward)
    double newVal = 5.0;
    double bVal = 3.0;
    recorder.setInput(0, &newVal);
    recorder.setInput(1, &bVal);

    // Test forwardAndBackward pass-through
    double output, inputGradients[2];
    recorder.forwardAndBackward(&output, inputGradients);
    EXPECT_DOUBLE_EQ(15.0, output);  // 5 * 3
}

TEST(GraphRecorder, constGetGraph)
{
    xad::GraphRecorder<double> recorder;
    using AD = xad::AReal<double, 1>;

    AD a = 2.0;
    recorder.registerInput(a);
    recorder.registerOutput(a);

    const auto& constRecorder = recorder;
    const auto& graph = constRecorder.getGraph();
    EXPECT_EQ(1u, graph.input_ids.size());
    EXPECT_EQ(1u, graph.output_ids.size());
}

// =============================================================================
// Additional Graph tests
// =============================================================================

#endif  // XAD_ENABLE_GRAPH

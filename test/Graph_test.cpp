/*******************************************************************************

   Unit tests for Graph

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
TEST(Graph, canAddNodesAndConstants)
{
    xad::Graph graph;
    uint32_t c1 = graph.addConstant(3.14);
    uint32_t c2 = graph.addConstant(2.71);
    uint32_t n1 = graph.addNode(xad::GraphOpCode::Add, c1, c2);

    EXPECT_EQ(3u, graph.nodeCount());
    EXPECT_DOUBLE_EQ(3.14, graph.getConstantValue(c1));
    EXPECT_DOUBLE_EQ(2.71, graph.getConstantValue(c2));
    EXPECT_EQ(xad::GraphOpCode::Add, graph.getOpCode(n1));
}

TEST(Graph, canAddInputsAndMarkOutputs)
{
    xad::Graph graph;
    uint32_t in1 = graph.addInput();
    uint32_t in2 = graph.addInput();
    uint32_t out = graph.addNode(xad::GraphOpCode::Mul, in1, in2);

    graph.markOutput(out);

    EXPECT_EQ(2u, graph.input_ids.size());
    EXPECT_EQ(1u, graph.output_ids.size());
    EXPECT_EQ(out, graph.output_ids[0]);
}

TEST(Graph, clearWorks)
{
    xad::Graph graph;
    graph.addConstant(1.0);
    graph.addInput();
    graph.addNode(xad::GraphOpCode::Add, 0, 1);

    EXPECT_GT(graph.nodeCount(), 0u);

    graph.clear();

    EXPECT_EQ(0u, graph.nodeCount());
    EXPECT_EQ(0u, graph.input_ids.size());
    EXPECT_EQ(0u, graph.output_ids.size());
}

TEST(Graph, empty)
{
    xad::Graph graph;
    EXPECT_TRUE(graph.empty());

    graph.addInput();
    EXPECT_FALSE(graph.empty());
}

TEST(Graph, reserve)
{
    xad::Graph graph;
    graph.reserve(100);
    // Just verify it doesn't crash - capacity is implementation detail
    EXPECT_TRUE(graph.empty());
}

TEST(Graph, addUnary)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t neg = graph.addUnary(xad::GraphOpCode::Neg, inp);

    EXPECT_EQ(xad::GraphOpCode::Neg, graph.getOpCode(neg));
}

TEST(Graph, addBinary)
{
    xad::Graph graph;
    uint32_t a = graph.addInput();
    uint32_t b = graph.addInput();
    uint32_t sum = graph.addBinary(xad::GraphOpCode::Add, a, b);

    EXPECT_EQ(xad::GraphOpCode::Add, graph.getOpCode(sum));
}

TEST(Graph, addTernary)
{
    xad::Graph graph;
    uint32_t cond = graph.addInput();
    uint32_t t = graph.addInput();
    uint32_t f = graph.addInput();
    uint32_t result = graph.addTernary(xad::GraphOpCode::If, cond, t, f);

    EXPECT_EQ(xad::GraphOpCode::If, graph.getOpCode(result));
}

TEST(Graph, isInput)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(1.0);

    EXPECT_TRUE(graph.isInput(inp));
    EXPECT_FALSE(graph.isInput(c));
}

TEST(Graph, isConstant)
{
    xad::Graph graph;
    uint32_t inp = graph.addInput();
    uint32_t c = graph.addConstant(1.0);

    EXPECT_FALSE(graph.isConstant(inp));
    EXPECT_TRUE(graph.isConstant(c));
}

TEST(Graph, constantPoolDeduplication)
{
    xad::Graph graph;
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

#endif  // XAD_ENABLE_GRAPH

/*******************************************************************************
 *
 *   Unit tests for JITExprTraits helpers
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2025 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Affero General Public License for more details.
 *
 *   You should have received a copy of the GNU Affero General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>

#ifdef XAD_ENABLE_JIT

TEST(JITExprTraits, getNestedDoubleValueRecursesThroughValue)
{
    // This exercises the template overload:
    //   getNestedDoubleValue(const T& x) { return getNestedDoubleValue(x.value()); }
    xad::AReal<float, 1> x = 1.25f;
    EXPECT_DOUBLE_EQ(1.25, xad::getNestedDoubleValue(x));
}

#endif  // XAD_ENABLE_JIT



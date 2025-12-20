##############################################################################
#
#  Cmake file for unit tests
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2025 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

set(testfiles
    Vec_test.cpp
    TapeContainer_test.cpp
    ChunkContainer_test.cpp
    TestHelpers.hpp
    Tape_test.cpp
    Expressions_test.cpp
    ExpressionsConversion_test.cpp
    ExpressionMath1_test.cpp
    ExpressionMath2_test.cpp
    ExpressionMath3_test.cpp
    Checkpointing_test.cpp
    ExternalFunc_test.cpp
    Exceptions_test.cpp
    HigherOrder_test.cpp
    Complex_test.cpp
    ComplexAD_test.cpp
    ExpressionMeta_test.cpp
    Tbr_test.cpp
    StreamOps_test.cpp
    StdCompatibility_test.cpp
    ReusableRange_test.cpp
    PartialRollback_test.cpp
    Hessian_test.cpp
    Jacobian_test.cpp
    TypeTraits_test.cpp
    OperationsContainer_test.cpp
    FRealDirect_test.cpp
    FReal_test.cpp
    ARealDirect_test.cpp
    DirectModeBenchmark.cpp
)

if (XAD_ENABLE_EIGEN_TESTS)
    list(APPEND testfiles Eigen_test.cpp)
endif()

if (XAD_ENABLE_JIT)
    list(APPEND testfiles
        JITCompiler_test.cpp
        JITTutorial_test.cpp
        JITExprTraits_test.cpp
        JITGraph_test.cpp
        JITGraphInterpreter_test.cpp
        JITABool_test.cpp
        JITExpressionMath_test.cpp
    )
endif()

if (CMAKE_COMPILER_IS_GNUCC)
    set_source_files_properties(ExpressionsConversion_test.cpp
        COMPILE_FLAGS "-Wno-conversion")
ENDIF()

xad_add_test(xad_test ${testfiles})

# extra tests for compile failures
add_subdirectory(compile_failures)

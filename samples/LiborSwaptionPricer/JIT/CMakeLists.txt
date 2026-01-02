##############################################################################
#
#  Cmake file for LiborSwaptionPricer JIT Benchmark
#
#  This builds the JIT-accelerated version of the LIBOR swaption pricer
#  using Forge JIT compilation backend.
#
#  Requires:
#  - XAD with JIT enabled (XAD_ENABLE_JIT=ON)
#  - xad-forge library
#  - Forge C API
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

find_package(xad-forge QUIET)

if(NOT xad-forge_FOUND)
    message(STATUS "xad-forge not found - skipping LiborSwaptionPricerJIT")
    return()
endif()

add_executable(LiborSwaptionPricerJIT
    main.cpp
    LiborPricersJIT.cpp
    LiborPricersJIT.hpp
    LiborFunctionsJIT.hpp
    ../LiborData.hpp
    ../LiborFunctions.hpp
    ../LiborPricers.hpp
    ../LiborPricers.cpp
)

target_link_libraries(LiborSwaptionPricerJIT
    PRIVATE
        xad::xad
        xad-forge::xad-forge
)

target_include_directories(LiborSwaptionPricerJIT
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set C++ standard
set_target_properties(LiborSwaptionPricerJIT PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add as a test (quick mode for CI)
add_test(
    NAME LiborSwaptionPricerJIT
    COMMAND LiborSwaptionPricerJIT --quick
)

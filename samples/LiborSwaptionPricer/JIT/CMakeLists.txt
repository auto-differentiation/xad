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

# Try to find xad-forge
if(NOT TARGET XADForge::xad-forge)
    find_package(xad-forge QUIET)
endif()

if(NOT TARGET XADForge::xad-forge)
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

# Link to xad (use target name based on context)
if(TARGET xad)
    set(XAD_TARGET xad)
elseif(TARGET XAD::xad)
    set(XAD_TARGET XAD::xad)
else()
    message(FATAL_ERROR "XAD target not found")
endif()

target_link_libraries(LiborSwaptionPricerJIT
    PRIVATE
        ${XAD_TARGET}
        XADForge::xad-forge
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

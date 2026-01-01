##############################################################################
#
#  Cmake file for 1st order adjoint mode for a Monte-Carlo Libor Swaption
#  portfolio pricer.
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

option(XAD_LIBOR_ENABLE_FORGE "Enable Forge JIT backend for LiborSwaptionPricer benchmark" OFF)

xad_add_sample(LiborSwaptionPricer
  SOURCES
    LiborData.hpp
    LiborFunctions.hpp
    LiborPricers.hpp
    LiborPricers.cpp
    main.cpp
  TEST_ARGS
    100
    test
)

# Optional xad-forge integration for JIT benchmarking
if(XAD_LIBOR_ENABLE_FORGE)
    # Try to find xad-forge
    if(NOT TARGET XADForge::xad-forge)
        find_package(xad-forge CONFIG QUIET)
    endif()

    if(TARGET XADForge::xad-forge)
        target_link_libraries(LiborSwaptionPricer PRIVATE XADForge::xad-forge)
        target_compile_definitions(LiborSwaptionPricer PRIVATE XAD_FORGE_ENABLED=1)
        message(STATUS "LiborSwaptionPricer: Forge JIT backend enabled")

        # Add JIT test case
        if(NOT XAD_SAMPLES_MAIN)
            add_test(NAME sample_LiborSwaptionPricer_jit
                     COMMAND LiborSwaptionPricer 100 jit test)
        endif()
    else()
        message(FATAL_ERROR "LiborSwaptionPricer: XAD_LIBOR_ENABLE_FORGE=ON but xad-forge not found. "
                            "Please set CMAKE_PREFIX_PATH to include xad-forge install location.")
    endif()
endif()

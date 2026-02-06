##############################################################################
#
#  Cmake file for all samples
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2026 Xcelerit Computing Ltd.
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

# This is configured as a sub-project, as it can be built against an installed
# version of XAD as well
cmake_minimum_required(VERSION 3.15.2)
cmake_policy(SET CMP0092 NEW)   # avoid setting warning flags by default in Windows
project(xad_samples
    LANGUAGES CXX
)

if(NOT TARGET XAD::xad)
    find_package(XAD REQUIRED)
    get_target_property(incdir XAD::xad INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "XAD include dir: ${incdir}")
endif()

# These commands only run if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL "xad_samples")
    message(STATUS "XAD Samples as main project")
    set(XAD_SAMPLES_MAIN TRUE)
else()
    set(XAD_SAMPLES_MAIN FALSE)
    message(STATUS "XAD Samples as sub project")
endif()

# add a sample - link to XAD and add common to include path
function(xad_add_sample name)
    cmake_parse_arguments(ARGS "" "" "SOURCES;TEST_ARGS" ${ARGN})

    if(NOT ARGS_SOURCES)
        message(FATAL_ERROR "The SOURCES argument must be given for samples")
    endif()

    add_executable(${name} ${ARGS_SOURCES})
    target_link_libraries(${name} PRIVATE XAD::xad)
    target_include_directories(${name} PRIVATE ../common)
    get_target_property(msvcrt XAD::xad MSVC_RUNTIME_LIBRARY)
    if(MSVC AND msvcrt)
        set_target_properties(${name} PROPERTIES MSVC_RUNTIME_LIBRARY "${msvcrt}")
    endif()
    if(NOT XAD_SAMPLES_MAIN)
        add_test(NAME sample_${name} COMMAND ${name} ${ARGS_TEST_ARGS})
        set_target_properties(${name} PROPERTIES FOLDER "samples")
        if(XAD_ENABLE_ADDRESS_SANITIZER)
            target_compile_options(${name} PRIVATE ${xad_cxx_asan_flags})
            target_link_options(${name} PRIVATE ${xad_link_asan_flags})
        endif()
    endif()
endfunction()

add_subdirectory(adj_1st)
add_subdirectory(adjvec_1st)
add_subdirectory(fwd_1st)
add_subdirectory(fwdvec_1st)
add_subdirectory(checkpointing)
add_subdirectory(external_function)
add_subdirectory(SwapPricer)
add_subdirectory(fwd_adj_2nd)
add_subdirectory(Hessian)
add_subdirectory(Jacobian)
add_subdirectory(LiborSwaptionPricer)
add_subdirectory(jit_tutorial)



##############################################################################
#   
#  Setup of CMake options
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
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

include(CMakeDependentOption)

# Build options
# Enable the tests only if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL "xad")
    option(XAD_ENABLE_TESTS "Enable the XAD tests" ON)
else()
    option(XAD_ENABLE_TESTS "Enable the XAD tests" OFF)
endif()
option(XAD_WARNINGS_PARANOID "Use extra-paranoid warning level" ON)
option(XAD_POSITION_INDEPENDENT_CODE "Generate PIC code, so it can be linked into a shared library" ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(XAD_SIMD_OPTION "APPLE_M1" CACHE STRING "SIMD instruction set to use")
        set_property(CACHE XAD_SIMD_OPTION PROPERTY STRINGS APPLE_M1 NATIVE)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(XAD_SIMD_OPTION "NATIVE" CACHE STRING "SIMD instruction set to use")
        set_property(CACHE XAD_SIMD_OPTION PROPERTY STRINGS NATIVE)
    endif()
else()
    set(XAD_SIMD_OPTION "AVX" CACHE STRING "SIMD instruction set to use")
    set_property(CACHE XAD_SIMD_OPTION PROPERTY STRINGS SSE2 AVX AVX2 AVX512 NATIVE) # for drop-down in GUI
endif()

message(STATUS "Using SIMD instruction set: ${XAD_SIMD_OPTION}")

option(XAD_ENABLE_ADDRESS_SANITIZER "Enable address sanitizer (Gcc/Clang only)" OFF)

if(MSVC)
    option(XAD_STATIC_MSVC_RUNTIME "Use static C++ Runtime in MSVC (/MT instead of /MD)" OFF)
    if(XAD_STATIC_MSVC_RUNTIME)
        message(STATUS "Using static MSVC runtime")
    else()
        message(STATUS "Using dynamic MSVC runtime")
    endif()
endif()

# Tape options: these end up in Config.hpp, a cmake-generated file
option(XAD_TAPE_REUSE_SLOTS "Reuse slots in tape that have become free (slower, less memory)" OFF)
option(XAD_NO_THREADLOCAL "Disable thread-local tape - only for single-threaded tape use" OFF)
if (MSVC AND MSVC_VERSION GREATER_EQUAL 1920)
    option(XAD_USE_STRONG_INLINE "Use forced inlining for higher preformance, at a higher compile time cost" OFF)
else()
    # in VS 2015 and 2017, without strong inlining, some long expressions in release mode get miscompiled
    set(XAD_USE_STRONG_INLINE ON CACHE BOOL "Use forced inlining for higher preformance, at a higher compile time cost" FORCE)
endif()
option(XAD_ALLOW_INT_CONVERSION "Add real->int conversion operator, potentially missing to track dependencies" ON)
option(XAD_REDUCED_MEMORY "Reduce memory required for tape, at a slight performance cost" OFF)

if(XAD_REDUCED_MEMORY)
    message(STATUS "Using reduced memory for tape storage at a slight performance cost")
endif()

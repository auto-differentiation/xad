##############################################################################
#   
#  Setup of unit tests, downloading Google Test on the fly
#
#  This file is part of XAD, a fast and comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2022 Xcelerit Computing Ltd.
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

# mask this in case parent project already has gtest
if (NOT TARGET GTest::gmock_main)
    include(FetchContent)

    # Prevent GoogleTest from overriding our compiler/linker options
    # when building with Visual Studio
    if(XAD_STATIC_MSVC_RUNTIME)
        set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    else()
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             release-1.11.0
    )
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES 
        FOLDER "test/gtest")
    if(XAD_STATIC_MSVC_RUNTIME)
        set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()
include(GoogleTest)

enable_testing()

# Adds a Google test executable
# Note that it auto-links with gmock_main, which includes gtest as well
function(xad_add_test name)
    xad_add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE xad GTest::gmock_main)
    set_property(TARGET ${name} PROPERTY FOLDER test)
    gtest_discover_tests(${name})
endfunction()


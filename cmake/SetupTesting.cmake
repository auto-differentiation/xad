##############################################################################
#   
#  Setup of unit tests, downloading Google Test on the fly
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2023 Xcelerit Computing Ltd.
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
    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
        set(_gtest_tag release-1.10.0)
    else()
        set(_gtest_tag release-1.11.0)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             ${_gtest_tag}
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
    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
        set(_gmock_target gmock_main)
    else()
        set(_gmock_target GTest::gmock_main)
    endif()
    xad_add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE xad ${_gmock_target})
    set_property(TARGET ${name} PROPERTY FOLDER test)
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 15)
endfunction()


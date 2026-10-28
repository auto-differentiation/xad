##############################################################################
#
#  Setup of unit tests, downloading Google Test on the fly
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2026 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, version 3 of the License only.
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
        set(_gtest_tag v1.15.2)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             ${_gtest_tag}
    )
    FetchContent_MakeAvailable(googletest)

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


if (XAD_ENABLE_EIGEN_TESTS)
    # an installed Eigen 3.4 or 5 (package manager or CMAKE_PREFIX_PATH), else the 3.4.0 release
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if (NOT Eigen3_FOUND AND NOT CMAKE_VERSION VERSION_LESS 3.19)
        find_package(Eigen3 3.4...5 QUIET NO_MODULE)
    endif()
    if (Eigen3_FOUND)
        message(STATUS "Using Eigen ${Eigen3_VERSION} from ${Eigen3_DIR}")
    else()
        if (POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        include(FetchContent)
        FetchContent_Declare(
            Eigen3
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
            URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
            SOURCE_SUBDIR cmake # no CMakeLists.txt in cmake, so this turns off configure
        )
        FetchContent_MakeAvailable(Eigen3)

        # add dependency ourselves - their CMake files seem broken
        add_library(Eigen INTERFACE)
        target_include_directories(Eigen INTERFACE ${eigen3_SOURCE_DIR})
        add_library(Eigen3::Eigen ALIAS Eigen)
    endif()
endif()

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
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 30)

    if (XAD_ENABLE_EIGEN_TESTS)
        target_link_libraries(${name} PRIVATE Eigen3::Eigen)
    endif()
endfunction()


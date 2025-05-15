##############################################################################
#   
#  Setup of unit tests, downloading Google Test on the fly
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

    list(FIND ARGN "Eigen_test.cpp" eigen_index)
    if(eigen_index GREATER -1)
        if(NOT DEFINED FetchContent_MakeAvailable)
            include(FetchContent)
        endif()
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            SOURCE_SUBDIR cmake # no CMakeLists.txt in cmake, so this turns off configure
        )
        FetchContent_MakeAvailable(Eigen3)

        find_package(Eigen3 3.3 REQUIRED NO_MODULE)
        target_link_libraries(${name} PRIVATE Eigen3::Eigen)
        target_include_directories(${name} PRIVATE ../../../eigen)
    endif()
endfunction()


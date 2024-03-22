##############################################################################
#   
#  Python bindings using Pybind11 - exension module.
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

pybind11_add_module(_xad_autodiff MODULE 
    module.cpp math.hpp tape.hpp real.hpp exceptions.hpp 
)

target_link_libraries(_xad_autodiff PRIVATE XAD::xad)
set_target_properties(_xad_autodiff PROPERTIES CXX_STANDARD 17)
if(MSVC)
    # to allow using M_PI and the like
    target_compile_definitions(_xad_autodiff PRIVATE _USE_MATH_DEFINES)
    # respect the static build setting - we do static builds on windows to avoid dependency problems 
    # with Python
    if(XAD_STATIC_MSVC_RUNTIME)
        set_target_properties(_xad_autodiff PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set_target_properties(_xad_autodiff PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

# write filename into source dir, so that python build can find it
add_custom_command(TARGET _xad_autodiff POST_BUILD
    COMMAND echo $<TARGET_FILE:_xad_autodiff> > ${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt_file.txt)

# setup venv using poetry + build with wheel, producing a stamp value to hook dependencies
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp
    DEPENDS _xad_autodiff
            ${CMAKE_CURRENT_SOURCE_DIR}/../pyproject.toml
            ${CMAKE_CURRENT_SOURCE_DIR}/../poetry.lock
    # install environment
    COMMAND ${POETRY_EXECUTABLE} install
    # generate stubs for typings
    COMMAND ${POETRY_EXECUTABLE} run pybind11-stubgen xad_autodiff --output-dir .
    # build the Python weel for the package
    COMMAND ${POETRY_EXECUTABLE} build -f wheel
    # update the stamp file, to keep track of when we last build it
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp
    COMMENT "Building Python wheel..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# main target for building the python wheel - custom command above will be hooked to this target
add_custom_target(python_wheel ALL 
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp)


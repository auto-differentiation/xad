##############################################################################
#   
#  Setup of Python bindings build, downloading pybind on the fly
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

if(XAD_ENABLE_PYTHON)
    
    # fetch pybind11 dependency on the fly
    include(FetchContent)

    FetchContent_Declare(pybind11
                GIT_REPOSITORY   https://github.com/pybind/pybind11.git
                GIT_TAG          v2.11.1)
    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()

    set_target_properties(pybind11_headers PROPERTIES
        FOLDER "bindings/python"
    )

    # Find poetry: https://python-poetry.org/docs/
    find_program(POETRY_EXECUTABLE poetry REQUIRED)
    
endif()
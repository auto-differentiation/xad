##############################################################################
#   
#  Setup of programs required to build the user docs
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

# allow to force-reconfigure the Venv
set (XAD_RECONFIGURE_VENV FALSE CACHE BOOL "Force reconfigure Sphinx Venv")

# first, setup a venv for python and use that
if(NOT EXISTS "${PROJECT_BINARY_DIR}/venv")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${PROJECT_BINARY_DIR}/venv")
    ## unset Python3_EXECUTABLE to make sure it searches again below
    unset (Python3_EXECUTABLE)
    set (XAD_RECONFIGURE_VENV TRUE CACHE BOOL "Force reconfigure Sphinx Venv" FORCE)
endif()

## search for python in virtual environment only
set (ENV{VIRTUAL_ENV} "${PROJECT_BINARY_DIR}/venv")
set (Python3_FIND_VIRTUALENV ONLY)   
find_package (Python3 COMPONENTS Interpreter REQUIRED)

## install the dependencies
if(XAD_RECONFIGURE_VENV)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip install --upgrade pip wheel)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip install -r "${PROJECT_SOURCE_DIR}/requirements.txt")
    set (XAD_RECONFIGURE_VENV FALSE CACHE BOOL "Force reconfigure Sphinx Venv" FORCE)
endif()

## now look for Sphinx
find_program(SPHINX_EXECUTABLE NAMES sphinx-build
    HINTS
        "${PROJECT_BINARY_DIR}/venv"
    PATH_SUFFIXES bin Scripts
    DOC "Sphinx documentation generator"
)
mark_as_advanced(SPHINX_EXECUTABLE)
if(NOT SPHINX_EXECUTABLE)
    message(FATAL_ERROR "Could not find sphinx - please delete ${PROJECT_BINARY_DIR}/venv to re-create the environment")
endif()

## we also need clang-format
find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format
    HINTS
        "${PROJECT_BINARY_DIR}/venv"
    PATH_SUFFIXES bin Scripts
    DOC "Clang format executable for docs"
)
mark_as_advanced(CLANG_FORMAT_EXECUTABLE)
if(NOT CLANG_FORMAT_EXECUTABLE)
    message(FATAL_ERROR "Could not find clang-format - please delete ${PROJECT_BINARY_DIR}/venv to re-create the environment")
endif()


## we also need Latex for SVG math - looks much better than mathjax
find_package(LATEX REQUIRED)

## and we need dvisvgm to convert SVG images for latex
# find dvisvgm
find_program(DVISVGM_CONVERTER
    NAMES dvisvgm
    PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin
)
if (NOT DVISVGM_CONVERTER)
    message(FATAL_ERROR "Could not find dvisvgm, which is requires for latex doc generation")
else()
    message(STATUS "dvisvgm: ${DVISVGM_CONVERTER}")
endif()

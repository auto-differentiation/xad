##############################################################################
#   
#  Main CMake file
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

############## setup ###################
cmake_minimum_required(VERSION 3.15.2)
cmake_policy(SET CMP0092 NEW)   # avoid setting warning flags by default in Windows

project(XAD 
    LANGUAGES CXX
    DESCRIPTION "Fast and comprehensive C++ library for automatic differentiation."
    HOMEPAGE_URL "https://getxad.com"
)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(SetupVersion)
setup_version(XAD 1 0 0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # allow using solution folders
include(GNUInstallDirs)   # for installation directories

include(SetupOptions)

############# build subdirectories ##########

if(NOT XAD_DOCS_ONLY)
    include(SetupCompiler)
    add_subdirectory(src)
    if(XAD_ENABLE_TESTS)
        include(SetupTesting)
        add_subdirectory(test)
        add_subdirectory(samples)
    endif()
endif()

if(XAD_BUILD_DOCS)
    include(SetupDocumentation)
    add_subdirectory(doc)
endif()

install(FILES CHANGELOG.md LICENSE.md TYPE DOC)



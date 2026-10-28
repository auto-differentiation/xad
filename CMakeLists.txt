##############################################################################
#
#  Main CMake file
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

############## setup ###################
cmake_minimum_required(VERSION 3.15.2)
cmake_policy(SET CMP0092 NEW)   # avoid setting warning flags by default in Windows

project(xad
    LANGUAGES CXX
    DESCRIPTION "Comprehensive C++ library for automatic differentiation."
    HOMEPAGE_URL "https://auto-differentiation.github.io"
)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(XADSetupVersion)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # allow using solution folders
include(GNUInstallDirs)   # for installation directories

include(XADSetupOptions)

############# build subdirectories ##########

if(NOT XAD_DOCS_ONLY)
    include(XADSetupCompiler)
    if(XAD_ENABLE_TESTS)
        include(XADSetupTesting)
    endif()

    if(XAD_ENABLE_CODEGEN)
        # Find xad-codegen: try subdirectory first, then find_package
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/codegen/CMakeLists.txt")
            add_subdirectory(codegen)
        elseif(DEFINED XAD_CODEGEN_DIR AND EXISTS "${XAD_CODEGEN_DIR}/CMakeLists.txt")
            add_subdirectory("${XAD_CODEGEN_DIR}" codegen)
        else()
            find_package(xad-codegen CONFIG QUIET)
            if(NOT xad-codegen_FOUND)
                message(FATAL_ERROR
                    "XAD_ENABLE_CODEGEN=ON but xad-codegen not found.\n"
                    "Either:\n"
                    "  1. Set XAD_CODEGEN_DIR to the xad-codegen source directory\n"
                    "  2. Set CMAKE_PREFIX_PATH to a xad-codegen installation\n"
                    "  3. Place xad-codegen source in ${CMAKE_CURRENT_SOURCE_DIR}/codegen/\n"
                    "\n"
                    "xad-codegen requires a commercial license.\n"
                    "See https://www.xcelerit.com/xad-enterprise-support"
                )
            endif()
        endif()
    endif()

    add_subdirectory(src)

    if(XAD_ENABLE_TESTS)
        add_subdirectory(test)
        add_subdirectory(samples)
    endif()
endif()

# When codegen is enabled and used as subdirectory, add its targets to XAD export set
if(XAD_ENABLE_CODEGEN AND TARGET xad_codegen)
    install(TARGETS xad_codegen asmjit
        EXPORT XADTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    install(TARGETS nlohmann_json EXPORT XADTargets)
    if(TARGET sleefavx2)
        install(TARGETS sleefavx2 sleef EXPORT XADTargets
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        )
    elseif(TARGET sleef)
        install(TARGETS sleef EXPORT XADTargets
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
endif()

install(FILES CHANGELOG.md LICENSE.md TYPE DOC)



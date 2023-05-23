##############################################################################
#   
#  Macro for setup of versioning variables
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


######################################
#
#  Setup of Versioning 
#
#  defines:
#  <name>_VERSION_[MAJOR/MINOR/PATCH/PATCHNUM]  Note: PATCH includes the suffix while PATCHNUM is numeric
#  <name>_VERSION_SUFFIX                        For prereleases, usually a string
#  <name>_VERSION                               Full version string, incl. suffix
#  VERSION                                      Same as <name> version
#  <name>_IS_PRERELEASE                         Boolean
#  <name>_VERSION_NUMERIC                       10000 * major + 100 * minor + patchnum
# 
#  CURRENT_YEAR (used in documentation generator)
#
#  It also defines the PROJECT_ variables of the version, to be consistent with CMake's project command
#
######################################
macro(setup_version name major minor patch)
    # keep these numbers as globally accessible variables
    set(${name}_VERSION_MAJOR ${major})
    set(${name}_VERSION_MINOR ${minor})
    if(NOT "${ARGN}" STREQUAL "")
        set(${name}_VERSION_PATCH "${patch}-${ARGN}")
        set(${name}_VERSION_SUFFIX "${ARGN}")
        set(${name}_IS_PRERELEASE TRUE)
    else()
        set(${name}_VERSION_PATCH ${patch})
        set(${name}_VERSION_SUFFIX "")
        set(${name}_IS_PRELEASE FALSE)
    endif()
    set(VERSION "${${name}_VERSION_MAJOR}.${${name}_VERSION_MINOR}.${${name}_VERSION_PATCH}")
    set(${name}_VERSION "${VERSION}")

    # in case there is an a or RC or b letter
    set(${name}_VERSION_PATCHNUM ${patch})

    # numeric version - major*10000 + minor * 100 + patchnum
    math(EXPR ${name}_VERSION_NUMERIC "${major} * 10000 + ${minor} * 100 + ${${name}_VERSION_PATCHNUM}")

    # project variables
    set(PROJECT_VERSION ${${name}_VERSION})
    set(PROJECT_VERSION_MAJOR ${${name}_VERSION_MAJOR})
    set(PROJECT_VERSION_MINOR ${${name}_VERSION_MINOR})
    set(PROJECT_VERSION_PATCH ${${name}_VERSION_PATCH})

    string(TIMESTAMP CURRENT_YEAR "%Y")

    message(STATUS "Building ${name}, Version ${VERSION}...")
endmacro()

# parse VERSION file
file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _raw_version)
string(STRIP "${_raw_version}" _raw_version)
string(REPLACE "." ";" _version_list "${_raw_version}")
list(LENGTH _version_list _version_list_length)
if(NOT _version_list_length EQUAL 3)
    message(FATAL_ERROR "Version string ${_raw_version} does not have the required 3 elements")
endif()
list(GET _version_list 0 _version_major)
list(GET _version_list 1 _version_minor)
list(GET _version_list 2 _version_patch_raw)
string(REPLACE "-" ";" _version_patch_list "${_version_patch_raw}")
list(LENGTH _version_patch_list _version_patch_list_len)
if(_version_patch_list_len EQUAL 2)
    list(GET _version_patch_list 0 _version_patch)
    list(GET _version_patch_list 1 _version_suffix)
else()
    set(_version_patch "${_version_patch_raw}")
    set(_version_suffix "")
endif()

setup_version(${PROJECT_NAME} "${_version_major}" "${_version_minor}" "${_version_patch}" ${_version_suffix})

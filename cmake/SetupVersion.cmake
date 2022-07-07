##############################################################################
#   
#  Macro for setup of versioning variables
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


######################################
#
#  Setup of Versioning 
#
#  defines:
#  <name>_VERSION_[MAJOR/MINOR/PATCH/PATCHNUM] 
#  VERSION
#  <name>_VERSION
#  CURRENT_YEAR (used in documentation generator)
#
######################################
macro(setup_version name major minor patch)
    # keep these numbers as globally accessible variables
    set(${name}_VERSION_MAJOR ${major})
    set(${name}_VERSION_MINOR ${minor})
    set(${name}_VERSION_PATCH ${patch})
    set(VERSION "${${name}_VERSION_MAJOR}.${${name}_VERSION_MINOR}.${${name}_VERSION_PATCH}")
    set(${name}_VERSION "${VERSION}")

    # in case there is an a or RC or b letter
    string(SUBSTRING "${${name}_VERSION_PATCH}" 0 1 ${name}_VERSION_PATCHNUM)

    # numeric version - major*10000 + minor * 100 + patchnum
    math(EXPR ${name}_VERSION_NUMERIC "${major} * 10000 + ${minor} * 100 + ${${name}_VERSION_PATCHNUM}")

    string(TIMESTAMP CURRENT_YEAR "%Y")

    message(STATUS "Building ${name}, Version ${VERSION}...")
endmacro()

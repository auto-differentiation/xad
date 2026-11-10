##############################################################################
#
#  Eigen sample CMakefile - built only when Eigen 3.4 or later is available
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

# Eigen3::Eigen already exists with XAD_ENABLE_EIGEN_TESTS; otherwise look for an installed Eigen
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
endif()

if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Skipping Eigen sample (Eigen 3.4 not found)")
else()
    # not named Eigen: that name is taken by the fetched Eigen library target
    xad_add_sample(EigenSample SOURCES main.cpp)
    target_link_libraries(EigenSample PRIVATE Eigen3::Eigen)
endif()

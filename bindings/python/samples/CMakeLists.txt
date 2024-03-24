##############################################################################
#
#  Adds samples as tests to the build, to ensure error-free execution
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

function(add_python_sample_test name )
    add_test(NAME "python_sample_${name}"
        COMMAND ${POETRY_EXECUTABLE} run python "${name}.py"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}) 
endfunction()

add_python_sample_test(adj_1st)
add_python_sample_test(fwd_1st)
add_python_sample_test(swap_pricer)
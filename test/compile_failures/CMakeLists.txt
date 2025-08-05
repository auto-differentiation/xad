##############################################################################
#
#  CMake helper to test if a compilation fails (it should)
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2025 Xcelerit Computing Ltd.
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


# add a failure test - make sure compilation is failing
function(compile_failure_test name)
    add_library(compile_fail_${name} OBJECT EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(compile_fail_${name} PRIVATE xad)
    set_target_properties(compile_fail_${name} PROPERTIES
        FOLDER "test/must_fail"
        EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    add_test(NAME compile_fail_${name}
        COMMAND ${CMAKE_COMMAND} --build . --target compile_fail_${name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(compile_fail_${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

compile_failure_test(ternary_operator_conversion)
compile_failure_test(convert_to_double)
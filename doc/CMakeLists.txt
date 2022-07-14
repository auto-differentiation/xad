##############################################################################
#   
#  Build the Sphinx-based user documentation
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

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
 
# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")
 
# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)
    
add_custom_target(documentation
    COMMAND ${SPHINX_EXECUTABLE}
        -b html
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        -j 4
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
    COMMAND ${CMAKE_COMMAND} -E touch "${SPHINX_HTML_DIR}/.nojekyll"
    COMMAND ${CMAKE_COMMAND} -DSPHINX_HTML_DIR="${SPHINX_HTML_DIR}" -P "${PROJECT_SOURCE_DIR}/cmake/SphinxPostProcess.cmake" 
    COMMENT "Building HTML documentation with Sphinx")
    

set_target_properties(documentation PROPERTIES FOLDER documentation)
install(DIRECTORY ${SPHINX_HTML_DIR} TYPE DOC)


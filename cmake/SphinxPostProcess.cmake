##############################################################################
#   
#  Post-processing script for Sphinx to replace the _static with static,
#  and _images with images. This is needed for website docs and also
#  makes file structure more regular.
#
#  This script is executed as a build step.
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

file(MAKE_DIRECTORY ${DOC_DEST_DIR})

# removing without underscores
execute_process(
     COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOC_TMP_DIR}/_static ${DOC_DEST_DIR}/static
     COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOC_TMP_DIR}/_images ${DOC_DEST_DIR}/images
)

# convert references to these underscore directories
# note: we explicitly ignore the .inv file here - no need for intersphinx references
file(GLOB base_files
     LIST_DIRECTORIES false
     RELATIVE ${DOC_TMP_DIR}
     ${DOC_TMP_DIR}/*.html)
file(GLOB ref_files
     LIST_DIRECTORIES false
     RELATIVE ${DOC_TMP_DIR}
     ${DOC_TMP_DIR}/ref/*.html)

set(srcfiles ${base_files} ${ref_files})
foreach(srcfile IN LISTS srcfiles)
     file(READ ${DOC_TMP_DIR}/${srcfile} contents)
     string(REPLACE "_static/" "static/" contents "${contents}")
     string(REPLACE "_images/" "images/" contents "${contents}")
     file(WRITE ${DOC_DEST_DIR}/${srcfile} "${contents}")
endforeach() 

# add a .nojekyll file for github
file(TOUCH  ${DOC_DEST_DIR}/.nojekyll)
    
     
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

# process the sitemap.xml, removing index.html to root
file(READ ${SPHINX_HTML_DIR}/sitemap.xml sitemap)
string(REPLACE "https://xad.xcelerit.com/index.html" "https://xad.xcelerit.com/" sitemap "${sitemap}")
file(WRITE ${SPHINX_HTML_DIR}/sitemap.xml "${sitemap}")
    
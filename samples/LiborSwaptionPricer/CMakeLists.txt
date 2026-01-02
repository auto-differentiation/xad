##############################################################################
#
#  Cmake file for 1st order adjoint mode for a Monte-Carlo Libor Swaption
#  portfolio pricer.
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

xad_add_sample(LiborSwaptionPricer
  SOURCES
    LiborData.hpp
    LiborFunctions.hpp
    LiborPricers.hpp
    LiborPricers.cpp
    main.cpp
  TEST_ARGS
    --quick
    --validate
)

# Add JIT subfolder if xad-forge is available
add_subdirectory(JIT)

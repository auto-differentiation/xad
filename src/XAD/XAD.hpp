/*******************************************************************************

   The main include file, including all other headers.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <XAD/ARealDirect.hpp>
#include <XAD/BinaryDerivativeImpl.hpp>
#include <XAD/BinaryExpr.hpp>
#include <XAD/BinaryFunctors.hpp>
#include <XAD/BinaryMathFunctors.hpp>
#include <XAD/BinaryOperators.hpp>
#include <XAD/CheckpointCallback.hpp>
#include <XAD/ChunkContainer.hpp>
#include <XAD/Complex.hpp>
#include <XAD/Config.hpp>
#include <XAD/Exceptions.hpp>
#include <XAD/Expression.hpp>
#include <XAD/FRealDirect.hpp>
#include <XAD/Interface.hpp>
#include <XAD/Literals.hpp>
#include <XAD/MathFunctions.hpp>
#include <XAD/Tape.hpp>
#include <XAD/TapeContainer.hpp>
#include <XAD/Traits.hpp>
#include <XAD/UnaryExpr.hpp>
#include <XAD/UnaryFunctors.hpp>
#include <XAD/UnaryMathFunctors.hpp>
#include <XAD/UnaryOperators.hpp>
#include <XAD/Vec.hpp>
#include <XAD/Version.hpp>

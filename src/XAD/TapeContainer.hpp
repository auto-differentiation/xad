/*******************************************************************************

   Selects container to use for the tape.

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

#include <XAD/ChunkContainer.hpp>
#ifdef XAD_REDUCED_MEMORY
#include <XAD/OperationsContainer.hpp>
#else
#include <XAD/OperationsContainerPaired.hpp>
#endif

#include <utility>

namespace xad
{

template <class T, class S = unsigned>
struct TapeContainerTraits
{
    using statements_type = ChunkContainer<std::pair<S, S>>;
#ifdef XAD_REDUCED_MEMORY
    using operations_type = OperationsContainer<T, S>;
#else
    using operations_type = OperationsContainerPaired<T, S>;
#endif
};

}  // namespace xad

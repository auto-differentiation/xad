/*******************************************************************************

   Implementation of helper traits.

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
#include <type_traits>

namespace xad
{

namespace detail
{

template <typename U>
static auto has_begin_impl(int) -> decltype(std::declval<U>().begin(), std::true_type{});

template <typename U>
static std::false_type has_begin_impl(...);

template <typename U>
struct has_begin : decltype(has_begin_impl<U>(0))
{
};

}  // namespace detail
}  // namespace xad

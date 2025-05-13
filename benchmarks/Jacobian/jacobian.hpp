/*******************************************************************************

   Jacobian benchmark functions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>

#include <functional>

template <typename T>
std::function<T(std::vector<T> &)> make_foo() {
    return [](std::vector<T> &x) -> T {
        return {sin(x[0] + x[1]), sin(x[1] + x[2]), cos(x[2] + x[3]), cos(x[3] + x[0])};
    };
}

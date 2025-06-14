/*******************************************************************************

   Implements the Eigen compatibility layer for XAD.

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

#ifdef _MSC_VER
#define EIGEN_HAS_STD_RESULT_OF 0
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

#include <type_traits>

namespace Eigen
{
namespace internal
{

#ifdef _MSC_VER

// https://gitlab.com/libeigen/eigen/-/issues/1894
template <typename T>
struct result_of
{
#if defined(__cplusplus) && __cplusplus >= 201703L
    typedef typename std::invoke_result<T>::type type1;
#else
    typedef typename std::result_of<T>::type type1;
#endif
};

#endif

}  // namespace internal
}  // namespace Eigen

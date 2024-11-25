/*******************************************************************************

   Utility macro declarations.

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

#include <XAD/Config.hpp>

namespace xad
{
namespace detail
{
template <class T>
void ignore_unused_variable(const T&)
{
}
}  // namespace detail
}  // namespace xad

#define XAD_UNUSED_VARIABLE(x) ::xad::detail::ignore_unused_variable(x)

#ifdef _WIN32
#define XAD_FORCE_INLINE __forceinline
#define XAD_NEVER_INLINE __declspec(noinline)
#else
#define XAD_FORCE_INLINE __attribute__((always_inline)) inline
#define XAD_NEVER_INLINE __attribute__((noinline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define XAD_LIKELY(x) __builtin_expect(!!(x), 1)
#define XAD_UNLIKELY(x) __builtin_expect(!!(x), 0)
#if defined(__GNUC__) && __GNUC__ >= 9
#define XAD_VERY_LIKELY(x) __builtin_expect_with_probability(!!(x), 1, 0.999)
#define XAD_VERY_UNLIKELY(x) __builtin_expect_with_probability(!!(x), 0, 0.999)
#else
#define XAD_VERY_LIKELY(x) XAD_LIKELY(x)
#define XAD_VERY_UNLIKELY(x) XAD_UNLIKELY(x)
#endif
#else
#define XAD_LIKELY(x) (x)
#define XAD_UNLIKELY(x) (x)
#define XAD_VERY_LIKELY(x) (x)
#define XAD_VERY_UNLIKELY(x) (x)
#endif

#ifdef XAD_USE_STRONG_INLINE
#define XAD_INLINE XAD_FORCE_INLINE
#else
#define XAD_INLINE inline
#endif

#ifdef XAD_NO_THREADLOCAL
#define XAD_THREAD_LOCAL
#else
// we can't use thread_local here, as MacOS has an issue with that
#ifdef _WIN32
#define XAD_THREAD_LOCAL __declspec(thread)
#else
#define XAD_THREAD_LOCAL __thread
#endif
#endif

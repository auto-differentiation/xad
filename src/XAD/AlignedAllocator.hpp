/*******************************************************************************

   TODO

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

#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>


#if defined(__APPLE__) || defined(__ANDROID__) ||                                                  \
    (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#include <cstdlib>

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

namespace xad
{
namespace detail
{
inline void* aligned_alloc(size_t alignment, size_t size)
{
    size = std::max(size, alignment);
#if defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_16)
#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_16
    // For C++14, usr/include/malloc/_malloc.h declares aligned_alloc()) only
    // with the MacOSX11.0 SDK in Xcode 12 (which is what adds
    // MAC_OS_X_VERSION_10_16), even though the function is marked
    // availabe for 10.15. That's why the preprocessor checks for 10.16 but
    // the __builtin_available checks for 10.15.
    // People who use C++17 could call aligned_alloc with the 10.15 SDK already.
    if (__builtin_available(macOS 10.15, *))
        return ::aligned_alloc(alignment, size);
#endif
#endif
    // alignment must be >= sizeof(void*)
    alignment = std::max(alignment, sizeof(void*));

    void* pointer;
    if (posix_memalign(&pointer, alignment, size) == 0)
        return pointer;

    return nullptr;
}
inline void aligned_free(void* p) { free(p); }

}  // namespace detail
}  // namespace xad
#elif defined(_WIN32)
namespace xad
{
namespace detail
{
inline void* aligned_alloc(size_t alignment, size_t size)
{
    size = std::max(size, alignment);
    return ::_aligned_malloc(size, alignment);
}
inline void aligned_free(void* p) { ::_aligned_free(p); }
}  // namespace detail
}  // namespace xad
#else
namespace xad
{
namespace detail
{
inline void* aligned_alloc(size_t alignment, size_t size)
{
    size = std::max(size, alignment);
    return ::aligned_alloc(alignment, size);
}
inline void aligned_free(void* p) { free(p); }
}  // namespace detail
}  // namespace xad
#endif


namespace xad
{
namespace detail
{

struct AlignedAllocator {
    static void* aligned_alloc(std::size_t alignment, std::size_t size) {
        return detail::aligned_alloc(alignment, size);
    }

    static void aligned_free(void* ptr) {
        return detail::aligned_free(ptr);
    }

    void operator()(void* ptr) const { aligned_free(ptr); }
};


} // namespace detail
} // namespace xad
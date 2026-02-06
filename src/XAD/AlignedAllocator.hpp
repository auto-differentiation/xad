/*******************************************************************************

   Cross platform helpers for aligned memory allocations.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

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

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

namespace xad
{
namespace detail
{

struct AlignedAllocator {
    static inline void* aligned_alloc(std::size_t alignment, std::size_t size) {
        size = std::max(size, alignment);

    #if defined(_WIN32)
        return ::_aligned_malloc(size, alignment);

    #elif defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_16)
    #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_16
    // For C++14, usr/include/malloc/_malloc.h declares aligned_alloc()) only
    // with the MacOSX11.0 SDK in Xcode 12 (which is what adds
    // MAC_OS_X_VERSION_10_16), even though the function is marked
    // availabe for 10.15. That's why the preprocessor checks for 10.16 but
    // the __builtin_available checks for 10.15.
    // People who use C++17 could call aligned_alloc with the 10.15 SDK already.
        if (__builtin_available(macOS 10.15, *)) {
            return ::aligned_alloc(alignment, size);
        }
    #endif
        alignment = std::max(alignment, sizeof(void*));

        void* pointer = nullptr;
        if (posix_memalign(&pointer, alignment, size) == 0) {
            return pointer;
        }
        return nullptr;

    #elif defined(__ANDROID__) || (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
        void* pointer = nullptr;
        if (posix_memalign(&pointer, alignment, size) == 0) {
            return pointer;
        }
        return nullptr;

    #else
        return ::aligned_alloc(alignment, size);

    #endif
    }

    static inline void aligned_free(void* ptr) {
    #if defined(_WIN32)
        ::_aligned_free(ptr);
    #else
        free(ptr);
    #endif
    }

    // The () operator is defined such that AlignedAllocator can be
    // used as a deleter for std:: smart pointers.
    void operator()(void* ptr) const { this->aligned_free(ptr); }
};


} // namespace detail
} // namespace xad
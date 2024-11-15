/*******************************************************************************

   Container storing continuous chunks of data.

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

#include <XAD/Exceptions.hpp>
#include <XAD/Macros.hpp>
#include <type_traits>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <memory>

// cross-platform aligned (de-)allocation

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
    if (size < alignment)
        size = alignment;
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
    if (alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

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
    if (size < alignment)
        size = alignment;
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
    if (size < alignment)
        size = alignment;
    return ::aligned_alloc(alignment, size);
}
inline void aligned_free(void* p) { free(p); }
}  // namespace detail
}  // namespace xad
#endif

namespace xad
{

template <class T, std::size_t ChunkSize = 1024U * 1024U * 8U>
class ChunkContainer
{
  public:
    typedef std::size_t size_type;
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    static const std::size_t chunk_size = ChunkSize;

    static const int ALIGNMENT = 128;

    ChunkContainer() : chunk_(0), idx_(0)
    {
        chunkList_.reserve(32);
        reserve(1);
    }

    ChunkContainer(ChunkContainer&& o) noexcept
        : chunkList_(std::move(o.chunkList_)), chunk_(o.chunk_), idx_(o.idx_)
    {
    }

    ChunkContainer& operator=(ChunkContainer&& o)
    {
        _free_memory();
        chunkList_ = std::move(o.chunkList_);
        chunk_ = o.chunk_;
        idx_ = o.idx_;
        return *this;
    }

    ChunkContainer(const ChunkContainer&) = delete;
    ChunkContainer& operator=(const ChunkContainer&) = delete;

    ~ChunkContainer() { _free_memory(); }

    void reserve(size_type s)
    {
        size_type nc = getNumChunks(s);
        if (nc > chunkList_.size())
        {
            size_type d = nc - chunkList_.size();
            for (size_type i = 0; i < d; ++i)
            {
                char* chunk = reinterpret_cast<char*>(
                    detail::aligned_alloc(ALIGNMENT, sizeof(value_type) * chunk_size));
                if (chunk == NULL)
                    throw std::bad_alloc();
                chunkList_.push_back(chunk);
            }
        }
    }

    template <bool, bool = true>
    struct destructAllImpl
    {
        // version that needs destruction
        static void make(ChunkContainer* c, size_type start, size_type end)
        {
            size_type ncstart = getHighPart(start);
            size_type lcstart = getLowPart(start);
            size_type ncend = getHighPart(end);
            size_type lcend = getLowPart(end);
            if (ncstart == ncend)
            {
                for (size_type i = lcstart; i < lcend; ++i)
                    reinterpret_cast<value_type*>(c->chunkList_[ncstart])[i].~value_type();
                return;
            }

            for (size_type i = lcstart; i < chunk_size; ++i)
                reinterpret_cast<value_type*>(c->chunkList_[ncstart])[i].~value_type();

            for (size_type i = ncstart + 1; i < ncend; ++i)
                for (size_type j = 0; j < chunk_size; ++j)
                    reinterpret_cast<value_type*>(c->chunkList_[i])[j].~value_type();

            for (size_type i = 0, e = lcend; i != e; ++i)
                reinterpret_cast<value_type*>(c->chunkList_[ncend])[i].~value_type();
        }
    };

    template <bool dummy>
    struct destructAllImpl<true, dummy>
    {
        static void make(ChunkContainer*, size_type, size_type) {}
    };

    void clear()
    {
        destructAllImpl<std::is_trivially_destructible<value_type>::value>::make(this, size_type(0),
                                                                                 size());
        chunk_ = idx_ = 0;
    }

    size_type capacity() const { return getNumElements(chunkList_.capacity()); }

    size_type size() const { return chunk_ * chunk_size + idx_; }

    bool empty() const { return chunk_ == 0 && idx_ == 0; }

    void uninitialized_extend(size_type i)
    {
        reserve(size() + i);
        idx_ += i;
        if (idx_ > chunk_size)
        {
            ++chunk_;
            idx_ = idx_ - chunk_size;
        }
    }

    XAD_FORCE_INLINE void push_back(const_reference v)
    {
        if (idx_ == chunk_size)
            check_space();

        ::new (reinterpret_cast<value_type*>(chunkList_[chunk_]) + idx_) value_type(v);
        ++idx_;
    }

    template <class... Args>
    void emplace_back(Args&&... args)
    {
        if (idx_ == chunk_size)
            check_space();
        ::new (chunkList_[chunk_] + idx_ * sizeof(value_type))
            value_type(std::forward<Args>(args)...);
        ++idx_;
    }

    void resize(size_type s, const_reference v = value_type())
    {
        if (s == size())
            return;

        if (s < size())
        {
            // clear the remainder - for now without destruct
            destructAllImpl<std::is_trivially_destructible<value_type>::value>::make(this, s,
                                                                                     size());
            chunk_ = getHighPart(s);
            idx_ = getLowPart(s);
            return;
        }

        // now we have something bigger
        reserve(s);
        size_type nc_new = getHighPart(s);
        size_type nc_old = chunk_;
        size_type ix = idx_;
        for (size_type i = nc_old; i < nc_new; ++i)
        {
            std::uninitialized_fill_n(reinterpret_cast<value_type*>(chunkList_[i]) + ix,
                                      chunk_size - ix, v);
            ix = 0;
        }

        size_type lc = getLowPart(s);  // maybe 0
        std::uninitialized_fill_n(reinterpret_cast<value_type*>(chunkList_[nc_new]) + ix, lc - ix,
                                  v);
        idx_ = lc;
        chunk_ = nc_new;
    }

    template <class It>
    void append(It first, It last)
    {
        auto n = static_cast<size_type>(std::distance(first, last));
        assert(n <= chunk_size);
#if defined(_MSC_VER)
        if (idx_ + n <= chunk_size)
#else
        if (__builtin_expect(idx_ + n <= chunk_size, 1))
#endif
        {
            auto dst = reinterpret_cast<value_type*>(chunkList_[chunk_]) + idx_;
            for (size_type i = 0; i < n; ++i) ::new (dst++) value_type(*first++);
            idx_ += n;
        }
        else
        {
            assert(chunk_size - idx_ < n);
            auto dst = reinterpret_cast<value_type*>(chunkList_[chunk_]) + idx_;
            auto endn = chunk_size - idx_;
            for (size_type i = 0; i < endn; ++i) ::new (dst++) value_type(*first++);
            idx_ = chunk_size;
            if (idx_ == chunk_size)
                check_space();  // appends a chunk, moves idx_ / chunk_
            endn = static_cast<size_type>(std::distance(first, last));
            dst = reinterpret_cast<value_type*>(chunkList_[chunk_]);
            for (size_type i = 0; i < endn; ++i) ::new (dst++) value_type(*first++);
            idx_ += endn;
        }
    }

    typedef value_type** chunk_iterator;
    typedef std::reverse_iterator<chunk_iterator> chunk_reverse_iterator;
    chunk_iterator chunk_begin()
    {
        return chunk_iterator(reinterpret_cast<value_type**>(&chunkList_[0]));
    }
    chunk_iterator chunk_end() { return chunk_begin() + chunkList_.size(); }
    chunk_reverse_iterator chunk_rbegin() { return chunk_reverse_iterator(chunk_end()); }
    chunk_reverse_iterator chunk_rend() { return chunk_reverse_iterator(chunk_begin()); }
    /*
    size_type chunk_size() const {
      return chunkList_.size();
    }
    */

    struct iterator
    {
        pointer point_;
        pointer next_chunk_;
        int space_left_;
        explicit iterator(pointer point = nullptr, int space_left = 0, pointer next = NULL)
            : point_(point), next_chunk_(next), space_left_(space_left)
        {
        }

        iterator& operator++()
        {
            ++point_;
            if (--space_left_ == 0)
            {
                point_ = next_chunk_;
                space_left_ = chunk_size;
            }
            return *this;
        }
        iterator operator++(int)
        {
            iterator r(*this);
            ++(*this);
            return r;
        }
        reference operator*() const { return *point_; }
        pointer operator->() const { return point_; }
        bool operator==(const iterator& o) const { return point_ == o.point_; }
        bool operator!=(const iterator& o) const { return point_ != o.point_; }
    };

    iterator iterator_at(size_type i)
    {
        auto nc = getHighPart(i);
        auto lc = getLowPart(i);
        return iterator(&reinterpret_cast<value_type*>(chunkList_[nc])[lc], int(chunk_size - lc),
                        reinterpret_cast<value_type*>(*(&chunkList_[nc] + 1)));
    }

    void assign(size_type s, const_reference v = value_type())
    {
        if (s == 0)
        {
            clear();
            return;
        }

        check_space(s);
        size_type nc = getHighPart(s);
        size_type lc = getLowPart(s);

        if (s <= size())
        {
            if (s < size())
                destructAllImpl<std::is_trivially_destructible<value_type>::value>::make(this, s,
                                                                                         size());
            // all assignments
            for (size_type i = 0; i < nc; ++i)
                std::fill(reinterpret_cast<value_type*>(chunkList_[i]),
                          reinterpret_cast<value_type*>(chunkList_[i]) + chunk_size, v);
            std::fill(reinterpret_cast<value_type*>(chunkList_[chunk_]),
                      reinterpret_cast<value_type*>(chunkList_[chunk_]) + lc, v);
            chunk_ = nc;
            idx_ = lc;

            return;
        }

        // here we have something bigger than it was - assign existing and fill rest
        for (int i = 0; i < chunk_; ++i)
            std::fill(reinterpret_cast<value_type*>(chunkList_[i]),
                      reinterpret_cast<value_type*>(chunkList_[i]) + chunk_size, v);
        std::fill(reinterpret_cast<value_type*>(chunkList_[chunk_]),
                  reinterpret_cast<value_type*>(chunkList_[chunk_]) + idx_, v);

        size_type fillsize = nc > chunk_ ? chunk_size : lc;
        std::uninitialized_fill(reinterpret_cast<value_type*>(chunkList_[chunk_]) + idx_,
                                reinterpret_cast<value_type*>(chunkList_[chunk_]) + fillsize, v);
        for (size_type i = chunk_ + 1; i < nc; ++i)
            std::uninitialized_fill_n(reinterpret_cast<value_type*>(chunkList_[i]), chunk_size, v);
        if (nc > chunk_)
            std::uninitialized_fill_n(reinterpret_cast<value_type*>(chunkList_[nc]), lc, v);
        chunk_ = nc;
        idx_ = lc;
        return;
    }

    reference operator[](size_type i)
    {
        return reinterpret_cast<pointer>(chunkList_[getHighPart(i)])[getLowPart(i)];
    }

    const_reference operator[](size_type i) const
    {
        return reinterpret_cast<const_pointer>(chunkList_[getHighPart(i)])[getLowPart(i)];
    }

    static size_type getNumChunks(size_type i)
    {
        return getHighPart(i) + size_type(getLowPart(i) > 0U);
    }
    static size_type getHighPart(size_type i) { return i / chunk_size; }
    static size_type getLowPart(size_type i) { return i % chunk_size; }
    static size_type getNumElements(size_type chunks) { return chunks * chunk_size; }

  private:
    void check_space()
    {
        // if (idx_ == chunk_size)
        // {
            if (chunk_ == chunkList_.size() - 1)
            {
                char* chunk = reinterpret_cast<char*>(
                    detail::aligned_alloc(ALIGNMENT, sizeof(value_type) * chunk_size));
                if (chunk == NULL)
                    throw std::bad_alloc();
                chunkList_.push_back(chunk);
            }
            ++chunk_;
            idx_ = 0;
        // }
    }

    void check_space(size_type i) { reserve(chunk_ * chunk_size + idx_ + i); }

    void _free_memory()
    {
        clear();
        for (auto& c : chunkList_)
        {
            detail::aligned_free(c);
        }
    }

    std::vector<char*> chunkList_;
    size_type chunk_, idx_;
};
}  // namespace xad

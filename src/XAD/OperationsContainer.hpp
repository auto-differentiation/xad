/*******************************************************************************

   Container storing operations on tape - slots and multipliers.

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

#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

namespace xad
{

namespace detail
{
struct AlignedDeleter
{
    void operator()(void* ptr) const { aligned_free(ptr); }
};

struct AlignedAllocHelper {
    static void* aligned_alloc(std::size_t alignment, std::size_t size) {
        return detail::aligned_alloc(alignment, size);
    }
};

struct NullAlignedAllocHelper {
    static void* aligned_alloc(std::size_t, std::size_t) {
        return nullptr; // simulate failure
    }
};


#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL
// Make a checked iterator to avoid MSVC warnings.
template <typename T>
using checked_ptr = stdext::checked_array_iterator<T*>;
template <typename T>
checked_ptr<T> make_checked(T* p, size_t size)
{
    return {p, size};
}
#else
template <typename T>
using checked_ptr = T*;
template <typename T>
inline T* make_checked(T* p, size_t)
{
    return p;
}
#endif

}  // namespace detail

template <typename T, typename S, std::size_t ChunkSize = 1024U * 1024U * 8U, class AllocHelper = detail::AlignedAllocHelper>
class OperationsContainer
{
  public:
    using size_type = std::size_t;
    using mul_type = T;
    using slot_type = S;
    static constexpr std::size_t chunk_size = ChunkSize;
    static constexpr std::size_t ALIGNMENT = 128;
    static_assert(std::is_integral<slot_type>::value, "S type must be an integral type");

    OperationsContainer()
    {
        chunks_.reserve(64);
        addChunks(1);
    }
    OperationsContainer(const OperationsContainer&) = delete;
    OperationsContainer(OperationsContainer&&) = default;
    OperationsContainer& operator=(OperationsContainer&&) = default;
    OperationsContainer& operator=(const OperationsContainer&) = delete;
    ~OperationsContainer() { destruct_elements(0); }

    bool empty() const { return chunk_ == 0 && idx_ == 0; }
    size_type size() const { return chunk_ * ChunkSize + idx_; }
    size_type capacity() const { return chunks_.size() * ChunkSize; }
    size_type chunks() const { return chunks_.size(); }

    void reserve(size_type s)
    {
        if (XAD_VERY_LIKELY(capacity() >= s))
        {
            return;
        }
        auto newChunks = (s + ChunkSize - 1) / ChunkSize - chunks_.size();
        addChunks(newChunks);
    }

    void resize(size_type s)
    {
        reserve(s);
        if (s < size())
        {
            destruct_elements(s);
        }
        else
        {
            construct_elements(s);
        }
        chunk_ = s / ChunkSize;
        idx_ = s % ChunkSize;
    }

    void clear()
    {
        destruct_elements(0);
        chunk_ = 0;
        idx_ = 0;
    }

    void push_back(T multiplier, S slot)
    {
        if (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            addChunks(1);
            idx_ = 0;
            ++chunk_;
        }
        push_back_unsafe(std::move(multiplier), std::move(slot));
    }

    XAD_FORCE_INLINE void push_back_unsafe(T multiplier, S slot)
    {
        if (XAD_VERY_UNLIKELY(idx_ == chunk_size))
        {
            ++chunk_;
            idx_ = 0;
        }
        ::new (&mul_chunk(chunk_)[idx_]) T(std::move(multiplier));
        ::new (&slot_chunk(chunk_)[idx_]) S(std::move(slot));
        ++idx_;
    }

    template <class MulIt, class SlotIt>
    XAD_FORCE_INLINE void append_n(MulIt muls, SlotIt slots, size_type n)
    {
        auto items = (std::min)(ChunkSize - idx_, n);
        std::uninitialized_copy_n(detail::make_checked(muls, items), items,
                                  detail::make_checked(mul_chunk(chunk_), ChunkSize) + idx_);
        std::uninitialized_copy_n(detail::make_checked(slots, items), items,
                                  detail::make_checked(slot_chunk(chunk_), ChunkSize) + idx_);
        idx_ += items;

        while (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            addChunks(1);
            ++chunk_;
            idx_ = 0;
            n -= items;
            muls += items;
            slots += items;

            items = (std::min)(ChunkSize, n);
            std::uninitialized_copy_n(detail::make_checked(muls, items), items,
                                      detail::make_checked(mul_chunk(chunk_), ChunkSize));
            std::uninitialized_copy_n(detail::make_checked(slots, items), items,
                                      detail::make_checked(slot_chunk(chunk_), ChunkSize));
            idx_ += items;
        }
    }

    XAD_FORCE_INLINE std::pair<mul_type, slot_type> operator[](size_type n) const
    {
        auto chunk = n / ChunkSize;
        auto idx = n % ChunkSize;
        return {mul_chunk(chunk)[idx], slot_chunk(chunk)[idx]};
    }

    // Apply the given function with the signature void f(mul_type, slot_type) to
    // all elements between startidx and endidx
    template <class Func>
    void for_each(size_type startidx, size_type endidx, Func f) const
    {
        size_type start_chunk = startidx / ChunkSize;
        size_type start_idx = startidx % ChunkSize;
        size_type end_chunk = endidx / ChunkSize;
        size_type end_idx = endidx % ChunkSize;

        size_type e1 = end_chunk != start_chunk ? ChunkSize : end_idx;
        auto chk_mul = mul_chunk(start_chunk);
        auto chk_slot = slot_chunk(start_chunk);
        for (auto i = start_idx; i < e1; ++i)
        {
            f(chk_mul[i], chk_slot[i]);
        }

        if (XAD_VERY_LIKELY(start_chunk == end_chunk))
            return;

        size_type e2 = end_idx;
        if (e2 == 0)
            return;
        chk_mul = mul_chunk(end_chunk);
        chk_slot = slot_chunk(end_chunk);
        for (size_type i = 0; i < e2; ++i)
        {
            f(chk_mul[i], chk_slot[i]);
        }
    }

  private:
    XAD_FORCE_INLINE mul_type* mul_chunk(size_type chunk)
    {
        return reinterpret_cast<mul_type*>(chunks_[chunk].get());
    }
    XAD_FORCE_INLINE slot_type* slot_chunk(size_type chunk)
    {
        constexpr std::size_t SLOT_OFFSET =
            ((ChunkSize * sizeof(mul_type) + alignof(slot_type) - 1) / alignof(slot_type)) *
            sizeof(slot_type);

        return reinterpret_cast<slot_type*>(chunks_[chunk].get() + SLOT_OFFSET);
    }
    XAD_FORCE_INLINE const mul_type* mul_chunk(size_type chunk) const
    {
        return reinterpret_cast<const mul_type*>(chunks_[chunk].get());
    }
    XAD_FORCE_INLINE const slot_type* slot_chunk(size_type chunk) const
    {
        constexpr std::size_t SLOT_OFFSET =
            ((ChunkSize * sizeof(mul_type) + alignof(slot_type) - 1) / alignof(slot_type)) *
            sizeof(slot_type);
        return reinterpret_cast<const slot_type*>(chunks_[chunk].get() + SLOT_OFFSET);
    }

    void addChunks(size_type newChunks)
    {
        constexpr std::size_t SLOT_OFFSET =
            ((ChunkSize * sizeof(mul_type) + alignof(slot_type) - 1) / alignof(slot_type)) *
            sizeof(slot_type);
        constexpr std::size_t CHUNK_BYTES = SLOT_OFFSET + sizeof(slot_type) * ChunkSize;

        for (size_type i = 0; i < newChunks; ++i)
        {
            auto chunk = AllocHelper::aligned_alloc(ALIGNMENT, CHUNK_BYTES);
            if (chunk == nullptr)
                throw std::bad_alloc();
            chunks_.emplace_back(reinterpret_cast<char*>(chunk));
        }
    }

    template <bool trivial, bool = true>
    struct destructHelper
    {
        static void destr(OperationsContainer* cont, size_type start)
        {
            auto start_chunk = start / ChunkSize;
            auto start_idx = start % ChunkSize;
            auto end_chunk = cont->size() / ChunkSize;
            auto end_idx = cont->size() % ChunkSize;

            if (start_chunk == end_chunk)
            {
                auto chk = cont->mul_chunk(start_chunk);
                for (size_type i = start_idx; i < end_idx; ++i)
                {
                    chk[i].~mul_type();
                }
                return;
            }

            auto chk = cont->mul_chunk(start_chunk);
            for (size_type i = start_idx; i < ChunkSize; ++i)
            {
                chk[i].~mul_type();
            }

            for (size_type c = start_chunk + 1; c < end_chunk; ++c)
            {
                chk = cont->mul_chunk(c);
                for (size_type i = 0; i < ChunkSize; ++i)
                {
                    chk[i].~mul_type();
                }
            }

            if (end_idx == 0) {
                return;
            }

            chk = cont->mul_chunk(end_chunk);
            for (size_type i = 0; i < end_idx; ++i)
            {
                chk[i].~mul_type();
            }
        }
    };

    template <bool dummy>
    struct destructHelper<true, dummy>
    {
        XAD_FORCE_INLINE static void destr(OperationsContainer*, size_type) {}
    };

    XAD_FORCE_INLINE void destruct_elements(size_type start)
    {
        destructHelper<std::is_trivially_destructible<mul_type>::value>::destr(this, start);
    }

    void construct_elements(size_type new_size)
    {
        auto start_chunk = chunk_;
        auto end_chunk = new_size / ChunkSize;
        auto start_idx = idx_;
        auto end_idx = new_size % ChunkSize;

        if (start_chunk == end_chunk)
        {
            auto chk = mul_chunk(start_chunk);
            auto chks = slot_chunk(start_chunk);
            for (size_type i = start_idx; i < end_idx; ++i)
            {
                ::new (&chk[i]) mul_type();
                ::new (&chks[i]) slot_type();
            }
            return;
        }

        auto chk = mul_chunk(start_chunk);
        auto chks = slot_chunk(start_chunk);
        for (size_type i = start_idx; i < ChunkSize; ++i)
        {
            ::new (&chk[i]) mul_type();
            ::new (&chks[i]) slot_type();
        }

        for (size_type c = start_chunk + 1; c < end_chunk; ++c)
        {
            chk = mul_chunk(c);
            chks = slot_chunk(c);
            for (size_type i = 0; i < ChunkSize; ++i)
            {
                ::new (&chk[i]) mul_type();
                ::new (&chks[i]) slot_type();
            }
        }

        if (end_idx == 0) {
            return;
        }
        chk = mul_chunk(end_chunk);
        chks = slot_chunk(end_chunk);
        for (size_type i = 0; i < end_idx; ++i)
        {
            ::new (&chk[i]) mul_type();
            ::new (&chks[i]) slot_type();
        }
    }

    std::vector<std::unique_ptr<char, detail::AlignedDeleter>> chunks_;
    size_type idx_ = 0;
    size_type chunk_ = 0;
};

}  // namespace xad

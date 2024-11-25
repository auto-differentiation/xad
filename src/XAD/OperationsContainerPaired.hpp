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

#include <XAD/OperationsContainer.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace xad
{

template <typename T, typename S, std::size_t ChunkSize = 1024U * 1024U * 8U>
class OperationsContainerPaired
{
  public:
    using size_type = std::size_t;
    using mul_type = T;
    using slot_type = S;
    static constexpr std::size_t ALIGNMENT = 128;
    static constexpr std::size_t chunk_size = ChunkSize;
    static_assert(std::is_integral<slot_type>::value, "S type must be an integral type");

    OperationsContainerPaired()
    {
        chunks_.reserve(64);
        addChunks(1);
    }
    OperationsContainerPaired(const OperationsContainerPaired&) = delete;
    OperationsContainerPaired(OperationsContainerPaired&&) = default;
    OperationsContainerPaired& operator=(OperationsContainerPaired&&) = default;
    OperationsContainerPaired& operator=(const OperationsContainerPaired&) = delete;
    ~OperationsContainerPaired() { destruct_elements(0); }

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
        if (XAD_LIKELY(s < size()))
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

    void push_back(const T& multiplier, const S& slot)
    {
        if (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            addChunks(1);
            idx_ = 0;
            ++chunk_;
        }
        push_back_unsafe(multiplier, slot);
    }

    XAD_FORCE_INLINE void push_back_unsafe(T multiplier, S slot)
    {
        if (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            ++chunk_;
            idx_ = 0;
        }
        ::new (&chunk(chunk_)[idx_]) std::pair<T, S>(std::move(multiplier), slot);
        ++idx_;
    }

#if defined(__GNUC__) && !defined(__clang__)
// GCC 12 flags this warning in the code below, although it is perfectly safe
// and tested
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    template <class MulIt, class SlotIt>
    XAD_FORCE_INLINE void append_n(MulIt muls, SlotIt slots, size_type n)
    {
        auto items = (std::min)(ChunkSize - idx_, n);
        auto dst = chunk(chunk_) + idx_;
        auto dst_end = dst + items;
        for (; dst != dst_end; ++dst) ::new (dst) std::pair<T, S>(*muls++, *slots++);
        idx_ += items;

        while (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            addChunks(1);
            ++chunk_;
            idx_ = 0;
            n -= items;

            items = (std::min)(ChunkSize, n);
            dst = chunk(chunk_);
            dst_end = dst + items;
            for (; dst != dst_end; ++dst) ::new (dst) std::pair<T, S>(*muls++, *slots++);
            idx_ += items;
        }
    }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    XAD_FORCE_INLINE std::pair<mul_type, slot_type> operator[](size_type n) const
    {
        auto ck = n / ChunkSize;
        auto idx = n % ChunkSize;
        return chunk(ck)[idx];
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
        auto chk = chunk(start_chunk);
        for (auto i = start_idx; i < e1; ++i)
        {
            f(chk[i].first, chk[i].second);
        }

        if (XAD_VERY_LIKELY(start_chunk == end_chunk))
            return;

        size_type e2 = end_idx;
        if (e2 == 0) {
            return;
        }
        chk = chunk(end_chunk);
        for (size_type i = 0; i < e2; ++i)
        {
            f(chk[i].first, chk[i].second);
        }
    }

  private:
    XAD_FORCE_INLINE std::pair<mul_type, slot_type>* chunk(size_type chunk)
    {
        return reinterpret_cast<std::pair<mul_type, slot_type>*>(chunks_[chunk].get());
    }
    XAD_FORCE_INLINE const std::pair<mul_type, slot_type>* chunk(size_type chunk) const
    {
        return reinterpret_cast<const std::pair<mul_type, slot_type>*>(chunks_[chunk].get());
    }

    void addChunks(size_type newChunks)
    {
        for (size_type i = 0; i < newChunks; ++i)
        {
            auto chunk = detail::aligned_alloc(ALIGNMENT,
                                               ChunkSize * sizeof(std::pair<mul_type, slot_type>));
            if (chunk == nullptr)
                throw std::bad_alloc();
            chunks_.emplace_back(reinterpret_cast<char*>(chunk));
        }
    }

    template <bool trivial, bool = true>
    struct destructHelper
    {
        static void destr(OperationsContainerPaired* cont, size_type start)
        {
            auto start_chunk = start / ChunkSize;
            auto start_idx = start % ChunkSize;
            auto end_chunk = cont->size() / ChunkSize;
            auto end_idx = cont->size() % ChunkSize;

            if (start_chunk == end_chunk)
            {
                auto chk = cont->chunk(start_chunk);
                for (size_type i = start_idx; i < end_idx; ++i)
                {
                    chk[i].first.~mul_type();
                }
                return;
            }

            auto chk = cont->chunk(start_chunk);
            for (size_type i = start_idx; i < ChunkSize; ++i)
            {
                chk[i].first.~mul_type();
            }

            for (size_type c = start_chunk + 1; c < end_chunk; ++c)
            {
                chk = cont->chunk(c);
                for (size_type i = 0; i < ChunkSize; ++i)
                {
                    chk[i].first.~mul_type();
                }
            }

            if (end_idx == 0) {
                return;
            }
            chk = cont->chunk(end_chunk);
            for (size_type i = 0; i < end_idx; ++i)
            {
                chk[i].first.~mul_type();
            }
        }
    };

    template <bool dummy>
    struct destructHelper<true, dummy>
    {
        static void destr(OperationsContainerPaired*, size_type) {}
    };

    void destruct_elements(size_type start)
    {
        destructHelper<std::is_trivially_destructible<mul_type>::value>::destr(this, start);
    }

    void construct_elements(size_type new_size)
    {
        auto start_chunk = chunk_;
        auto end_chunk = new_size / ChunkSize;
        auto start_idx = idx_;
        auto end_idx = new_size % ChunkSize;
        using vtype = std::pair<mul_type, slot_type>;

        if (start_chunk == end_chunk)
        {
            auto chk = chunk(start_chunk);
            for (size_type i = start_idx; i < end_idx; ++i)
            {
                ::new (&chk[i]) vtype();
            }
            return;
        }

        auto chk = chunk(start_chunk);
        for (size_type i = start_idx; i < ChunkSize; ++i)
        {
            ::new (&chk[i]) vtype();
        }

        for (size_type c = start_chunk + 1; c < end_chunk; ++c)
        {
            chk = chunk(c);
            for (size_type i = 0; i < ChunkSize; ++i)
            {
                ::new (&chk[i]) vtype();
            }
        }

        if (end_idx == 0) {
            return;
        }
        chk = chunk(end_chunk);
        for (size_type i = 0; i < end_idx; ++i)
        {
            ::new (&chk[i]) vtype();
        }
    }

    std::vector<std::unique_ptr<char, detail::AlignedDeleter>> chunks_;
    size_type idx_ = 0;
    size_type chunk_ = 0;
};

}  // namespace xad

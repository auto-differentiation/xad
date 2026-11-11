/*******************************************************************************

   Container storing operations on tape - slots and multipliers.

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

#include <XAD/ChunkContainer.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace xad
{

// Space reserved in an operations container: the entries go to multipliers[pos], slots[pos] and on,
// in the arrays of the current chunk; multipliers is null if there is none
template <typename T, typename S>
struct OperationsRun
{
    T* multipliers;
    S* slots;
    std::size_t pos;
};

template <typename T, typename S, std::size_t ChunkSize = 1024U * 1024U * 8U,
          class AllocHelper = detail::AlignedAllocator>
class OperationsContainer
{
  public:
    using size_type = std::size_t;
    using mul_type = T;
    using slot_type = S;
    using run_type = OperationsRun<T, S>;
    static constexpr std::size_t chunk_size = ChunkSize;
    static constexpr std::size_t ALIGNMENT = 128;
    static_assert(std::is_integral<slot_type>::value, "S type must be an integral type");

    OperationsContainer()
    {
        chunks_.reserve(64);
        addChunks(1);
    }
    OperationsContainer(const OperationsContainer&) = delete;
    OperationsContainer(OperationsContainer&& o) noexcept
        : chunks_(std::move(o.chunks_)), idx_(o.idx_), chunk_(o.chunk_)
    {
        o.idx_ = o.chunk_ = 0;
    }
    OperationsContainer& operator=(OperationsContainer&& o) noexcept
    {
        if (this != &o)
        {
            destruct_elements(0);
            chunks_ = std::move(o.chunks_);
            idx_ = o.idx_;
            chunk_ = o.chunk_;
            o.idx_ = o.chunk_ = 0;
        }
        return *this;
    }
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
        if (chunks_.empty())  // moved from
            addChunks(1);
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

    // Reserve space for a run of n entries in the current chunk, so callers can write their
    // partials straight into the tape, from run.pos on, then commit_run the position after the
    // last one written. Gives null pointers when the run would cross a chunk boundary.
    XAD_FORCE_INLINE run_type try_reserve_run(size_type n)
    {
        if (XAD_VERY_UNLIKELY(idx_ + n > ChunkSize))
            return {nullptr, nullptr, 0};
        return {mul_chunk(chunk_), slot_chunk(chunk_), idx_};
    }

    XAD_FORCE_INLINE void commit_run(size_type end) { idx_ = end; }

    // This container keeps multipliers and slots in separate arrays, so it cannot
    // hand out a contiguous run of pairs to write into. Returning null here means
    // callers stage into their own buffer and append.
    XAD_FORCE_INLINE std::pair<T, S>* try_reserve(size_type) { return nullptr; }

    XAD_FORCE_INLINE void commit(size_type) {}

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
        append_here(muls, slots, items);

        while (XAD_VERY_UNLIKELY(idx_ == ChunkSize))
        {
            addChunks(1);
            ++chunk_;
            idx_ = 0;
            n -= items;

            items = (std::min)(ChunkSize, n);
            append_here(muls, slots, items);
        }
    }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

    // the same, from the last element to the first
    template <class Func>
    void for_each_reverse(size_type startidx, size_type endidx, Func f) const
    {
        size_type start_chunk = startidx / ChunkSize;
        size_type end_chunk = endidx / ChunkSize;
        size_type e1 = end_chunk != start_chunk ? ChunkSize : endidx % ChunkSize;
        if (XAD_VERY_UNLIKELY(start_chunk != end_chunk && endidx % ChunkSize != 0))
        {
            const auto* m = mul_chunk(end_chunk);
            const auto* s = slot_chunk(end_chunk);
            for (size_type i = endidx % ChunkSize; i > 0; --i) f(m[i - 1], s[i - 1]);
        }
        const auto* m = mul_chunk(start_chunk) + startidx % ChunkSize;
        const auto* s = slot_chunk(start_chunk) + startidx % ChunkSize;
        for (size_type i = e1 - startidx % ChunkSize; i > 0; --i) f(m[i - 1], s[i - 1]);
    }

    // The entries of one chunk, for for_each/for_each_reverse without chunk lookups;
    // empty if [startidx, endidx) is empty or spans chunks
    struct flat_view
    {
        const mul_type* muls;
        const slot_type* slots;
        size_type offset;
        explicit operator bool() const { return muls != nullptr; }
        template <class Func>
        XAD_FORCE_INLINE void for_each_reverse(size_type startidx, size_type endidx, Func f) const
        {
            // one index counting down over both arrays: one step per entry
            const mul_type* m = muls + (startidx - offset);
            const slot_type* s = slots + (startidx - offset);
            for (size_type i = endidx - startidx; i > 0; --i) f(m[i - 1], s[i - 1]);
        }
        template <class Func>
        XAD_FORCE_INLINE void for_each(size_type startidx, size_type endidx, Func f) const
        {
            for (size_type i = startidx - offset, e = endidx - offset; i < e; ++i)
                f(muls[i], slots[i]);
        }
    };

    XAD_FORCE_INLINE flat_view flat(size_type startidx, size_type endidx) const
    {
        size_type c = startidx / ChunkSize;
        if (startidx >= endidx || (endidx - 1) / ChunkSize != c)
            return {nullptr, nullptr, 0};
        return {mul_chunk(c), slot_chunk(c), c * ChunkSize};
    }

  private:
    // the next n entries of the current chunk, from the iterators
    template <class MulIt, class SlotIt>
    XAD_FORCE_INLINE void append_here(MulIt& muls, SlotIt& slots, size_type n)
    {
        auto m = mul_chunk(chunk_) + idx_;
        auto s = slot_chunk(chunk_) + idx_;
        for (size_type i = 0; i < n; ++i)
        {
            ::new (m + i) T(*muls++);
            ::new (s + i) S(*slots++);
        }
        idx_ += n;
    }

    // the slots of a chunk follow its multipliers in the same allocation. Functions, not static
    // members: Tape's extern template declarations instantiate this class while mul_type may still
    // be incomplete, and MSVC evaluates static member initializers then.
    static constexpr std::size_t slot_offset()
    {
        return ((ChunkSize * sizeof(mul_type) + alignof(slot_type) - 1) / alignof(slot_type)) *
               alignof(slot_type);
    }
    static constexpr std::size_t chunk_bytes()
    {
        return slot_offset() + sizeof(slot_type) * ChunkSize;
    }

    XAD_FORCE_INLINE mul_type* mul_chunk(size_type chunk)
    {
        return reinterpret_cast<mul_type*>(chunks_[chunk].get());
    }
    XAD_FORCE_INLINE slot_type* slot_chunk(size_type chunk)
    {
        return reinterpret_cast<slot_type*>(chunks_[chunk].get() + slot_offset());
    }
    XAD_FORCE_INLINE const mul_type* mul_chunk(size_type chunk) const
    {
        return reinterpret_cast<const mul_type*>(chunks_[chunk].get());
    }
    XAD_FORCE_INLINE const slot_type* slot_chunk(size_type chunk) const
    {
        return reinterpret_cast<const slot_type*>(chunks_[chunk].get() + slot_offset());
    }

    void addChunks(size_type newChunks)
    {
        for (size_type i = 0; i < newChunks; ++i)
        {
            auto chunk = AllocHelper::aligned_alloc(ALIGNMENT, chunk_bytes());
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
            if (start >= cont->size())
                return;
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

            if (end_idx == 0)
            {
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

        if (end_idx == 0)
        {
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

    std::vector<std::unique_ptr<char, AllocHelper>> chunks_;
    size_type idx_ = 0;
    size_type chunk_ = 0;
};

}  // namespace xad

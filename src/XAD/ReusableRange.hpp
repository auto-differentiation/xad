/*******************************************************************************

   ReusableRange, used in Tape to keep track of slots that can be re-used

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2023 Xcelerit Computing Ltd.

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

#include <cassert>
#include <iosfwd>

namespace xad
{
template <class T>
class ReusableRange
{
  public:
    explicit ReusableRange(T start = T(), T end = T()) : first_(start), second_(end) {}
    bool isClosed() const { return first_ >= second_; }
    T size() const { return second_ - first_; }

    T first() const { return first_; }
    T second() const { return second_; }
    void first(const T& f) { first_ = f; }
    void second(const T& s) { second_ = s; }
    bool isInRange(T item) const { return item >= first_ && item < second_; }

    T insert()
    {
        assert(!isClosed());
        return first_++;
    }

    enum ExpandResult
    {
        FAILED,
        START,
        END
    };

    ExpandResult expand(T item)
    {
        if (item == first_ - 1)
        {
            --first_;
            return START;
        }
        if (item == second_)
        {
            ++second_;
            return END;
        }
        return FAILED;
    }

    ReusableRange& joinEnd(const ReusableRange& other)
    {
        assert(isJoinableEnd(other));
        second_ = other.second_;
        return *this;
    }

    ReusableRange& joinStart(const ReusableRange& other)
    {
        assert(isJoinableStart(other));
        first_ = other.first_;
        return *this;
    }

    bool isJoinableStart(const ReusableRange& other) const { return other.second_ == first_; }
    bool isJoinableEnd(const ReusableRange& other) const { return other.first_ == second_; }
    ExpandResult isJoinable(const ReusableRange& other) const
    {
        if (isJoinableEnd(other))
            return END;
        if (isJoinableStart(other))
            return START;
        return FAILED;
    }

  private:
    T first_;
    T second_;  // range boundaries
};

// define a comparison based on the start of the range only, for sorting
template <class T>
inline bool operator<(const ReusableRange<T>& a, const ReusableRange<T>& b)
{
    return a.first() < b.first();
}

template <class T>
inline bool operator==(const ReusableRange<T>& a, const ReusableRange<T>& b)
{
    return a.first() == b.first() && a.second() == b.second();
}

template <class C, class Traits, class T>
inline std::basic_ostream<C, Traits>& operator<<(std::basic_ostream<C, Traits>& os,
                                                 const ReusableRange<T>& r)
{
    return os << "[" << r.first() << ", " << r.second() << ")";
}

}  // namespace xad
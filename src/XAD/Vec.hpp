/*******************************************************************************

   Declaration of a Vec type for tracking multiple derivates.

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

*********************************************************************************/

#pragma once
#include <XAD/Macros.hpp>
#include <array>

namespace xad
{

template <class T, std::size_t N>
struct Vec
{
    std::array<T, N> data_;
    typedef T value_type;

    const T& operator[](std::size_t i) const { return data_[i]; }
    T& operator[](std::size_t i) { return data_[i]; }

    constexpr std::size_t size() const { return N; }
    constexpr bool empty() const { return N == 0; }

    typedef typename std::array<T, N>::const_iterator iter;
    iter begin() const { return data_.begin(); }
    iter end() const { return data_.end(); }

    Vec<T, N>& operator=(const T& scalar)
    {
        for (std::size_t i = 0; i < this->size(); i++) data_[i] = scalar;
        return *this;
    }

    bool operator==(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++)
            if (data_[i] != v[i])
                return false;
        return true;
    }

    bool operator!=(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++)
            if (data_[i] != v[i])
                return true;
        return false;
    }

    bool operator==(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++)
            if (data_[i] != scalar)
                return false;
        return true;
    }

    bool operator!=(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++)
            if (data_[i] == scalar)
                return false;
        return true;
    }

    Vec<T, N>& operator+=(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] += scalar;
        return *this;
    }

    Vec<T, N>& operator+=(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] += v[i];
        return *this;
    }

    Vec<T, N>& operator-=(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] -= scalar;
        return *this;
    }

    Vec<T, N>& operator-=(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] -= v[i];
        return *this;
    }

    Vec<T, N>& operator*=(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] *= scalar;
        return *this;
    }

    Vec<T, N>& operator*=(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] *= v[i];
        return *this;
    }

    Vec<T, N>& operator/=(const T& scalar)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] /= scalar;
        return *this;
    }

    Vec<T, N>& operator/=(const Vec<T, N>& v)
    {
        for (std::size_t i = 0; i < N; i++) data_[i] /= v[i];
        return *this;
    }
};

// addition
template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator+(Vec<T, N> v, const T& scalar)
{
    return v += scalar;
}

template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator+(const T& scalar, Vec<T, N> v)
{
    return v += scalar;
}

template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator+(Vec<T, N> v, const Vec<T, N>& y)
{
    return v += y;
}

// substraction
template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator-(Vec<T, N> v, const T& scalar)
{
    return v -= scalar;
}

template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator-(const T& scalar, Vec<T, N> v)
{
    v *= -1;
    return v += scalar;
}

// multiplication
template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator*(Vec<T, N> v, const T& scalar)
{
    return v *= scalar;
}

template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator*(const T& scalar, Vec<T, N> v)
{
    return v *= scalar;
}

// division
template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator/(Vec<T, N> v, const T& scalar)
{
    return v /= scalar;
}

template <typename T, std::size_t N>
XAD_INLINE Vec<T, N> operator/(const T& scalar, Vec<T, N> v)
{
    for (std::size_t i = 0; i < N; i++) v[i] = scalar / v[i];
    return v;
}

}  // namespace xad

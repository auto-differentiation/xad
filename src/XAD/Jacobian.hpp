/*******************************************************************************

   Implementation of Jacobian matrix computing methods.

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

#include <XAD/TypeTraits.hpp>
#include <XAD/XAD.hpp>

#include <functional>
#include <type_traits>
#include <vector>

namespace xad
{
// adj 2d vector
template <typename T>
std::vector<std::vector<T>> computeJacobian(
    const std::vector<xad::AReal<T>> &vec,
    std::function<std::vector<xad::AReal<T>>(std::vector<xad::AReal<T>> &)> foo,
    xad::Tape<T> *tape = xad::Tape<T>::getActive())
{
    auto v(vec);
    std::vector<std::vector<T>> matrix(foo(v).size(), std::vector<T>(v.size(), 0.0));
    computeJacobian(vec, foo, begin(matrix), end(matrix), tape);
    return matrix;
}

// adj iterator
template <class RowIterator, typename T>
void computeJacobian(const std::vector<xad::AReal<T>> &vec,
                     std::function<std::vector<xad::AReal<T>>(std::vector<xad::AReal<T>> &)> foo,
                     RowIterator first, RowIterator last,
                     xad::Tape<T> *tape = xad::Tape<T>::getActive())
{
    auto v(vec);
    unsigned int domain = static_cast<unsigned int>(vec.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());

    if (std::distance(first, last) != codomain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(
        xad::detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
        "RowIterator must dereference to a type that implements a begin() method");
    std::unique_ptr<xad::Tape<T>> t;
    if (!tape)
    {
        t = std::unique_ptr<xad::Tape<T>>(new xad::Tape<T>());
        tape = t.get();
    }

    tape->registerInputs(v);

    auto row = first;
    for (unsigned int i = 0; i < codomain; i++, row++)
    {
        auto col = row->begin();
        for (unsigned int j = 0; j < domain; j++, col++)
        {
            tape->newRecording();
            xad::AReal<T> y = foo(v)[i];
            tape->registerOutput(y);
            derivative(y) = 1.0;
            tape->computeAdjoints();
            *col = derivative(v[j]);
        }
    }
}

// fwd 2d vector
template <typename T>
std::vector<std::vector<T>> computeJacobian(
    const std::vector<xad::FReal<T>> &vec,
    std::function<std::vector<xad::FReal<T>>(std::vector<xad::FReal<T>> &)> foo)
{
    auto v(vec);
    std::vector<std::vector<T>> matrix(foo(v).size(), std::vector<T>(v.size(), 0.0));
    computeJacobian(vec, foo, begin(matrix), end(matrix));
    return matrix;
}

// fwd iterator
template <class RowIterator, typename T>
void computeJacobian(const std::vector<xad::FReal<T>> &vec,
                     std::function<std::vector<xad::FReal<T>>(std::vector<xad::FReal<T>> &)> foo,
                     RowIterator first, RowIterator last)
{
    auto v(vec);
    unsigned int domain = static_cast<unsigned int>(vec.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());

    if (std::distance(first, last) != codomain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(
        xad::detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
        "RowIterator must dereference to a type that implements a begin() method");

    auto row = first;
    for (unsigned int i = 0; i < codomain; i++, row++)
    {
        auto col = row->begin();
        for (unsigned int j = 0; j < domain; j++, col++)
        {
            derivative(v[j]) = 1.0;
            xad::FReal<T> y = foo(v)[i];
            *col = derivative(y);
            derivative(v[j]) = 0.0;
        }
    }
}
}  // namespace xad

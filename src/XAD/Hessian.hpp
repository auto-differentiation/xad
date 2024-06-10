/*******************************************************************************

   Implementation of Hessian matrix computing methods.

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

// fwd_adj 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<xad::AReal<xad::FReal<T>>> &vec,
    std::function<xad::AReal<xad::FReal<T>>(std::vector<xad::AReal<xad::FReal<T>>> &)> foo,
    xad::Tape<xad::FReal<T>> *tape = xad::Tape<xad::FReal<T>>::getActive())
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix), tape);
    return matrix;
}

// fwd_adj iterator
template <class RowIterator, typename T>
void computeHessian(
    const std::vector<xad::AReal<xad::FReal<T>>> &vec,
    std::function<xad::AReal<xad::FReal<T>>(std::vector<xad::AReal<xad::FReal<T>>> &)> foo,
    RowIterator first, RowIterator last,
    xad::Tape<xad::FReal<T>> *tape = xad::Tape<xad::FReal<T>>::getActive())
{
    unsigned int domain(static_cast<unsigned int>(vec.size()));

    if (std::distance(first, last) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(
        xad::detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
        "RowIterator must dereference to a type that implements a begin() method");
    std::unique_ptr<xad::Tape<xad::FReal<T>>> t;
    if (!tape)
    {
        t = std::unique_ptr<xad::Tape<xad::FReal<T>>>(new xad::Tape<xad::FReal<T>>());
        tape = t.get();
    }

    auto v(vec);
    tape->registerInputs(v);

    auto row = first;
    for (unsigned int i = 0; i < domain; i++, row++)
    {
        derivative(value(v[i])) = 1.0;
        tape->newRecording();
        xad::AReal<xad::FReal<T>> y = foo(v);
        tape->registerOutput(y);
        value(derivative(y)) = 1.0;
        tape->computeAdjoints();

        auto col = row->begin();
        for (unsigned int j = 0; j < domain; j++, col++)
        {
            *col = derivative(derivative(v[j]));
        }
        derivative(value(v[i])) = 0.0;
    }
}

// fwd_fwd 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<xad::FReal<xad::FReal<T>>> &vec,
    std::function<xad::FReal<xad::FReal<T>>(std::vector<xad::FReal<xad::FReal<T>>> &)> foo)
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix));
    return matrix;
}

// fwd_fwd iterator
template <class RowIterator, typename T>
void computeHessian(
    const std::vector<xad::FReal<xad::FReal<T>>> &vec,
    std::function<xad::FReal<xad::FReal<T>>(std::vector<xad::FReal<xad::FReal<T>>> &)> foo,
    RowIterator first, RowIterator last)
{
    unsigned int domain(static_cast<unsigned int>(vec.size()));

    if (std::distance(first, last) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(
        xad::detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
        "RowIterator must dereference to a type that implements a begin() method");

    auto v(vec);

    auto row = first;
    for (unsigned int i = 0; i < domain; i++, row++)
    {
        value(derivative(v[i])) = 1.0;
        auto col = row->begin();
        for (unsigned int j = 0; j < domain; j++, col++)
        {
            derivative(value(v[j])) = 1.0;
            xad::FReal<xad::FReal<T>> y = foo(v);
            *col = derivative(derivative(y));
            derivative(value(v[j])) = 0.0;
        }
        value(derivative(v[i])) = 0.0;
    }
}
}  // namespace xad

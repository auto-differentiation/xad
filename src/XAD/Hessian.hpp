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
#include <memory>
#include <type_traits>
#include <vector>

namespace xad
{

// fwd_adj 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<AReal<FReal<T>>> &vec,
    std::function<AReal<FReal<T>>(std::vector<AReal<FReal<T>>> &)> foo,
    Tape<FReal<T>> *tape = Tape<FReal<T>>::getActive())
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix), tape);
    return matrix;
}

// fwd_adj iterator
template <typename RowIterator, typename T>
void computeHessian(const std::vector<AReal<FReal<T>>> &vec,
                    std::function<AReal<FReal<T>>(std::vector<AReal<FReal<T>>> &)> foo,
                    RowIterator first, RowIterator last,
                    Tape<FReal<T>> *tape = Tape<FReal<T>>::getActive())
{
    std::size_t domain(static_cast<std::size_t>(vec.size()));

    if (static_cast<std::size_t>(std::distance(first, last)) != domain ||
        static_cast<std::size_t>(std::distance(first->cbegin(), first->cend())) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
                  "RowIterator must dereference to a type that implements a begin() method");
    std::unique_ptr<Tape<FReal<T>>> t;
    if (!tape)
    {
        t = std::unique_ptr<Tape<FReal<T>>>(new Tape<FReal<T>>());
        tape = t.get();
    }

    auto v(vec);
    tape->registerInputs(v);

    auto row = first;
    for (std::size_t i = 0; i < domain; i++, row++)
    {
        derivative(value(v[i])) = 1.0;
        tape->newRecording();
        AReal<FReal<T>> y = foo(v);
        tape->registerOutput(y);
        value(derivative(y)) = 1.0;
        tape->computeAdjoints();

        auto col = row->begin();
        for (std::size_t j = 0; j < domain; j++, col++)
        {
            *col = derivative(derivative(v[j]));
        }
        derivative(value(v[i])) = 0.0;
    }
}

// fwd_fwd 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<FReal<FReal<T>>> &vec,
    std::function<FReal<FReal<T>>(std::vector<FReal<FReal<T>>> &)> foo)
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix));
    return matrix;
}

// fwd_fwd iterator
template <typename RowIterator, typename T>
void computeHessian(const std::vector<FReal<FReal<T>>> &vec,
                    std::function<FReal<FReal<T>>(std::vector<FReal<FReal<T>>> &)> foo,
                    RowIterator first, RowIterator last)
{
    std::size_t domain(static_cast<std::size_t>(vec.size()));

    if (static_cast<std::size_t>(std::distance(first, last)) != domain ||
        static_cast<std::size_t>(std::distance(first->cbegin(), first->cend())) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
                  "RowIterator must dereference to a type that implements a begin() method");

    auto v(vec);

    auto row = first;
    for (std::size_t i = 0; i < domain; i++, row++)
    {
        value(derivative(v[i])) = 1.0;
        auto col = row->begin();
        for (std::size_t j = 0; j < domain; j++, col++)
        {
            derivative(value(v[j])) = 1.0;
            FReal<FReal<T>> y = foo(v);
            *col = derivative(derivative(y));
            derivative(value(v[j])) = 0.0;
        }
        value(derivative(v[i])) = 0.0;
    }
}
}  // namespace xad

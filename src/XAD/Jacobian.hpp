/*******************************************************************************

   Implementation of jacobian computing methods.

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

#include <functional>
#include <type_traits>
#include <vector>

namespace xad
{
// adj 2d vector
template <typename T>
std::vector<std::vector<xad::AReal<T>>> computeJacobian(
    std::vector<xad::AReal<T>> &v,
    std::function<std::vector<xad::AReal<T>>(std::vector<xad::AReal<T>> &)> foo, xad::Tape<T> *tape)
{
    tape->registerInputs(v);
    unsigned int domain = static_cast<unsigned int>(v.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());

    std::vector<std::vector<xad::AReal<T>>> matrix(
        std::vector<std::vector<xad::AReal<T>>>(codomain, std::vector<xad::AReal<T>>(domain, 0.0)));

    for (unsigned int i = 0; i < domain; i++)
    {
        for (unsigned int j = 0; j < codomain; j++)
        {
            tape->newRecording();
            xad::AReal<T> y = foo(v)[j];
            tape->registerOutput(y);
            derivative(y) = 1.0;
            tape->computeAdjoints();

            // std::cout << "df" << j << "/dx" << i << " = " << derivative(v[i]) << std::endl;
            matrix[i][j] = derivative(v[i]);
        }
    }

    return matrix;
}

// adj iterator
template <class RowIterator, typename T>
void computeJacobian(std::vector<xad::AReal<T>> &v,
                     std::function<std::vector<xad::AReal<T>>(std::vector<xad::AReal<T>> &)> foo,
                     xad::Tape<T> *tape, RowIterator first, RowIterator last)
{
    tape->registerInputs(v);
    unsigned int domain = static_cast<unsigned int>(v.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());

    if (std::distance(first, last) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
                  "RowIterator must dereference to a type that implements a begin() method");

    auto row = first;

    for (unsigned int i = 0; i < domain; i++, row++)
    {
        auto col = row->begin();

        for (unsigned int j = 0; j < codomain; j++, col++)
        {
            tape->newRecording();
            xad::AReal<T> y = foo(v)[j];
            tape->registerOutput(y);
            derivative(y) = 1.0;
            tape->computeAdjoints();

            // std::cout << "df" << j << "/dx" << i << " = " << derivative(v[i]) << std::endl;
            *col = derivative(v[i]);
        }
    }
}

// fwd 2d vector
template <typename T>
std::vector<std::vector<xad::FReal<T>>> computeJacobian(
    std::vector<xad::FReal<T>> &v,
    std::function<std::vector<xad::FReal<T>>(std::vector<xad::FReal<T>> &)> foo)
{
    unsigned int domain = static_cast<unsigned int>(v.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());
    std::vector<std::vector<xad::FReal<T>>> matrix(
        std::vector<std::vector<xad::FReal<T>>>(codomain, std::vector<xad::FReal<T>>(domain, 0.0)));

    for (unsigned int i = 0; i < domain; i++)
    {
        derivative(v[i]) = 1.0;

        for (unsigned int j = 0; j < codomain; j++)
        {
            xad::FReal<T> y = foo(v)[j];
            // std::cout << "df" << j << "/dx" << i << " = " << derivative(y) << std::endl;
            matrix[i][j] = derivative(y);
        }

        derivative(v[i]) = 0.0;
    }

    return matrix;
}

// fwd iterator
template <class RowIterator, typename T>
void computeJacobian(std::vector<xad::FReal<T>> &v,
                     std::function<std::vector<xad::FReal<T>>(std::vector<xad::FReal<T>> &)> foo,
                     RowIterator first, RowIterator last)
{
    unsigned int domain = static_cast<unsigned int>(v.size()),
                 codomain = static_cast<unsigned int>(foo(v).size());

    if (std::distance(first, last) != domain)
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
                  "RowIterator must dereference to a type that implements a begin() method");

    auto row = first;

    for (unsigned int i = 0; i < domain; i++, row++)
    {
        derivative(v[i]) = 1.0;

        auto col = row->begin();

        for (unsigned int j = 0; j < codomain; j++, col++)
        {
            xad::FReal<T> y = foo(v)[j];
            // std::cout << "df" << j << "/dx" << i << " = " << derivative(y) << std::endl;
            *col = derivative(y);
        }

        derivative(v[i]) = 0.0;
    }
}
}  // namespace xad

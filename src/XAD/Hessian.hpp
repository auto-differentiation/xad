/*******************************************************************************

   Implementation of Hessian matrix computing methods.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, version 3 of the License only.

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

namespace detail
{

// forward over adjoint implementation, used for AReal<FReal<T>> and ARealDirect<FRealDirect<T>>
template <typename RowIterator, typename Active, typename Inner>
void computeHessianFwdAdj(const std::vector<Active> &vec,
                          std::function<Active(std::vector<Active> &)> foo, RowIterator first,
                          RowIterator last, Tape<Inner> *tape)
{
    std::size_t domain = vec.size();

    if (static_cast<std::size_t>(std::distance(first, last)) != domain ||
        (domain != 0 &&
         static_cast<std::size_t>(std::distance(first->cbegin(), first->cend())) != domain))
        throw OutOfRange("Iterator not allocated enough space");
    static_assert(detail::has_begin<typename std::iterator_traits<RowIterator>::value_type>::value,
                  "RowIterator must dereference to a type that implements a begin() method");
    std::unique_ptr<Tape<Inner>> t;
    if (!tape)
    {
        t = std::unique_ptr<Tape<Inner>>(new Tape<Inner>());
        tape = t.get();
    }

    auto v(vec);
    tape->registerInputs(v);

    auto row = first;
    for (std::size_t i = 0; i < domain; i++, row++)
    {
        derivative(value(v[i])) = 1.0;
        tape->newRecording();
        Active y = foo(v);
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

}  // namespace detail

// fwd_adj iterator
template <typename RowIterator, typename T>
void computeHessian(const std::vector<AReal<FReal<T>>> &vec,
                    std::function<AReal<FReal<T>>(std::vector<AReal<FReal<T>>> &)> foo,
                    RowIterator first, RowIterator last,
                    Tape<FReal<T>> *tape = Tape<FReal<T>>::getActive())
{
    detail::computeHessianFwdAdj(vec, foo, first, last, tape);
}

// fwdd_adjd 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<ARealDirect<FRealDirect<T>>> &vec,
    std::function<ARealDirect<FRealDirect<T>>(std::vector<ARealDirect<FRealDirect<T>>> &)> foo,
    Tape<FRealDirect<T>> *tape = Tape<FRealDirect<T>>::getActive())
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix), tape);
    return matrix;
}

// fwdd_adjd iterator
template <typename RowIterator, typename T>
void computeHessian(
    const std::vector<ARealDirect<FRealDirect<T>>> &vec,
    std::function<ARealDirect<FRealDirect<T>>(std::vector<ARealDirect<FRealDirect<T>>> &)> foo,
    RowIterator first, RowIterator last,
    Tape<FRealDirect<T>> *tape = Tape<FRealDirect<T>>::getActive())
{
    detail::computeHessianFwdAdj(vec, foo, first, last, tape);
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

namespace detail
{

// forward over forward implementation, used for FReal<FReal<T>> and FRealDirect<FRealDirect<T>>
template <typename RowIterator, typename Active>
void computeHessianFwdFwd(const std::vector<Active> &vec,
                          std::function<Active(std::vector<Active> &)> foo, RowIterator first,
                          RowIterator last)
{
    std::size_t domain = vec.size();

    if (static_cast<std::size_t>(std::distance(first, last)) != domain ||
        (domain != 0 &&
         static_cast<std::size_t>(std::distance(first->cbegin(), first->cend())) != domain))
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
            Active y = foo(v);
            *col = derivative(derivative(y));
            derivative(value(v[j])) = 0.0;
        }
        value(derivative(v[i])) = 0.0;
    }
}

}  // namespace detail

// fwd_fwd iterator
template <typename RowIterator, typename T>
void computeHessian(const std::vector<FReal<FReal<T>>> &vec,
                    std::function<FReal<FReal<T>>(std::vector<FReal<FReal<T>>> &)> foo,
                    RowIterator first, RowIterator last)
{
    detail::computeHessianFwdFwd(vec, foo, first, last);
}

// fwdd_fwdd 2d vector
template <typename T>
std::vector<std::vector<T>> computeHessian(
    const std::vector<FRealDirect<FRealDirect<T>>> &vec,
    std::function<FRealDirect<FRealDirect<T>>(std::vector<FRealDirect<FRealDirect<T>>> &)> foo)
{
    std::vector<std::vector<T>> matrix(vec.size(), std::vector<T>(vec.size(), 0.0));
    computeHessian(vec, foo, begin(matrix), end(matrix));
    return matrix;
}

// fwdd_fwdd iterator
template <typename RowIterator, typename T>
void computeHessian(
    const std::vector<FRealDirect<FRealDirect<T>>> &vec,
    std::function<FRealDirect<FRealDirect<T>>(std::vector<FRealDirect<FRealDirect<T>>> &)> foo,
    RowIterator first, RowIterator last)
{
    detail::computeHessianFwdFwd(vec, foo, first, last);
}
}  // namespace xad

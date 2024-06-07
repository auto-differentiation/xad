/*******************************************************************************

   Implementation of Hessian computing methods.

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

template <class T>
class Hessian
{
  public:
    // fwd_adj 2d vector constructor
    Hessian(std::function<T(std::vector<T> &)> func, const std::vector<T> &v,
            xad::Tape<xad::FReal<double>> *tape)
        : foo_(func),
          v_(v),
          domain_(static_cast<unsigned int>(v_.size())),
          matrix_(domain_, std::vector<T>(domain_, 0.0))
    {
        compute(tape);
    }

    // fwd_adj iterator constructor
    template <class RowIterator>
    Hessian(std::function<T(std::vector<T> &)> func, const std::vector<T> &v,
            xad::Tape<xad::FReal<double>> *tape, RowIterator first, RowIterator last)
        : foo_(func), v_(v), domain_(static_cast<unsigned int>(v_.size())), matrix_(0)
    {
        if (std::distance(first, last) != domain_)
            throw OutOfRange("Iterator not allocated enough space");
        static_assert(has_begin<decltype(*first)>::value,
                      "RowIterator must dereference to a type that implements a begin() method");

        compute(tape, first, last);
    }

    // tapeless (fwd_fwd) 2d vector constructor
    Hessian(std::function<T(std::vector<T> &)> func, std::vector<T> &v)
        : foo_(func),
          v_(v),
          domain_(static_cast<unsigned int>(v_.size())),
          matrix_(domain_, std::vector<T>(domain_, 0.0))
    {
        compute();
    }

    // tapeless (fwd_fwd) iterator constructor
    template <class RowIterator>
    Hessian(std::function<T(std::vector<T> &)> func, std::vector<T> &v, RowIterator first,
            RowIterator last)
        : foo_(func), v_(v), domain_(static_cast<unsigned int>(v_.size())), matrix_(0)
    {
        if (std::distance(first, last) != domain_)
            throw OutOfRange("Iterator not allocated enough space");
        static_assert(has_begin<decltype(*first)>::value,
                      "RowIterator must dereference to a type that implements a begin() method");

        compute(first, last);
    }

    // fwd_adj 2d vector
    void compute(xad::Tape<xad::FReal<double>> *tape)
    {
        tape->registerInputs(v_);

        for (unsigned int i = 0; i < domain_; i++)
        {
            derivative(value(v_[i])) = 1.0;
            tape->newRecording();

            T y = foo_(v_);
            tape->registerOutput(y);
            value(derivative(y)) = 1.0;

            tape->computeAdjoints();

            for (unsigned int j = 0; j < domain_; j++)
            {
                // std::cout << "d2y/dx" << i << "dx" << j << " = " << derivative(derivative(v[j]))
                //           << "\n";
                matrix_[i][j] = derivative(derivative(v_[j]));
            }

            derivative(value(v_[i])) = 0.0;
        }
    }

    // fwd_adj iterator
    template <class RowIterator>
    void compute(xad::Tape<xad::FReal<double>> *tape, RowIterator first, RowIterator last)
    {
        tape->registerInputs(v_);

        auto row = first;

        for (unsigned int i = 0; i < domain_; i++, row++)
        {
            derivative(value(v_[i])) = 1.0;
            tape->newRecording();

            T y = foo_(v_);
            tape->registerOutput(y);
            value(derivative(y)) = 1.0;

            tape->computeAdjoints();

            auto col = row->begin();

            for (unsigned int j = 0; j < domain_; j++, col++)
            {
                // std::cout << "d2y/dx" << i << "dx" << j << " = " << derivative(derivative(v[j]))
                //           << "\n";
                *col = derivative(derivative(v_[j]));
            }

            derivative(value(v_[i])) = 0.0;
        }
    }

    // fwd_fwd 2d vector
    void compute()
    {
        for (unsigned int i = 0; i < domain_; i++)
        {
            value(derivative(v_[i])) = 1.0;

            for (unsigned int j = 0; j < domain_; j++)
            {
                derivative(value(v_[j])) = 1.0;

                T y = foo_(v_);

                // std::cout << "d2y/dx" << i << "dx" << j << " = " << derivative(derivative(y))
                //           << "\n";ß
                matrix_[i][j] = derivative(derivative(y));

                derivative(value(v_[j])) = 0.0;
            }

            value(derivative(v_[i])) = 0.0;
        }
    }

    // fwd_fwd iterator
    template <class RowIterator>
    void compute(RowIterator first, RowIterator last)
    {
        auto row = first;

        for (unsigned int i = 0; i < domain_; i++, row++)
        {
            value(derivative(v_[i])) = 1.0;

            auto col = row->begin();

            for (unsigned int j = 0; j < domain_; j++, col++)
            {
                derivative(value(v_[j])) = 1.0;

                T y = foo_(v_);

                // std::cout << "d2y/dx" << i << "dx" << j << " = " << derivative(derivative(y))
                //           << "\n";
                *col = derivative(derivative(y));

                derivative(value(v_[j])) = 0.0;
            }

            value(derivative(v_[i])) = 0.0;
        }
    }

    std::vector<std::vector<T>> get() { return matrix_; }

  private:
    std::function<T(std::vector<T> &)> foo_;
    std::vector<T> v_;
    unsigned int domain_;
    std::vector<std::vector<T>> matrix_;

    template <typename U>
    static auto has_begin_impl(int) -> decltype(std::declval<U>().begin(), std::true_type{});

    template <typename U>
    static std::false_type has_begin_impl(...);

    template <typename U>
    struct has_begin : decltype(has_begin_impl<U>(0))
    {
    };
};
}  // namespace xad

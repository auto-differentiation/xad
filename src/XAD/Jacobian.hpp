/*******************************************************************************

   Implementation of Jacobian computing methods.

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
#include <vector>

namespace xad
{

template <class T>
class Jacobian
{
  public:
    // adj 2d vector constructor
    Jacobian(std::function<std::vector<T>(std::vector<T> &)> foo, const std::vector<T> &v,
             xad::Tape<double> *tape)
        : foo_(foo), v_(v), domain_(static_cast<unsigned int>(v_.size())), codomain_(0), matrix_(0)
    {
        compute(tape);
    }

    // adj iterator constructor
    template <class RowIterator>
    Jacobian(std::function<std::vector<T>(std::vector<T> &)> foo, const std::vector<T> &v,
             xad::Tape<double> *tape, RowIterator first, RowIterator last)
        : foo_(foo), v_(v), domain_(static_cast<unsigned int>(v_.size())), codomain_(0), matrix_(0)
    {
        if (std::distance(first, last) != domain_)
        {
            // throw exception
        }
        compute(tape, first, last);
    }

    // tapeless fwd 2d vector constructor
    Jacobian(std::function<std::vector<T>(std::vector<T> &)> foo, const std::vector<T> &v)
        : foo_(foo), v_(v), domain_(static_cast<unsigned int>(v_.size())), codomain_(0), matrix_(0)
    {
        compute();
    }

    // tapeless fwd iterator constructor
    template <class RowIterator>
    Jacobian(std::function<std::vector<T>(std::vector<T> &)> foo, const std::vector<T> &v,
             RowIterator first, RowIterator last)
        : foo_(foo), v_(v), domain_(static_cast<unsigned int>(v_.size())), codomain_(0), matrix_(0)
    {
        if (std::distance(first, last) != domain_)
        {
            // throw exception
        }
        compute(first, last);
    }

    // adj 2d vector
    void compute(xad::Tape<double> *tape)
    {
        tape->registerInputs(v_);
        tape->newRecording();
        std::vector<T> res = foo_(v_);
        codomain_ = static_cast<unsigned int>(res.size());
        matrix_ = std::vector<std::vector<T>>(codomain_, std::vector<T>(domain_, 0.0));

        for (unsigned int i = 0; i < codomain_; i++)
        {
            T y = res[i];
            tape->registerOutput(y);
            derivative(y) = 1.0;

            tape->computeAdjoints();

            for (unsigned int j = 0; j < domain_; j++)
            {
                std::cout << "df" << j << "/dx" << i << " = " << derivative(v_[j]) << std::endl;
                matrix_[i][j] = derivative(v_[j]);
            }
            derivative(y) = 0.0;
        }
    }

    // adj iterator
    template <class RowIterator>
    void compute(xad::Tape<double> *tape, RowIterator first, RowIterator last)
    {
        tape->registerInputs(v_);
        tape->newRecording();
        std::vector<T> res = foo_(v_);
        codomain_ = static_cast<unsigned int>(res.size());
        auto row = first;

        for (unsigned int i = 0; i < codomain_; i++, row++)
        {
            tape->registerOutput(res[i]);
            derivative(res[i]) = 1.0;

            tape->computeAdjoints();

            auto col = row->begin();

            for (unsigned int j = 0; j < domain_; j++, col++)
            {
                std::cout << "df" << j << "/dx" << i << " = " << derivative(v_[j]) << std::endl;
                *col = derivative(v_[j]);
            }

            derivative(res[i]) = 0.0;
        }
    }

    // fwd 2d vector
    void compute()
    {
        // TODO: optimise to use less computations of foo_()
        codomain_ = static_cast<unsigned int>(foo_(v_).size());
        matrix_ = std::vector<std::vector<T>>(codomain_, std::vector<T>(domain_, 0.0));

        for (unsigned int i = 0; i < domain_; i++)
        {
            derivative(v_[i]) = 1.0;

            for (unsigned int j = 0; j < codomain_; j++)
            {
                T y = foo_(v_)[j];

                std::cout << "df" << j << "/dx" << i << " = " << derivative(y) << std::endl;

                matrix_[i][j] = derivative(y);
            }

            derivative(v_[i]) = 0.0;
        }
    }

    // fwd iterator
    template <class RowIterator>
    void compute(RowIterator first, RowIterator last)
    {
        codomain_ = static_cast<unsigned int>(foo_(v_).size());

        auto row = first;

        for (unsigned int i = 0; i < domain_; i++, row++)
        {
            derivative(v_[i]) = 1.0;

            auto col = row->begin();

            for (unsigned int j = 0; j < codomain_; j++, col++)
            {
                T y = foo_(v_)[j];

                std::cout << "df" << j << "/dx" << i << " = " << derivative(y) << std::endl;

                *col = derivative(y);
            }

            derivative(v_[i]) = 0.0;
        }
    }

    std::vector<std::vector<T>> get() { return matrix_; }

  private:
    std::function<std::vector<T>(std::vector<T> &)> foo_;
    std::vector<T> v_;
    unsigned int domain_, codomain_;
    std::vector<std::vector<T>> matrix_;
};
}  // namespace xad

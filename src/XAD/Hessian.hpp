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
#include <vector>

namespace xad
{

template <class T>
class Hessian
{
  public:
    // fwd_adj constructor
    Hessian(std::function<T(std::vector<T> &)> func, const std::vector<T> &v,
            xad::Tape<xad::FReal<double>> *tape)
        : foo_(func),
          v_(v),
          domain_(static_cast<unsigned int>(v_.size())),
          matrix_(domain_, std::vector<T>(domain_, 0.0))
    {
        compute(tape);
    }

    // tapeless (fwd_fwd) constructor
    Hessian(std::function<T(std::vector<T> &)> func, std::vector<T> &v)
        : foo_(func),
          v_(v),
          domain_(static_cast<unsigned int>(v_.size())),
          matrix_(domain_, std::vector<T>(domain_, 0.0))
    {
        compute();
    }

    // fwd_adj
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

    // fwd_fwd
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
                //           << "\n";
                matrix_[i][j] = derivative(derivative(y));

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
};
}  // namespace xad

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
    Jacobian(std::function<std::vector<T>(std::vector<T> &)> foo, const std::vector<T> &v,
             xad::Tape<double> *tape)
        : foo_(foo),
          v_(v),
          tape_(tape),
          domain_(static_cast<unsigned int>(v_.size())),
          codomain_(0),
          matrix_(0)
    {
        compute();
    }

    void compute()
    {
        tape_->registerInputs(v_);
        std::vector<T> res = foo_(v_);
        codomain_ = static_cast<unsigned int>(res.size());
        matrix_ = std::vector<std::vector<T>>(codomain_, std::vector<T>(domain_, 0.0));

        for (unsigned int i = 0; i < domain_; i++)
        {
            for (unsigned int j = 0; j < codomain_; j++)
            {
                derivative(v_[i]) = 1.0;
                tape_->newRecording();

                T y = res[j];
                tape_->registerOutput(y);
                derivative(y) = 1.0;

                tape_->computeAdjoints();

                std::cout << "df" << j << "/dx" << i << " = " << derivative(v_[i]) << std::endl;
                matrix_[i][j] = derivative(v_[i]);
                derivative(v_[i]) = 0.0;
            }
        }

        std::cout << "reached" << std::endl;
    }

    std::vector<std::vector<T>> get() { return matrix_; }

  private:
    std::function<std::vector<T>(std::vector<T> &)> foo_;
    std::vector<T> v_;
    Tape<double> *tape_;
    unsigned int domain_, codomain_;
    std::vector<std::vector<T>> matrix_;
};
}  // namespace xad

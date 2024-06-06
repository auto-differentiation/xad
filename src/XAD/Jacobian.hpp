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
    Jacobian(std::vector<std::function<T(std::vector<T> &)>> foos, std::vector<T> &v,
             xad::Tape<double> *tape)
        : foos(foos),
          v(v),
          tape(tape),
          domain(static_cast<unsigned int>(v.size())),
          codomain(static_cast<unsigned int>(foos.size()))
    {
    }

    std::vector<std::vector<T>> compute()
    {
        std::vector<std::vector<T>> matrix(codomain, std::vector<T>(domain, 0.0));

        tape->registerInputs(v);

        for (unsigned int i = 0; i < domain; i++)
        {
            for (unsigned int j = 0; j < codomain; j++)
            {
                derivative(v[i]) = 1.0;
                tape->newRecording();

                T y = foos[j](v);
                tape->registerOutput(y);
                derivative(y) = 1.0;

                tape->computeAdjoints();

                // std::cout << "df" << j << "/dx" << i << " = " << derivative(v[i]) << std::endl;
                matrix[i][j] = derivative(v[i]);
                derivative(v[i]) = 0.0;
            }
        }

        return matrix;
    }

  private:
    std::vector<std::function<T(std::vector<T> &)>> foos;
    std::vector<T> v;
    Tape<double> *tape;
    unsigned int domain, codomain;
};
}  // namespace xad

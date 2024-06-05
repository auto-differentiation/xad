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

#include <vector>

namespace xad
{

template <class T>
class Hessian
{
  public:
    Hessian() {}

    std::vector<std::vector<T>> compute(std::vector<T> &v)
    {
        xad::Tape<xad::FReal<double>> tape;

        domain = static_cast<unsigned int>(v.size());
        std::vector<std::vector<T>> matrix(domain, std::vector<T>(domain, 0.0));

        tape.registerInputs(v);

        for (unsigned int i = 0; i < domain; i++)
        {
            tape.registerInputs(v);
            derivative(value(v[i])) = 1.0;
            tape.newRecording();

            T y = q(v[0], v[1]);
            tape.registerOutput(y);
            value(derivative(y)) = 1.0;

            tape.computeAdjoints();

            for (unsigned int j = 0; j < domain; j++)
            {
                std::cout << "d2y/dx" << i << "dx" << j << " = " << derivative(derivative(v[j]))
                          << "\n";
                matrix[i][j] = derivative(derivative(v[j]));
            }

            derivative(value(v[i])) = 0.0;
        }

        return matrix;
    }

  private:
    T q(T a, T b)
    {
        T c = a * a;
        T d = b * b;
        return c + d;
    }

    unsigned int domain;
};
}  // namespace xad

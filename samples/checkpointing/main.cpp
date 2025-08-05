/*******************************************************************************

   Computes
     xout = sin(sin(sin(sin(... sin(xin)))))
     (sin applied n times)
   and its first order derivative
     dxout/dxin
   using checkpointing.

   A checkpoint is created after equi-distant applications of the sin function
   to save memory.


   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

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

#include "sin_checkpoint.hpp"

#include <XAD/XAD.hpp>
#include <iostream>

int main()
{
    int n = 20;       // number of iterations
    double x = 2.1;   // input value
    double xa = 1.0;  // initial adjoint of output

    // run checkpointed version
    repeated_sin_checkpointed(n, x, xa);

    // output results
    std::cout << "xout       = " << x << "\n"
              << "dxout/dxin = " << xa << "\n";
}

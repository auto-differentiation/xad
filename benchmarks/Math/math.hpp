/*******************************************************************************

   Helpers for Math benchmarks.

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

#include <XAD/XAD.hpp>
#include <XAD/StdCompatibility.hpp>


template <typename T>
std::vector<T (*)(T)> &make_unary_functions()
{
    return std::vector<T (*)(T)>({
            *abs,
            *acos,
            *acosh,
            *asin,
            *asinh,
            *atan,
            *atan2,
            *atanh,
            *cbrt,
            *ceil,
            *cos,
            *cosh,
            *erf,
            *erfc,
            *exp,
            *exp2,
            *expm1,
            *fabs,
            *floor,
            *fpclassify,
            *ilogb,
            *isfinite,
            *isinf,
            *isnan,
            *isnormal,
            *llround,
            *log,
            *log10,
            *log1p,
            *log2,
            *lround,
            *round,
            *signbit,
            *sin,
            *sinh,
            *sqrt,
            *tan,
            *tanh,
            *trunc,
    }),
}

template <typename T>
std::vector<void (*)> &make_binary_functions()
{
    return std::vector<void (*)>({
            *copysign, 
            *fmax, 
            *fmin,  
            *fmod, 
            *frexp, 
            *hypot, 
            *ldexp, 
            *max, 
            *min, 
            *modf, 
            *nextafter,
            *pow, 
            *remainder,
            *scalbn,
    }),
}

template <typename T>
std::vector<void (*)> &make_ternary_functions()
{
    return std::vector<void (*)>({
        *remquo,  
    }),
}

template <typename T>
std::vector<void (*)> &make_functions()
{
    return std::vector<void (*)>({
        *abs,
        *acos,
        *acosh,
        *asin,
        *asinh,
        *atan,
        *atan2,
        *atanh,
        *cbrt,
        *ceil,
        *copysign, 
        *cos,
        *cosh,
        *erf,
        *erfc,
        *exp,
        *exp2,
        *expm1,
        *fabs,
        *floor,
        *fmax, 
        *fmin,  
        *fmod, 
        *fpclassify,
        *frexp, 
        *hypot, 
        *ilogb,
        *isfinite,
        *isinf,
        *isnan,
        *isnormal,
        *ldexp, 
        *llround,
        *log,
        *log10,
        *log1p,
        *log2,
        *lround,
        *max, 
        *min, 
        *modf, 
        *nextafter, 
        *pow, 
        *remainder, 
        *remquo,  
        *round,
        *scalbn, 
        *signbit,
        *sin,
        *sinh,
        *sqrt,
        *tan,
        *tanh,
        *trunc,
    }),
}
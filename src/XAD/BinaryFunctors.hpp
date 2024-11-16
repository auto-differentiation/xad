/*******************************************************************************

   Functors for binary arithmetic operators.

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

#pragma once
#include <XAD/Macros.hpp>
#include <immintrin.h>

namespace xad
{
template <class Scalar>
struct add_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(1); }
};

template <>
struct add_op<double>
{
    XAD_INLINE double operator()(const double a, const double b) const
    {
        return _mm_cvtsd_f64(_mm_add_sd(_mm_set_sd(a), _mm_set_sd(b)));
    }

    XAD_INLINE double derivative_a(const double, const double) const { return 1; }
    XAD_INLINE double derivative_b(const double, const double) const { return 1; }
};

template <>
struct add_op<float>
{
    XAD_INLINE float operator()(const float a, const float b) const
    {
        return _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(a), _mm_set_ss(b)));
    }
    XAD_INLINE float derivative_a(const float, const float) const { return 1; }
    XAD_INLINE float derivative_b(const float, const float) const { return 1; }
};

template <class Scalar>
struct prod_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a * b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar&) const { return a; }
};

template <>
struct prod_op<double>
{
    XAD_INLINE double operator()(const double a, const double b) const
    {
        return _mm_cvtsd_f64(_mm_mul_sd(_mm_set_sd(a), _mm_set_sd(b)));
    }

    XAD_INLINE double derivative_a(const double, const double b) const { return b; }
    XAD_INLINE double derivative_b(const double a, const double) const { return a; }
};

template <>
struct prod_op<float>
{
    XAD_INLINE float operator()(const float a, const float b) const
    {
        return _mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(a), _mm_set_ss(b)));
    }
    XAD_INLINE float derivative_a(const float, const float b) const { return b; }
    XAD_INLINE float derivative_b(const float a, const float) const { return a; }
};

template <class Scalar>
struct sub_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a - b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-1); }
};

template <>
struct sub_op<double>
{
    XAD_INLINE double operator()(const double a, const double b) const
    {
        return _mm_cvtsd_f64(_mm_sub_sd(_mm_set_sd(a), _mm_set_sd(b)));
    }

    XAD_INLINE double derivative_a(const double, const double) const { return 1; }
    XAD_INLINE double derivative_b(const double, const double) const { return -1; }
};

template <>
struct sub_op<float>
{
    XAD_INLINE float operator()(const float a, const float b) const
    {
        return _mm_cvtss_f32(_mm_sub_ss(_mm_set_ss(a), _mm_set_ss(b)));
    }
    XAD_INLINE float derivative_a(const float, const float) const { return 1; }
    XAD_INLINE float derivative_b(const float, const float) const { return -1; }
};

template <class Scalar>
struct div_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a / b; }

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return Scalar(1) / b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -a / (b * b); }
};

template <>
struct div_op<double>
{
    XAD_INLINE double operator()(const double a, const double b) const
    {
        return _mm_cvtsd_f64(_mm_div_sd(_mm_set_sd(a), _mm_set_sd(b)));
    }

    XAD_INLINE double derivative_a(const double, const double b) const { return 1 / b; }
    XAD_INLINE double derivative_b(const double a, const double b) const { return -a / (b * b); }
};

template <>
struct div_op<float>
{
    XAD_INLINE float operator()(const float a, const float b) const
    {
        return _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(a), _mm_set_ss(b)));
    }
    XAD_INLINE float derivative_a(const float, const float b) const { return 1 / b; }
    XAD_INLINE float derivative_b(const float a, const float b) const { return -a / (b * b); }
};
}  // namespace xad

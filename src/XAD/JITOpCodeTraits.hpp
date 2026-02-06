/**
 *
 *   Mapping from XAD operator/functor types to JITOpCode values.
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Affero General Public License for more details.
 *
 *   You should have received a copy of the GNU Affero General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_JIT

#include <XAD/JITGraph.hpp>

namespace xad
{

// Forward declarations of operator types
template <class> struct add_op;
template <class> struct sub_op;
template <class> struct prod_op;
template <class> struct div_op;
template <class> struct negate_op;
template <class> struct pow_op;
template <class> struct max_op;
template <class> struct min_op;
template <class> struct fmax_op;
template <class> struct fmin_op;
template <class> struct fmod_op;
template <class> struct atan2_op;
template <class> struct remainder_op;
template <class> struct remquo_op;
template <class> struct hypot_op;
template <class> struct nextafter_op;
template <class> struct smooth_abs_op;

// Derived scalar ops (degrees/radians inherit from scalar_prod_op)
template <class> struct degrees_op;
template <class> struct radians_op;

// Unary math functors
template <class> struct sin_op;
template <class> struct cos_op;
template <class> struct tan_op;
template <class> struct asin_op;
template <class> struct acos_op;
template <class> struct atan_op;
template <class> struct sinh_op;
template <class> struct cosh_op;
template <class> struct tanh_op;
template <class> struct exp_op;
template <class> struct log_op;
template <class> struct log10_op;
template <class> struct log2_op;
template <class> struct sqrt_op;
template <class> struct cbrt_op;
template <class> struct abs_op;
template <class> struct fabs_op;
template <class> struct floor_op;
template <class> struct ceil_op;
template <class> struct erf_op;
template <class> struct erfc_op;
template <class> struct expm1_op;
template <class> struct log1p_op;
template <class> struct asinh_op;
template <class> struct acosh_op;
template <class> struct atanh_op;
template <class> struct exp2_op;
template <class> struct trunc_op;
template <class> struct round_op;
template <class> struct ldexp_op;
template <class> struct frexp_op;
template <class, class> struct modf_op;

// Scalar ops
template <class, class> struct scalar_add_op;
template <class, class> struct scalar_sub1_op;
template <class, class> struct scalar_sub2_op;
template <class, class> struct scalar_prod_op;
template <class, class> struct scalar_div1_op;
template <class, class> struct scalar_div2_op;
template <class, class> struct scalar_pow1_op;
template <class, class> struct scalar_pow2_op;
template <class, class> struct scalar_max_op;
template <class, class> struct scalar_min_op;
template <class, class> struct scalar_fmax_op;
template <class, class> struct scalar_fmin_op;
template <class, class> struct scalar_fmod1_op;
template <class, class> struct scalar_fmod2_op;
template <class, class> struct scalar_atan21_op;
template <class, class> struct scalar_atan22_op;
template <class, class> struct scalar_remainder1_op;
template <class, class> struct scalar_remainder2_op;
template <class, class> struct scalar_remquo1_op;
template <class, class> struct scalar_remquo2_op;
template <class, class> struct scalar_hypot1_op;
template <class, class> struct scalar_hypot2_op;
template <class, class> struct scalar_nextafter1_op;
template <class, class> struct scalar_nextafter2_op;
template <class, class> struct scalar_smooth_abs1_op;
template <class, class> struct scalar_smooth_abs2_op;

// Primary template - unknown operator
template <class Op>
struct JITOpCodeFor
{
    // Use an invalid sentinel value for unmapped operators.
    static constexpr JITOpCode value = static_cast<JITOpCode>(0xFFFF);
};

// Binary arithmetic
template <class S> struct JITOpCodeFor<add_op<S>> { static constexpr JITOpCode value = JITOpCode::Add; };
template <class S> struct JITOpCodeFor<sub_op<S>> { static constexpr JITOpCode value = JITOpCode::Sub; };
template <class S> struct JITOpCodeFor<prod_op<S>> { static constexpr JITOpCode value = JITOpCode::Mul; };
template <class S> struct JITOpCodeFor<div_op<S>> { static constexpr JITOpCode value = JITOpCode::Div; };

// Unary
template <class S> struct JITOpCodeFor<negate_op<S>> { static constexpr JITOpCode value = JITOpCode::Neg; };

// Binary math
template <class S> struct JITOpCodeFor<pow_op<S>> { static constexpr JITOpCode value = JITOpCode::Pow; };
template <class S> struct JITOpCodeFor<max_op<S>> { static constexpr JITOpCode value = JITOpCode::Max; };
template <class S> struct JITOpCodeFor<min_op<S>> { static constexpr JITOpCode value = JITOpCode::Min; };
template <class S> struct JITOpCodeFor<fmax_op<S>> { static constexpr JITOpCode value = JITOpCode::Max; };
template <class S> struct JITOpCodeFor<fmin_op<S>> { static constexpr JITOpCode value = JITOpCode::Min; };
template <class S> struct JITOpCodeFor<fmod_op<S>> { static constexpr JITOpCode value = JITOpCode::Mod; };
template <class S> struct JITOpCodeFor<atan2_op<S>> { static constexpr JITOpCode value = JITOpCode::Atan2; };
template <class S> struct JITOpCodeFor<remainder_op<S>> { static constexpr JITOpCode value = JITOpCode::Remainder; };
template <class S> struct JITOpCodeFor<remquo_op<S>> { static constexpr JITOpCode value = JITOpCode::Remquo; };
template <class S> struct JITOpCodeFor<hypot_op<S>> { static constexpr JITOpCode value = JITOpCode::Hypot; };
template <class S> struct JITOpCodeFor<nextafter_op<S>> { static constexpr JITOpCode value = JITOpCode::Nextafter; };
template <class S> struct JITOpCodeFor<smooth_abs_op<S>> { static constexpr JITOpCode value = JITOpCode::SmoothAbs; };

// Unary math
template <class S> struct JITOpCodeFor<sin_op<S>> { static constexpr JITOpCode value = JITOpCode::Sin; };
template <class S> struct JITOpCodeFor<cos_op<S>> { static constexpr JITOpCode value = JITOpCode::Cos; };
template <class S> struct JITOpCodeFor<tan_op<S>> { static constexpr JITOpCode value = JITOpCode::Tan; };
template <class S> struct JITOpCodeFor<asin_op<S>> { static constexpr JITOpCode value = JITOpCode::Asin; };
template <class S> struct JITOpCodeFor<acos_op<S>> { static constexpr JITOpCode value = JITOpCode::Acos; };
template <class S> struct JITOpCodeFor<atan_op<S>> { static constexpr JITOpCode value = JITOpCode::Atan; };
template <class S> struct JITOpCodeFor<sinh_op<S>> { static constexpr JITOpCode value = JITOpCode::Sinh; };
template <class S> struct JITOpCodeFor<cosh_op<S>> { static constexpr JITOpCode value = JITOpCode::Cosh; };
template <class S> struct JITOpCodeFor<tanh_op<S>> { static constexpr JITOpCode value = JITOpCode::Tanh; };
template <class S> struct JITOpCodeFor<exp_op<S>> { static constexpr JITOpCode value = JITOpCode::Exp; };
template <class S> struct JITOpCodeFor<log_op<S>> { static constexpr JITOpCode value = JITOpCode::Log; };
template <class S> struct JITOpCodeFor<log10_op<S>> { static constexpr JITOpCode value = JITOpCode::Log10; };
template <class S> struct JITOpCodeFor<log2_op<S>> { static constexpr JITOpCode value = JITOpCode::Log2; };
template <class S> struct JITOpCodeFor<sqrt_op<S>> { static constexpr JITOpCode value = JITOpCode::Sqrt; };
template <class S> struct JITOpCodeFor<cbrt_op<S>> { static constexpr JITOpCode value = JITOpCode::Cbrt; };
template <class S> struct JITOpCodeFor<abs_op<S>> { static constexpr JITOpCode value = JITOpCode::Abs; };
template <class S> struct JITOpCodeFor<fabs_op<S>> { static constexpr JITOpCode value = JITOpCode::Abs; };
template <class S> struct JITOpCodeFor<floor_op<S>> { static constexpr JITOpCode value = JITOpCode::Floor; };
template <class S> struct JITOpCodeFor<ceil_op<S>> { static constexpr JITOpCode value = JITOpCode::Ceil; };
template <class S> struct JITOpCodeFor<erf_op<S>> { static constexpr JITOpCode value = JITOpCode::Erf; };
template <class S> struct JITOpCodeFor<erfc_op<S>> { static constexpr JITOpCode value = JITOpCode::Erfc; };
template <class S> struct JITOpCodeFor<expm1_op<S>> { static constexpr JITOpCode value = JITOpCode::Expm1; };
template <class S> struct JITOpCodeFor<log1p_op<S>> { static constexpr JITOpCode value = JITOpCode::Log1p; };
template <class S> struct JITOpCodeFor<asinh_op<S>> { static constexpr JITOpCode value = JITOpCode::Asinh; };
template <class S> struct JITOpCodeFor<acosh_op<S>> { static constexpr JITOpCode value = JITOpCode::Acosh; };
template <class S> struct JITOpCodeFor<atanh_op<S>> { static constexpr JITOpCode value = JITOpCode::Atanh; };
template <class S> struct JITOpCodeFor<exp2_op<S>> { static constexpr JITOpCode value = JITOpCode::Exp2; };
template <class S> struct JITOpCodeFor<trunc_op<S>> { static constexpr JITOpCode value = JITOpCode::Trunc; };
template <class S> struct JITOpCodeFor<round_op<S>> { static constexpr JITOpCode value = JITOpCode::Round; };
template <class S> struct JITOpCodeFor<ldexp_op<S>> { static constexpr JITOpCode value = JITOpCode::Ldexp; };
template <class S> struct JITOpCodeFor<frexp_op<S>> { static constexpr JITOpCode value = JITOpCode::Frexp; };
template <class S, class T> struct JITOpCodeFor<modf_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Modf; };

// Scalar operations
template <class S, class T> struct JITOpCodeFor<scalar_add_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Add; };
template <class S, class T> struct JITOpCodeFor<scalar_sub1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Sub; };
template <class S, class T> struct JITOpCodeFor<scalar_sub2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Sub; };
template <class S, class T> struct JITOpCodeFor<scalar_prod_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Mul; };
template <class S, class T> struct JITOpCodeFor<scalar_div1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Div; };
template <class S, class T> struct JITOpCodeFor<scalar_div2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Div; };
template <class S, class T> struct JITOpCodeFor<scalar_pow1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Pow; };
template <class S, class T> struct JITOpCodeFor<scalar_pow2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Pow; };
template <class S, class T> struct JITOpCodeFor<scalar_max_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Max; };
template <class S, class T> struct JITOpCodeFor<scalar_min_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Min; };
template <class S, class T> struct JITOpCodeFor<scalar_fmax_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Max; };
template <class S, class T> struct JITOpCodeFor<scalar_fmin_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Min; };
template <class S, class T> struct JITOpCodeFor<scalar_fmod1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Mod; };
template <class S, class T> struct JITOpCodeFor<scalar_fmod2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Mod; };
template <class S, class T> struct JITOpCodeFor<scalar_atan21_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Atan2; };
template <class S, class T> struct JITOpCodeFor<scalar_atan22_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Atan2; };
template <class S, class T> struct JITOpCodeFor<scalar_remainder1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Remainder; };
template <class S, class T> struct JITOpCodeFor<scalar_remainder2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Remainder; };
template <class S, class T> struct JITOpCodeFor<scalar_remquo1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Remquo; };
template <class S, class T> struct JITOpCodeFor<scalar_remquo2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Remquo; };
template <class S, class T> struct JITOpCodeFor<scalar_hypot1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Hypot; };
template <class S, class T> struct JITOpCodeFor<scalar_hypot2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Hypot; };
template <class S, class T> struct JITOpCodeFor<scalar_nextafter1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Nextafter; };
template <class S, class T> struct JITOpCodeFor<scalar_nextafter2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::Nextafter; };
template <class S, class T> struct JITOpCodeFor<scalar_smooth_abs1_op<S, T>> { static constexpr JITOpCode value = JITOpCode::SmoothAbs; };
template <class S, class T> struct JITOpCodeFor<scalar_smooth_abs2_op<S, T>> { static constexpr JITOpCode value = JITOpCode::SmoothAbs; };

// Derived types that inherit from scalar_prod_op (need explicit specializations since C++ doesn't match base class)
template <class S> struct JITOpCodeFor<degrees_op<S>> { static constexpr JITOpCode value = JITOpCode::Mul; };
template <class S> struct JITOpCodeFor<radians_op<S>> { static constexpr JITOpCode value = JITOpCode::Mul; };

}  // namespace xad

#endif  // XAD_ENABLE_JIT

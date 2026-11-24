/**
 *
 *   Mapping from XAD operator/functor types to GraphOpCode values.
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

#ifdef XAD_ENABLE_GRAPH

#include <XAD/Graph.hpp>

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
template <class> struct fdim_op;
template <class> struct atan2_op;
template <class> struct remainder_op;
template <class> struct remquo_op;
template <class> struct hypot_op;
template <class> struct nextafter_op;
template <class> struct smooth_abs_op;
template <class> struct copysign_op;

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
template <class> struct tgamma_op;
template <class> struct lgamma_op;
template <class> struct digamma_op;
template <class> struct rint_op;
template <class> struct nearbyint_op;
template <class> struct logb_op;
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
template <class, class> struct scalar_max1_op;
template <class, class> struct scalar_min1_op;
template <class, class> struct scalar_fmod1_op;
template <class, class> struct scalar_fmod2_op;
template <class, class> struct scalar_fdim1_op;
template <class, class> struct scalar_fdim2_op;
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
template <class, class> struct scalar_copysign2_op;

// Primary template - unknown operator
template <class Op>
struct GraphOpCodeFor
{
    // Use an invalid sentinel value for unmapped operators.
    static constexpr GraphOpCode value = static_cast<GraphOpCode>(0xFFFF);
};

// Binary arithmetic
template <class S> struct GraphOpCodeFor<add_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Add; };
template <class S> struct GraphOpCodeFor<sub_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Sub; };
template <class S> struct GraphOpCodeFor<prod_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Mul; };
template <class S> struct GraphOpCodeFor<div_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Div; };

// Unary
template <class S> struct GraphOpCodeFor<negate_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Neg; };

// Binary math
template <class S> struct GraphOpCodeFor<pow_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Pow; };
template <class S> struct GraphOpCodeFor<max_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Max; };
template <class S> struct GraphOpCodeFor<min_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Min; };
template <class S> struct GraphOpCodeFor<fmax_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Max; };
template <class S> struct GraphOpCodeFor<fmin_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Min; };
template <class S> struct GraphOpCodeFor<fmod_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Mod; };
template <class S> struct GraphOpCodeFor<fdim_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Fdim; };
template <class S> struct GraphOpCodeFor<atan2_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Atan2; };
template <class S> struct GraphOpCodeFor<remainder_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Remainder; };
template <class S> struct GraphOpCodeFor<remquo_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Remquo; };
template <class S> struct GraphOpCodeFor<hypot_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Hypot; };
template <class S> struct GraphOpCodeFor<nextafter_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Nextafter; };
template <class S> struct GraphOpCodeFor<smooth_abs_op<S>> { static constexpr GraphOpCode value = GraphOpCode::SmoothAbs; };
template <class S> struct GraphOpCodeFor<copysign_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Copysign; };

// Unary math
template <class S> struct GraphOpCodeFor<sin_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Sin; };
template <class S> struct GraphOpCodeFor<cos_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Cos; };
template <class S> struct GraphOpCodeFor<tan_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Tan; };
template <class S> struct GraphOpCodeFor<asin_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Asin; };
template <class S> struct GraphOpCodeFor<acos_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Acos; };
template <class S> struct GraphOpCodeFor<atan_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Atan; };
template <class S> struct GraphOpCodeFor<sinh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Sinh; };
template <class S> struct GraphOpCodeFor<cosh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Cosh; };
template <class S> struct GraphOpCodeFor<tanh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Tanh; };
template <class S> struct GraphOpCodeFor<exp_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Exp; };
template <class S> struct GraphOpCodeFor<log_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Log; };
template <class S> struct GraphOpCodeFor<log10_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Log10; };
template <class S> struct GraphOpCodeFor<log2_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Log2; };
template <class S> struct GraphOpCodeFor<sqrt_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Sqrt; };
template <class S> struct GraphOpCodeFor<cbrt_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Cbrt; };
template <class S> struct GraphOpCodeFor<abs_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Abs; };
template <class S> struct GraphOpCodeFor<fabs_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Abs; };
template <class S> struct GraphOpCodeFor<floor_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Floor; };
template <class S> struct GraphOpCodeFor<ceil_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Ceil; };
template <class S> struct GraphOpCodeFor<erf_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Erf; };
template <class S> struct GraphOpCodeFor<erfc_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Erfc; };
template <class S> struct GraphOpCodeFor<expm1_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Expm1; };
template <class S> struct GraphOpCodeFor<log1p_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Log1p; };
template <class S> struct GraphOpCodeFor<asinh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Asinh; };
template <class S> struct GraphOpCodeFor<acosh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Acosh; };
template <class S> struct GraphOpCodeFor<atanh_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Atanh; };
template <class S> struct GraphOpCodeFor<exp2_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Exp2; };
template <class S> struct GraphOpCodeFor<trunc_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Trunc; };
template <class S> struct GraphOpCodeFor<round_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Round; };
template <class S> struct GraphOpCodeFor<tgamma_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Tgamma; };
template <class S> struct GraphOpCodeFor<lgamma_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Lgamma; };
template <class S> struct GraphOpCodeFor<digamma_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Digamma; };
template <class S> struct GraphOpCodeFor<rint_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Rint; };
template <class S> struct GraphOpCodeFor<nearbyint_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Nearbyint; };
template <class S> struct GraphOpCodeFor<logb_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Logb; };
template <class S> struct GraphOpCodeFor<ldexp_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Ldexp; };
template <class S> struct GraphOpCodeFor<frexp_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Frexp; };
template <class S, class T> struct GraphOpCodeFor<modf_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Modf; };

// Scalar operations
template <class S, class T> struct GraphOpCodeFor<scalar_add_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Add; };
template <class S, class T> struct GraphOpCodeFor<scalar_sub1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Sub; };
template <class S, class T> struct GraphOpCodeFor<scalar_sub2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Sub; };
template <class S, class T> struct GraphOpCodeFor<scalar_prod_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Mul; };
template <class S, class T> struct GraphOpCodeFor<scalar_div1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Div; };
template <class S, class T> struct GraphOpCodeFor<scalar_div2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Div; };
template <class S, class T> struct GraphOpCodeFor<scalar_pow1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Pow; };
template <class S, class T> struct GraphOpCodeFor<scalar_pow2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Pow; };
template <class S, class T> struct GraphOpCodeFor<scalar_max_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Max; };
template <class S, class T> struct GraphOpCodeFor<scalar_min_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Min; };
template <class S, class T> struct GraphOpCodeFor<scalar_fmax_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Max; };
template <class S, class T> struct GraphOpCodeFor<scalar_fmin_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Min; };
template <class S, class T> struct GraphOpCodeFor<scalar_max1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Max; };
template <class S, class T> struct GraphOpCodeFor<scalar_min1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Min; };
template <class S, class T> struct GraphOpCodeFor<scalar_fmod1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Mod; };
template <class S, class T> struct GraphOpCodeFor<scalar_fmod2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Mod; };
template <class S, class T> struct GraphOpCodeFor<scalar_fdim1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Fdim; };
template <class S, class T> struct GraphOpCodeFor<scalar_fdim2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Fdim; };
template <class S, class T> struct GraphOpCodeFor<scalar_atan21_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Atan2; };
template <class S, class T> struct GraphOpCodeFor<scalar_atan22_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Atan2; };
template <class S, class T> struct GraphOpCodeFor<scalar_remainder1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Remainder; };
template <class S, class T> struct GraphOpCodeFor<scalar_remainder2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Remainder; };
template <class S, class T> struct GraphOpCodeFor<scalar_remquo1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Remquo; };
template <class S, class T> struct GraphOpCodeFor<scalar_remquo2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Remquo; };
template <class S, class T> struct GraphOpCodeFor<scalar_hypot1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Hypot; };
template <class S, class T> struct GraphOpCodeFor<scalar_hypot2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Hypot; };
template <class S, class T> struct GraphOpCodeFor<scalar_nextafter1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Nextafter; };
template <class S, class T> struct GraphOpCodeFor<scalar_nextafter2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Nextafter; };
template <class S, class T> struct GraphOpCodeFor<scalar_smooth_abs1_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::SmoothAbs; };
template <class S, class T> struct GraphOpCodeFor<scalar_smooth_abs2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::SmoothAbs; };
template <class S, class T> struct GraphOpCodeFor<scalar_copysign2_op<S, T>> { static constexpr GraphOpCode value = GraphOpCode::Copysign; };

// Derived types that inherit from scalar_prod_op (need explicit specializations since C++ doesn't match base class)
template <class S> struct GraphOpCodeFor<degrees_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Mul; };
template <class S> struct GraphOpCodeFor<radians_op<S>> { static constexpr GraphOpCode value = GraphOpCode::Mul; };

}  // namespace xad

#endif  // XAD_ENABLE_GRAPH

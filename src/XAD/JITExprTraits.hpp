/**
 *
 *   Helpers and traits for recording XAD expressions into a JITGraph.
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
#include <XAD/JITOpCodeTraits.hpp>
#include <type_traits>

namespace xad
{

// Helper to extract nested double value from potentially nested AD types
// Non-template overloads for primitive types (preferred over template)
// These are needed to terminate recursion for different scalar types
inline double getNestedDoubleValue(double x) { return x; }
inline double getNestedDoubleValue(float x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(long double x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(int x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(long x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(long long x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(unsigned int x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(unsigned long x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(unsigned long long x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(short x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(unsigned short x) { return static_cast<double>(x); }

// Template for AD types - recurses via value()
template <class T>
double getNestedDoubleValue(const T& x) { return getNestedDoubleValue(x.value()); }

// Helper to detect if Op has a scalar constant (b_ member from UnaryFunctors.hpp)
template <class Op, class = void>
struct HasScalarConstantB_ : std::false_type {};

template <class Op>
struct HasScalarConstantB_<Op, decltype(void(std::declval<Op>().b_))> : std::true_type {};

// Helper to detect if Op has a scalar constant (b member from UnaryMathFunctors.hpp)
template <class Op, class = void>
struct HasScalarConstantB : std::false_type {};

template <class Op>
struct HasScalarConstantB<Op, decltype(void(std::declval<Op>().b))> : std::true_type {};

// Combined trait - has either b_ or b member
template <class Op>
struct HasScalarConstant : std::integral_constant<bool,
    HasScalarConstantB_<Op>::value || HasScalarConstantB<Op>::value> {};

// Helper to detect ldexp_op (has exp_ member of type int, not int*)
template <class Op, class = void>
struct IsLdexpOp : std::false_type {};

template <class Op>
struct IsLdexpOp<Op, typename std::enable_if<
    std::is_same<decltype(std::declval<Op>().exp_), int>::value
>::type> : std::true_type {};

// Helper to get constant value from scalar ops with b_ member
template <class Op>
typename std::enable_if<HasScalarConstantB_<Op>::value, double>::type
getScalarConstant(const Op& op) { return getNestedDoubleValue(op.b_); }

// Helper to get constant value from scalar ops with b member (math functors)
template <class Op>
typename std::enable_if<!HasScalarConstantB_<Op>::value && HasScalarConstantB<Op>::value, double>::type
getScalarConstant(const Op& op) { return getNestedDoubleValue(op.b); }

// Helper to get exponent value from ldexp_op
template <class Op>
typename std::enable_if<IsLdexpOp<Op>::value, double>::type
getLdexpExponent(const Op& op) { return static_cast<double>(op.exp_); }

// Detect if Op is scalar_sub1 or scalar_div1 (scalar is first operand)
template <class> struct IsScalarFirstOp : std::false_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_sub1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_div1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_pow1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_fmod1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_atan21_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_remainder1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_remquo1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_hypot1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_nextafter1_op<S, T>> : std::true_type {};
template <class S, class T> struct IsScalarFirstOp<scalar_smooth_abs1_op<S, T>> : std::true_type {};

// Helper to record scalar value as constant
inline uint32_t recordJITConstant(JITGraph& graph, double value)
{
    // addConstant already creates the Constant node and returns its ID
    return graph.addConstant(value);
}

}  // namespace xad

#endif  // XAD_ENABLE_JIT

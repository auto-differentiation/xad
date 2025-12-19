#pragma once

#include <XAD/JITGraph.hpp>
#include <XAD/JITOpCodeTraits.hpp>
#include <type_traits>

namespace xad
{

// Helper to extract nested double value from potentially nested AD types
// Non-template overloads for primitive types (preferred over template)
inline double getNestedDoubleValue(double x) { return x; }
inline double getNestedDoubleValue(float x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(long double x) { return static_cast<double>(x); }
inline double getNestedDoubleValue(int x) { return static_cast<double>(x); }

// Template for AD types - recurses via value()
template <class T>
double getNestedDoubleValue(const T& x) { return getNestedDoubleValue(x.value()); }

// Helper to detect if Op has a scalar constant (b_ member)
template <class Op, class = void>
struct HasScalarConstant : std::false_type {};

template <class Op>
struct HasScalarConstant<Op, decltype(void(std::declval<Op>().b_))> : std::true_type {};

// Helper to get constant value from scalar ops (handles nested AD types)
template <class Op>
typename std::enable_if<HasScalarConstant<Op>::value, double>::type
getScalarConstant(const Op& op) { return getNestedDoubleValue(op.b_); }

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

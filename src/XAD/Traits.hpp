/*******************************************************************************

   Declaration of traits classes.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

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

#include <XAD/Vec.hpp>
#include <type_traits>

namespace xad
{

enum Direction
{
    DIR_NONE,
    DIR_FORWARD,
    DIR_REVERSE
};

template <class T>
struct ExprTraits
{
    static const bool isExpr = false;
    static const int numVariables = 0;
    static const bool isForward = false;
    static const bool isReverse = false;
    static const bool isLiteral = false;
    static const Direction direction = Direction::DIR_NONE;
    static const std::size_t vector_size = 1;

    typedef T nested_type;
    typedef T value_type;
    typedef T scalar_type;
};

template <class T>
struct ExprTraits<const T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<volatile T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<const volatile T> : ExprTraits<T>
{
};

template <class Op>
struct OperatorTraits
{
    enum
    {
        useResultBasedDerivatives = 0
    };
};

template <class T>
struct float_or_double : public std::false_type
{
};
template <>
struct float_or_double<float> : public std::true_type
{
};
template <>
struct float_or_double<double> : public std::true_type
{
};

template <class, std::size_t>
struct ARealDirect;
template <class, std::size_t>
struct FRealDirect;

namespace detail
{

template <class T>
struct is_direct : public std::false_type
{
};

template <class T, std::size_t N>
struct is_direct<ARealDirect<T, N>> : public std::true_type
{
};

template <class T, std::size_t N>
struct is_direct<FRealDirect<T, N>> : public std::true_type
{
};

// built-in arithmetic types (is_fundamental: StdCompatibility.hpp makes active types arithmetic)
template <class T>
struct is_builtin_arithmetic
    : public std::integral_constant<bool,
                                    std::is_arithmetic<T>::value && std::is_fundamental<T>::value>
{
};

// operands forwarded to the underlying AReal / FReal: built-in arithmetic types, expressions
template <class T>
struct direct_operand
    : public std::integral_constant<bool, is_builtin_arithmetic<T>::value || ExprTraits<T>::isExpr>
{
};

// other operands implicitly convertible to D (e.g. QuantLib's Null<Real>); not enums
template <class D, class T>
struct direct_convertible_operand
    : public std::integral_constant<bool, !direct_operand<T>::value && !is_direct<T>::value &&
                                              !std::is_enum<T>::value &&
                                              std::is_convertible<const T&, D>::value>
{
};

// float, double and active types based on them (higher-order direct mode)
template <class T>
struct direct_scalar : public float_or_double<typename ExprTraits<T>::nested_type>
{
};

// active types: expressions (AReal, FReal, ...) and direct mode types
template <class T>
struct is_active : public std::integral_constant<bool, ExprTraits<T>::isExpr || is_direct<T>::value>
{
};

}  // namespace detail

template <class T, std::size_t N>
struct DerivativesTraits
{
    using type = Vec<T, N>;
};

template <class T>
struct DerivativesTraits<T, 1>
{
    using type = T;
};

}  // namespace xad

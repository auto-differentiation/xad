/*******************************************************************************

   Declaration of traits classes.

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
}  // namespace xad

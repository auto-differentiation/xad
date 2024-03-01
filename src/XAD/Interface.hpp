/*******************************************************************************

   Declaration of the convenience typedef interface.

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

template <class>
struct AReal;
template <class>
struct FReal;
template <class>
class Tape;

template <class T>
struct adj
{
    typedef Tape<T> tape_type;
    typedef typename tape_type::active_type active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T>
struct fwd
{
    typedef void tape_type;
    typedef FReal<T> active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T>
struct fwd_adj
{
    typedef FReal<T> inner_type;
    typedef AReal<inner_type> active_type;
    typedef Tape<inner_type> tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T>
struct fwd_fwd
{
    typedef FReal<T> inner_type;
    typedef FReal<inner_type> active_type;
    typedef void tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T>
struct adj_fwd
{
    typedef AReal<T> inner_type;
    typedef FReal<inner_type> active_type;
    typedef typename inner_type::tape_type tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T>
struct adj_adj
{
    typedef AReal<T> inner_type;
    typedef AReal<inner_type> active_type;
    typedef typename inner_type::tape_type inner_tape_type;
    typedef typename active_type::tape_type outer_tape_type;
    typedef T passive_type;
    typedef T value_type;
};
}  // namespace xad

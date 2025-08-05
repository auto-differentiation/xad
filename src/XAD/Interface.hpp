/*******************************************************************************

   Declaration of the convenience typedef interface.

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

#pragma once

namespace xad
{

template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct FReal;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct ARealDirect;
template <class, std::size_t>
class Tape;

template <class T, std::size_t N = 1>
struct adj
{
    typedef Tape<T, N> tape_type;
    typedef typename tape_type::active_type active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T, std::size_t N = 1>
struct adjd
{
    typedef Tape<T, N> tape_type;
    typedef typename tape_type::active_type active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T, std::size_t N = 1>
struct fwd
{
    typedef void tape_type;
    typedef FReal<T, N> active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T, std::size_t N = 1>
struct fwdd
{
    typedef void tape_type;
    typedef FRealDirect<T, N> active_type;
    typedef T passive_type;
    typedef passive_type value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct fwd_adj
{
    typedef FReal<T, N> inner_type;
    typedef AReal<inner_type, M> active_type;
    typedef Tape<inner_type, M> tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N, std::size_t M = 1>
struct fwdd_adj
{
    typedef FRealDirect<T, N> inner_type;
    typedef AReal<inner_type, M> active_type;
    typedef Tape<inner_type, M> tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct fwd_adjd
{
    typedef FReal<T, N> inner_type;
    typedef ARealDirect<inner_type, M> active_type;
    typedef Tape<inner_type, M> tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct fwdd_adjd
{
    typedef FRealDirect<T, N> inner_type;
    typedef ARealDirect<inner_type, M> active_type;
    typedef Tape<inner_type, M> tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct adj_adj
{
    typedef AReal<T, N> inner_type;
    typedef AReal<inner_type, N> active_type;
    typedef typename inner_type::tape_type inner_tape_type;
    typedef typename active_type::tape_type outer_tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct adjd_adj
{
    typedef ARealDirect<T, N> inner_type;
    typedef AReal<inner_type, N> active_type;
    typedef typename inner_type::tape_type inner_tape_type;
    typedef typename active_type::tape_type outer_tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct adj_adjd
{
    typedef AReal<T, N> inner_type;
    typedef ARealDirect<inner_type, N> active_type;
    typedef typename inner_type::tape_type inner_tape_type;
    typedef typename active_type::tape_type outer_tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct adjd_adjd
{
    typedef ARealDirect<T, N> inner_type;
    typedef ARealDirect<inner_type, N> active_type;
    typedef typename inner_type::tape_type inner_tape_type;
    typedef typename active_type::tape_type outer_tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct fwd_fwd
{
    typedef FReal<T, N> inner_type;
    typedef FReal<inner_type, N> active_type;
    typedef void tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct fwdd_fwd
{
    typedef FRealDirect<T, N> inner_type;
    typedef FReal<inner_type, N> active_type;
    typedef void tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct fwd_fwdd
{
    typedef FReal<T, N> inner_type;
    typedef FRealDirect<inner_type, N> active_type;
    typedef void tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1>
struct fwdd_fwdd
{
    typedef FRealDirect<T, N> inner_type;
    typedef FRealDirect<inner_type, N> active_type;
    typedef void tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct adj_fwd
{
    typedef AReal<T, M> inner_type;
    typedef FReal<inner_type, N> active_type;
    typedef typename inner_type::tape_type tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct adjd_fwd
{
    typedef ARealDirect<T, M> inner_type;
    typedef FReal<inner_type, N> active_type;
    typedef typename inner_type::tape_type tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct adj_fwdd
{
    typedef AReal<T, M> inner_type;
    typedef FRealDirect<inner_type, N> active_type;
    typedef typename inner_type::tape_type tape_type;
    typedef T passive_type;
    typedef T value_type;
};

template <class T, std::size_t N = 1, std::size_t M = 1>
struct adjd_fwdd
{
    typedef ARealDirect<T, M> inner_type;
    typedef FRealDirect<inner_type, N> active_type;
    typedef typename inner_type::tape_type tape_type;
    typedef T passive_type;
    typedef T value_type;
};

}  // namespace xad

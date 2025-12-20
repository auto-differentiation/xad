/**
 *
 *   JIT compiler TLS storage (explicit instantiations).
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2025 Xcelerit Computing Ltd.
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

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_JIT

// Mirror XAD's Tape TLS instantiation pattern (see src/Tape.cpp):
// - provide TLS storage in exactly one translation unit
// - emit TLS definitions for all configured XAD modes (to satisfy references from headers)
#include <XAD/ARealDirect.hpp>
#include <XAD/BinaryOperators.hpp>
#include <XAD/FRealDirect.hpp>
#include <XAD/Literals.hpp>
#include <XAD/Tape.hpp>
#include <XAD/UnaryOperators.hpp>

#include <XAD/JITCompiler.hpp>

namespace xad
{

// Define TLS storage for active_jit_ for a given specialization without forcing full class
// template instantiation (we only need the static data member definition for linking).
#define MAKE_JIT_TLS(type)                                                                         \
    template <>                                                                                    \
    XAD_THREAD_LOCAL JITCompiler<type>* JITCompiler<type>::active_jit_ = nullptr;

// JIT is intentionally limited to scalar, first-order mode only:
// - no vector mode (N>1)
// - no higher-order AD types
//
// Therefore we only provide TLS storage for the scalar specializations.
MAKE_JIT_TLS(float)
MAKE_JIT_TLS(double)

#undef MAKE_JIT_TLS

}  // namespace xad

#endif  // XAD_ENABLE_JIT



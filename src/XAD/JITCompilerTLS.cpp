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

#include <XAD/JITCompiler.hpp>

namespace xad
{

// Explicit definitions for the thread-local "active JIT" pointer.
// These must live in exactly one translation unit to avoid ODR / multiple-definition issues.
template <>
XAD_THREAD_LOCAL JITCompiler<double, 1>* JITCompiler<double, 1>::active_jit_ = nullptr;
template <>
XAD_THREAD_LOCAL JITCompiler<float, 1>* JITCompiler<float, 1>::active_jit_ = nullptr;

// Note: higher-order adjoints (N>1) are not currently supported end-to-end by the JIT backend
// interface (which is scalar/double-based). We still provide TLS symbols for common N values
// to avoid link errors when compiling XAD modes that include N>1 types without using JIT.
template <>
XAD_THREAD_LOCAL JITCompiler<double, 2>* JITCompiler<double, 2>::active_jit_ = nullptr;
template <>
XAD_THREAD_LOCAL JITCompiler<float, 2>* JITCompiler<float, 2>::active_jit_ = nullptr;
template <>
XAD_THREAD_LOCAL JITCompiler<double, 4>* JITCompiler<double, 4>::active_jit_ = nullptr;
template <>
XAD_THREAD_LOCAL JITCompiler<float, 4>* JITCompiler<float, 4>::active_jit_ = nullptr;

}  // namespace xad

#endif  // XAD_ENABLE_JIT



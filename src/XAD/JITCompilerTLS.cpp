/**
 *
 *   JIT compiler TLS storage (explicit instantiations).
 *
 *   This file is part of XAD, a comprehensive C++ library for
 *   automatic differentiation.
 *
 *   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Affero General Public License as published
 *   by the Free Software Foundation, version 3 of the License only.
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

#ifdef XAD_SHARED_LIBS
namespace
{
// not a static member of JITCompiler: thread-local data cannot be exported from a DLL
template <class Real, std::size_t N>
struct ActiveJIT
{
    static XAD_THREAD_LOCAL JITCompiler<Real, N>* jit;
};

template <class Real, std::size_t N>
XAD_THREAD_LOCAL JITCompiler<Real, N>* ActiveJIT<Real, N>::jit = nullptr;
}  // namespace

template <class Real, std::size_t N>
JITCompiler<Real, N>* JITCompiler<Real, N>::getActive()
{
    return ActiveJIT<Real, N>::jit;
}

template <class Real, std::size_t N>
void JITCompiler<Real, N>::setActive(JITCompiler* j)
{
    if (ActiveJIT<Real, N>::jit != nullptr)
        throw OutOfRange("JIT Compiler already active");
    ActiveJIT<Real, N>::jit = j;
}

template <class Real, std::size_t N>
void JITCompiler<Real, N>::setActiveUnchecked(JITCompiler* j)
{
    ActiveJIT<Real, N>::jit = j;
}

template <class Real, std::size_t N>
void JITCompiler<Real, N>::deactivateAll()
{
    ActiveJIT<Real, N>::jit = nullptr;
}
#else
// Define TLS storage for active_jit_ (mirrors Tape.cpp pattern).
template <class Real, std::size_t N>
XAD_THREAD_LOCAL JITCompiler<Real, N>* JITCompiler<Real, N>::active_jit_ = nullptr;
#endif

// JIT is intentionally limited to scalar, first-order mode only:
// - no vector mode (N>1)
// - no higher-order AD types
//
// Therefore we only provide explicit instantiations for the scalar specializations.
#ifdef _MSC_VER
#pragma warning(disable : 4251)
#endif
template class XAD_TEMPLATE_EXPORT JITCompiler<float>;
template class XAD_TEMPLATE_EXPORT JITCompiler<double>;

}  // namespace xad

#endif  // XAD_ENABLE_JIT



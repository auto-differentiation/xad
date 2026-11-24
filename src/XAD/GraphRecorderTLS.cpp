/**
 *
 *   Graph recorder TLS storage (explicit instantiations).
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

#include <XAD/Config.hpp>

#ifdef XAD_ENABLE_GRAPH

// Mirror XAD's Tape TLS instantiation pattern (see src/Tape.cpp):
// - provide TLS storage in exactly one translation unit
// - emit TLS definitions for all configured XAD modes (to satisfy references from headers)
#include <XAD/ARealDirect.hpp>
#include <XAD/BinaryOperators.hpp>
#include <XAD/FRealDirect.hpp>
#include <XAD/Literals.hpp>
#include <XAD/Tape.hpp>
#include <XAD/UnaryOperators.hpp>

#include <XAD/GraphRecorder.hpp>

namespace xad
{

#ifdef XAD_SHARED_LIBS
namespace
{
// not a static member of GraphRecorder: thread-local data cannot be exported from a DLL
template <class Real, std::size_t N>
struct ActiveGraphRecorder
{
    static XAD_THREAD_LOCAL GraphRecorder<Real, N>* recorder;
};

template <class Real, std::size_t N>
XAD_THREAD_LOCAL GraphRecorder<Real, N>* ActiveGraphRecorder<Real, N>::recorder = nullptr;
}  // namespace

template <class Real, std::size_t N>
GraphRecorder<Real, N>* GraphRecorder<Real, N>::getActive()
{
    return ActiveGraphRecorder<Real, N>::recorder;
}

template <class Real, std::size_t N>
void GraphRecorder<Real, N>::setActive(GraphRecorder* j)
{
    if (ActiveGraphRecorder<Real, N>::recorder != nullptr)
        throw OutOfRange("GraphRecorder already active");
    ActiveGraphRecorder<Real, N>::recorder = j;
}

template <class Real, std::size_t N>
void GraphRecorder<Real, N>::setActiveUnchecked(GraphRecorder* j)
{
    ActiveGraphRecorder<Real, N>::recorder = j;
}

template <class Real, std::size_t N>
void GraphRecorder<Real, N>::deactivateAll()
{
    ActiveGraphRecorder<Real, N>::recorder = nullptr;
}
#else
// Define TLS storage for active_recorder_ (mirrors Tape.cpp pattern).
template <class Real, std::size_t N>
XAD_THREAD_LOCAL GraphRecorder<Real, N>* GraphRecorder<Real, N>::active_recorder_ = nullptr;
#endif

// Graph recording is intentionally limited to scalar, first-order mode only:
// - no vector mode (N>1)
// - no higher-order AD types
//
// Therefore we only provide explicit instantiations for the scalar specializations.
#ifdef _MSC_VER
#pragma warning(disable : 4251)
#endif
template class XAD_TEMPLATE_EXPORT GraphRecorder<float>;
template class XAD_TEMPLATE_EXPORT GraphRecorder<double>;

}  // namespace xad

#endif  // XAD_ENABLE_GRAPH



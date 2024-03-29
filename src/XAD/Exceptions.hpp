/*******************************************************************************

   Declaration of exceptions.

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

#include <stdexcept>
#include <string>

namespace xad
{

class Exception : public std::runtime_error
{
  public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

class TapeAlreadyActive : public Exception
{
  public:
    TapeAlreadyActive() : Exception("A tape is already active for the current thread") {}
};

class OutOfRange : public Exception
{
  public:
    explicit OutOfRange(const std::string& msg) : Exception(msg) {}
};

class DerivativesNotInitialized : public Exception
{
  public:
    explicit DerivativesNotInitialized(
        const std::string& msg = "At least one derivative must be set before computing adjoints")
        : Exception(msg)
    {
    }
};

class NoTapeException : public Exception
{
  public:
    explicit NoTapeException(const std::string& msg = "No active tape for the current thread")
        : Exception(msg)
    {
    }
};
}  // namespace xad

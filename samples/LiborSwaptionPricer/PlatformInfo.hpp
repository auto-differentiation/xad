/*******************************************************************************

   Platform Information Utilities

   This header provides cross-platform functions for detecting and reporting
   system information such as CPU, memory, OS, compiler, and SIMD capabilities.

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

#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

namespace platform_info
{

/// Get CPU brand string (e.g., "Intel Core i7-9700K")
inline std::string getCpuInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    char brand[49] = {0};
    unsigned int regs[4];

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 0x80000000);
#else
    __get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[0] >= 0x80000004)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
#ifdef _WIN32
            __cpuid(reinterpret_cast<int*>(regs), 0x80000002 + i);
#else
            __get_cpuid(0x80000002 + i, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            std::memcpy(brand + i * 16, regs, 16);
        }
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos)
            result = result.substr(start);
        return result;
    }
#endif
    return "Unknown CPU";
}

/// Get OS/platform information (e.g., "Windows 10.0 (Build 19041)" or "Linux 5.4.0")
inline std::string getPlatformInfo()
{
#ifdef _WIN32
    typedef LONG(WINAPI * RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    HMODULE hMod = GetModuleHandleW(L"ntdll.dll");
    if (hMod)
    {
        auto RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(hMod, "RtlGetVersion");
        if (RtlGetVersion)
        {
            RTL_OSVERSIONINFOW rovi = {0};
            rovi.dwOSVersionInfoSize = sizeof(rovi);
            if (RtlGetVersion(&rovi) == 0)
            {
                std::ostringstream oss;
                oss << "Windows " << rovi.dwMajorVersion << "." << rovi.dwMinorVersion << " (Build "
                    << rovi.dwBuildNumber << ")";
                return oss.str();
            }
        }
    }
    return "Windows";
#else
    struct utsname buf;
    if (uname(&buf) == 0)
    {
        std::ostringstream oss;
        oss << buf.sysname << " " << buf.release;
        return oss.str();
    }
    return "Unknown";
#endif
}

/// Get total system memory (e.g., "16 GB")
inline std::string getMemoryInfo()
{
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo))
    {
        double gb = static_cast<double>(memInfo.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
    {
        double gb = static_cast<double>(pages) * page_size / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << gb << " GB";
        return oss.str();
    }
#endif
    return "Unknown";
}

/// Get compiler information (e.g., "GCC 11.2.0" or "MSVC 19.29 (Release)")
inline std::string getCompilerInfo()
{
#if defined(_MSC_VER)
    std::ostringstream oss;
    oss << "MSVC " << _MSC_VER / 100 << "." << _MSC_VER % 100;
#if defined(_DEBUG)
    oss << " (Debug)";
#else
    oss << " (Release)";
#endif
    return oss.str();
#elif defined(__clang__)
    std::ostringstream oss;
    oss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    return oss.str();
#elif defined(__GNUC__)
    std::ostringstream oss;
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return oss.str();
#else
    return "Unknown Compiler";
#endif
}

/// Get supported SIMD instruction sets (e.g., "SSE3, SSE4.1, SSE4.2, AVX, AVX2")
inline std::string getSimdInfo()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int regs[4];
    std::vector<std::string> features;

#ifdef _WIN32
    __cpuid(reinterpret_cast<int*>(regs), 1);
#else
    __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[2] & (1 << 0)) features.push_back("SSE3");
    if (regs[2] & (1 << 19)) features.push_back("SSE4.1");
    if (regs[2] & (1 << 20)) features.push_back("SSE4.2");
    if (regs[2] & (1 << 28)) features.push_back("AVX");

#ifdef _WIN32
    __cpuidex(reinterpret_cast<int*>(regs), 7, 0);
#else
    __get_cpuid_count(7, 0, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

    if (regs[1] & (1 << 5)) features.push_back("AVX2");
    if (regs[1] & (1 << 16)) features.push_back("AVX512F");

    if (features.empty())
        return "None detected";

    std::ostringstream oss;
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (i > 0) oss << ", ";
        oss << features[i];
    }
    return oss.str();
#else
    return "N/A (non-x86)";
#endif
}

}  // namespace platform_info

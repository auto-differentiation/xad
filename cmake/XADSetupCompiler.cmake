##############################################################################
#   
#  Setup of compiler and corresponding warning levels, and linker flags
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   
##############################################################################

# check if 64bit machine 
if(CMAKE_SIZEOF_VOID_P MATCHES "8")
  set(ARCH 64)
else()
  set(ARCH 32)
endif()

find_package(Threads REQUIRED)  # needed for thread-local 

# set global compiler flag variables, that we'll use when building things
if(MSVC)
    set(xad_cxx_flags -nologo -utf-8 -D_UNICODE -DUNICODE -DWIN32_LEAN_AND_MEAN -DWIN32 -D_WIN32)
    # flag warnings as errors
    set(xad_cxx_flags_warnings -W4 -WX)
    if(XAD_USE_STRONG_INLINE)
        list(APPEND xad_cxx_flags_warnings -wd4714)  # function marked forceinline wasn't inlined
    endif()
    if(NOT "${CMAKE_GENERATOR}" MATCHES "Ninja")
        list(APPEND xad_cxx_flags -MP)
    endif()
    if(XAD_SIMD_OPTION STREQUAL "SSE2" AND NOT ARCH EQUAL 64)
        set(xad_cxx_extra "/arch:SSE2")
    else()
        set(xad_cxx_extra "/arch:${XAD_SIMD_OPTION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(xad_cxx_flags -Wall -Wshadow -Wconversion)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # otherwise we get clashes with complex headers and other things on MacOS
        list(APPEND xad_cxx_flags -stdlib=libc++ -mmacosx-version-min=10.9)
    endif()
    set(xad_cxx_flags_warnings -Werror -W -Wpointer-arith -Wreturn-type -Wcast-qual -Wwrite-strings -Wswitch -Wunused-parameter -Wcast-align -Wchar-subscripts -Winline -Wredundant-decls)
    if(XAD_SIMD_OPTION STREQUAL SSE2)
        set(xad_cxx_extra -msse2)
    elseif(XAD_SIMD_OPTION STREQUAL AVX)
        set(xad_cxx_extra )#-mavx)
    elseif(XAD_SIMD_OPTION STREQUAL AVX2)
        set(xad_cxx_extra )#-mavx2)
    elseif(XAD_SIMD_OPTION STREQUAL AVX512)
        set(xad_cxx_extra -march=cascadelake)
    elseif(XAD_SIMD_OPTION STREQUAL NATIVE)
        set(xad_cxx_extra  -march=native)
    elseif(XAD_SIMD_OPTION STREQUAL APPLE_M1)
        set(xad_cxx_extra -march=armv8.5-a)
    endif()
    set(xad_cxx_asan_flags -fsanitize=address -fno-omit-frame-pointer)
    set(xad_link_asan_flags -fsanitize=address)
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(xad_cxx_flags -Wall -Wshadow -Wconversion)
    set(xad_cxx_flags_warnings -Werror -Wextra)
    if(XAD_SIMD_OPTION STREQUAL SSE2)
        set(xad_cxx_extra -msse2)
    elseif(XAD_SIMD_OPTION STREQUAL AVX)
        set(xad_cxx_extra )#-mavx)
    elseif(XAD_SIMD_OPTION STREQUAL AVX2)
        set(xad_cxx_extra )#-mavx2)
    elseif(XAD_SIMD_OPTION STREQUAL AVX512)
        set(xad_cxx_extra  -march=cascadelake)
    elseif(XAD_SIMD_OPTION STREQUAL NATIVE)
        set(xad_cxx_extra  -march=native)
    elseif(XAD_SIMD_OPTION STREQUAL APPLE_M1)
        set(xad_cxx_extra  -march=armv8.5-a)
    endif()
    set(xad_cxx_asan_flags -fsanitize=address -fno-omit-frame-pointer)
    set(xad_link_asan_flags -fsanitize=address)
else()
    # Note: Add new compilers here with their appropriate settings
    message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()


#
# Add an XAD library of the given Type (STATIC/SHARED)
#
# Works the same as the standard add_library, but adds common settings
#
function(xad_add_library name type)
    # setup suffix
    if(MSVC_VERSION VERSION_GREATER_EQUAL 1900)
        set(archsuff "${ARCH}")
        string(SUBSTRING "${MSVC_VERSION}" 2 1 toolset_version)  # extract the 19?x 
        set(comp "_vc14${toolset_version}")
        if(XAD_STATIC_MSVC_RUNTIME)
            set(rls_suffix "_mt")
            set(dbg_suffix "_mtd")
        else()
            set(rls_suffix "_md")
            set(dbg_suffix "_mdd")
        endif()
    elseif(UNIX)
        SET(comp "")
        SET(rls_suffix "")
        SET(dbg_suffix "_d")
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()

    add_library(${name} ${type} ${ARGN})
    add_library(XAD::${name} ALIAS ${name})
    target_compile_options(${name} PRIVATE ${xad_cxx_flags} ${xad_cxx_extra})
    if(XAD_WARNINGS_PARANOID)
        target_compile_options(${name} PRIVATE ${xad_cxx_flags_warnings})
    endif()
    if(XAD_ENABLE_ADDRESS_SANITIZER)
        target_compile_options(${name} PRIVATE ${xad_cxx_asan_flags})
        target_link_options(${name} PRIVATE ${xad_link_asan_flags})
    endif()
    set(rls_name "${name}${archsuff}${comp}${rls_suffix}")
    set(dbg_name "${name}${archsuff}${comp}${dbg_suffix}")
    set_target_properties(${name} PROPERTIES
        ARCHIVE_OUTPUT_NAME_DEBUG ${dbg_name}
        ARCHIVE_OUTPUT_NAME ${rls_name}
        DEBUG_POSTFIX ""
        RELEASE_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        MINSIZEREL_POSTFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        PDB_NAME "${rls_name}"
        PDB_NAME_DEBUG "${dbg_name}"
        COMPILE_PDB_NAME "${rls_name}"
        COMPILE_PDB_NAME_DEBUG "${dbg_name}"
        POSITION_INDEPENDENT_CODE "${XAD_POSITION_INDEPENDENT_CODE}"
    )

    target_link_libraries(${name} PUBLIC Threads::Threads)
    target_compile_features(${name} PUBLIC cxx_std_11)
    if(XAD_STATIC_MSVC_RUNTIME)
        set_target_properties(${name} PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set_target_properties(${name} PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    if(MSVC)
        # in MSVC/Debug, we used checked iterators, and MSVC displays a deprecation warning
        # without this flag
        target_compile_definitions(${name} PUBLIC "$<$<CONFIG:DEBUG>:_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING>")
    endif()

    list(FIND ARGN "XAD/EigenCompatibility.hpp" eigen_index)
    if(eigen_index GREATER -1)
        find_package(Eigen3 3.3 REQUIRED NO_MODULE)
        target_link_libraries(${name} PRIVATE Eigen3::Eigen)
        target_include_directories(${name} PRIVATE ../../../eigen)
    endif()
endfunction()


# Adds an exectuable with all the common compilation settings
function(xad_add_executable name)
    add_executable(${name} ${ARGN})
    if(MSVC)
        target_compile_options(${name} PRIVATE -bigobj)
    endif()
    target_compile_options(${name} PRIVATE ${xad_cxx_flags} ${xad_cxx_extra})
    if(XAD_WARNINGS_PARANIOD)
        target_compile_options(${name} PRIVATE ${xad_cxx_flags_warnings})
    endif()
    if(XAD_ENABLE_ADDRESS_SANITIZER)
        target_compile_options(${name} PRIVATE ${xad_cxx_asan_flags})
        target_link_options(${name} PRIVATE ${xad_link_asan_flags})
    endif()
    if(XAD_STATIC_MSVC_RUNTIME)
        set_target_properties(${name} PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set_target_properties(${name} PROPERTIES 
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    set_target_properties(${name} PROPERTIES
        DEBUG_POSTFIX ""
        RELEASE_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        MINSIZEREL_POSTFIX ""
    )
endfunction()

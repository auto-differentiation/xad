cmake_minimum_required(VERSION 3.15.2)
cmake_policy(SET CMP0092 NEW)
project(xad_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17) # TEMPORARY
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(googlebenchmark)

if(NOT TARGET XAD::xad)
    find_package(XAD REQUIRED)
    get_target_property(incdir XAD::xad INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "XAD include dir: ${incdir}")
endif()


function(xad_add_benchmark name)
    cmake_parse_arguments(ARGS "" "" "SOURCES;TEST_ARGS" ${ARGN})

    if(NOT ARGS_SOURCES)
        message(FATAL_ERROR "The SOURCES argument must be given for samples")
    endif()

    add_executable(${name} ${ARGS_SOURCES})
    target_link_libraries(${name} PRIVATE XAD::xad)
    target_link_libraries(${name} PRIVATE benchmark::benchmark)
    target_link_libraries(${name} PRIVATE benchmark::benchmark_main)
    target_include_directories(${name} PRIVATE ../common)
    get_target_property(msvcrt XAD::xad MSVC_RUNTIME_LIBRARY)
    if(MSVC AND msvcrt)
        set_target_properties(${name} PROPERTIES MSVC_RUNTIME_LIBRARY "${msvcrt}")
    endif()
    if(NOT XAD_SAMPLES_MAIN)
        add_test(NAME sample_${name} COMMAND ${name} ${ARGS_TEST_ARGS})
        set_target_properties(${name} PROPERTIES FOLDER "samples")
        if(XAD_ENABLE_ADDRESS_SANITIZER)
            target_compile_options(${name} PRIVATE ${xad_cxx_asan_flags})
            target_link_options(${name} PRIVATE ${xad_link_asan_flags})
        endif()
    endif()
endfunction()

add_subdirectory(Hessian)
add_subdirectory(Jacobian)
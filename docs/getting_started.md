---
description: >
  Installation, building, configuration, and testing instructions for the XAD automatic differentiation tool.
hide:
  - navigation
---


# Getting Started

## Building XAD

### Prerequisites

*   [CMake](https://cmake.org), version 3.15 or newer
*   Linux: GCC 5.4 or newer, or Clang 11 or newer
*   Windows:
    *   Visual Studio 2015 or newer
    *   Visual Studio with Clang toolset, 2019 or newer
*   MacOS: 10.9 or higher, with Apple Clang 11 or newer
*   Git client

(See [tested platforms](#tested-platforms) for the list of platforms covered by continuous integration.)

### Cloning the Repository

```bash
    git clone https://github.com/auto-differentiation/XAD.git
```

### Building

1.  Create a directory for the build artefacts
    ```bash
    cd XAD
    mkdir build
    cd build
    ```
2.  Run cmake to generate the build files
    ```bash
    cmake ..
    ```
3.  Build using the native build system or with the generic cmake build command
    ```bash
    cmake --build .
    ```

### Running the tests

The tests are executed with the `test` target:

```bash
cmake --build . --target test
```

Alternatively, `ctest` can be used to run them:

```bash
ctest
```

Or if only the unit tests should be run, the `xad_test` executable in the bin directory
can be executed directly.

### Installing

Run the `install` build target to place the header files, library files, docs, and samples
into the `CMAKE_INSTALL_PREFIX` (configurable with CMake).

```bash
cmake --install .
```

## Integration Approaches

In order to use XAD as part of other code, we recommend one of the following approaches.

### 1: Submodule + CMake

If your codebase is using CMake, XAD can be integrated easily into your project
by adding it as a git submodule.

To add the submodule in a subdirectory `extern/XAD`:

```bash
git submodule add https://github.com/auto-differentiation/XAD.git extern/XAD
```

Users then need to clone recursively (`git clone --recursive ...`) or initialise and update
the submodules (`git submodule init && git submodule update`).
More information about submodules can be found in the Git documentation.

To add XAD to the project, all that is needed in one of the `CMakeLists.txt` files is
to add the xad directory, and then link the relevant libraries or executables to `XAD::xad`:

```cmake
add_subdirectory(extern/XAD)

add_executable(some_executable ...)
target_link_libraries(some_executable PRIVATE XAD::xad)
```

### 2: FetchContent + CMake

The CMake FetchContent module allows to clone the git repository at configure-time into the
build folder and add it to your project after:

```cmake
include(FetchContent)

FetchContent_Declare(XAD
    GIT_REPOSITORY https://github.com/auto-differentiation/XAD.git
    GIT_TAG 1.1.0    # pick a tag, hash, or branch here
)
FetchContent_MakeAvailable(XAD)
```

Note that this requires at least CMake version 3.14.

### 3: Install XAD and Link

Another approach is to install XAD into a convenient prefix (e.g. `/usr/local/`) first
(instructions above, setting `CMAKE_INSTALL_PREFIX` appropriately).
Note that the package can also be zipped on one machine and downloaded/extracted on another.

**Important:** Since XAD is built as a static library, be careful to use the same compiler and flags for your project as well as XAD itself. Otherwise
the binaries may not be compatible. We therefore recommend to the subproject
approach, building from source within your project.
The library builds very fast.

#### CMake

Then, when you use CMake, you can setup your project to find the XAD dependency in a `CMakeLists.txt` file as:

```cmake
find_package(XAD REQUIRED)
```

If XAD is installed in a standard location, CMake automatically looks for it there and finds it.
Otherwise, the `CMAKE_PREFIX_PATH` variable can be set at configure-time to
add a different directory to its search path:

```bash
cmake /path/to/src -DCMAKE_PREFIX_PATH=/path/to/XAD/installprefix
```

#### Other Build Tools

If your project does not use CMake, an installed package can also be linked by adding the following settings:

*   Add `/path/to/XAD/include` to the compiler's include path
*   Enable at least C++ 11 support (`-std=c++11` in GCC)
*   Enable threading (requires `-pthread` in GCC for compile and link)
*   Add the library path `/path/to/XAD/lib` to the linker search paths
*   Link `libxad.a` (Release) or `libxad_d.a` (Debug) - or the alternative names on Windows

## Tuning Behaviour and Performance

A number of options are available via CMake to control the build
and tune the performance.
They can be specified using the CMake command-line with `-DVARIABLE=value`,
or with the CMake GUI.

Influential variables controlling the build are:

| Variable | Description | Default |
|---|---|---|
| `XAD_ENABLE_TESTS` | Enable building tests and samples. | `ON` if main project<br>`OFF` if sub project |
| `XAD_WARNINGS_PARANOID` | Enable a high warning level and flag warnings as errors. | `ON` |
| `XAD_STATIC_MSVC_RUNTIME` | Use the static multi-threaded runtime in Visual C++ (default is dynamic) |
| `XAD_POSITION_INDEPENDENT_CODE` | Generate position-indepent code, i.e. allow linking into a shared library. | `ON` |
| `XAD_ENABLE_ADDRESS_SANITIZER` | Enable address sanitizer (leak detector) - GCC/Clang only. | `OFF` |

Options with an impact on the performance of the tape in adjoint mode (application-specific).
These should not be changed in client code after the tape has been compiled:

| Variable | Description | Default |
|---|---|---|
| `XAD_SIMD_OPTION` | Select between `SSE2`, `AVX`, `AVX2`, and `AVX512` instruction sets. Only enable what the target CPU supports.  | `AVX` |
| `XAD_TAPE_REUSE_SLOTS` | Keep track of unused slots in tape and re-use them (less memory, more compute) | `OFF` |
| `XAD_NO_THREADLOCAL` | Disable thread-local tapes (use with single-threaded code only | `OFF` |

Options that can be set by client code as well, adjusting settings after the
XAD library has already been compiled (in `Config.hpp` or client code compiler definitions):

| Variable | Description | Default |
|---|---|---|
| `XAD_USE_STRONG_INLINE` | Force inlining expression templates, rather than letting the compiler decide. (faster, slow compilation, possible compiler crashes) | `OFF` |
| `XAD_ALLOW_INT_CONVERSION` | Add real -> integer conversion operator, similar to `double`. This may result missing some variable dependency tracking for AAD. | `ON`  |

## Adapting the User Documentation

The user documentation uses the popular [MkDocs Material tool](https://squidfunk.github.io/mkdocs-material/).
It is entirely generated from the markdown files located in the `docs` folder
and can be generated easily using docker by running in the project root:

```bash
docker run --rm -it -p 8000:8000 -e GOOGLE_ANALYTICS_KEY=devkey -e LATEST_VERSION=dev -v ${PWD}:/docs squidfunk/mkdocs-material
```

This will serve the documentation on `http://127.0.0.1:8000` and watch for changes in the
`docs` folder automatically.
The files can now be modified and conveniently viewed.

Note that the environment variables `GOOGLE_ANALYTICS_KEY` and `LATEST_VERSION` must be set in the above
docker command, otherwise mkdocs will show an error that `mkdocs.yaml` does not exist.

## Getting Help

If you have found an issue, want to report a bug, or have a feature request, please raise a [GitHub issue](https://github.com/auto-differentiation/XAD/issues).

For general questions about XAD, sharing ideas, engaging with community members, etc, please use [GitHub Discussions](https://github.com/auto-differentiation/XAD/discussion).

## Tested Platforms

The following platforms are part of the [continuous integration workflow][ci], i.e. they are tested on each commit. You can use other configurations at your own risk,
or [submit a PR](https://github.com/auto-differentiation/XAD/blob/feature/new-site/CONTRIBUTING.md) to include it in the [CI workflow][ci].

| Operating System     |  Compiler                         | Configurations                                      | Test Coverage Recorded |
|----------------------|-----------------------------------|-----------------------------------------------------|-------------------|
| Windows Server 2019  | Visual Studio 2015 (Toolset 14.0) | Debug, Release                                      | no       |
| Windows Server 2022  | Visual Studio 2017 (Toolset 14.1) | Debug, Release                                      | no       |
| Windows Server 2022  | Visual Studio 2019 (Toolset 14.2) | Debug, Release                                      | no       |
| Windows Server 2022  | Visual Studio 2022 (Toolset 14.3) | Debug, Release                                      | no       |
| Windows Server 2022  | Clang 14.0         (Toolset 14.3) | Debug, Release                                      | no       |
| Ubuntu 16.04         | GCC 5.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 17.10         | GCC 6.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 17.10         | GCC 7.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 18.04         | GCC 8.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 19.10         | GCC 9.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 20.04         | GCC 10.3.0                        | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 20.04         | GCC 11.1.0                        | Debug, Release, + both with `XAD_TAPE_REUSE_SLOTS`  | yes      |
| Ubuntu 20.04         | Clang 11.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 22.04         | Clang 12.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 22.04         | Clang 13.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| Ubuntu 22.04         | Clang 14.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no       |
| MacOS 12.6.5         | AppleClang 14.0.0                 | Debug, Release                                      | yes      |

[cmake]: https://cmake.org

[ci]: https://github.com/auto-differentiation/XAD/blob/main/.github/workflows/ci.yml

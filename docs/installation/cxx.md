---
title: "C++ Setup Guide"
description: "Step-by-step guide to seamlessly integrate XAD for C++ across platforms with prerequisites, building, and testing instructions"
---
# C++ 

## Prerequisites

=== "Windows"

    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Compiler/IDE options:
        *   Visual Studio 2015 or newer (for Microsoft compilers)
        *   Visual Studio 2019 or newer with Clang toolset (for Clang)
    *   [Git client](https://git-scm.com/downloads)

    For Windows, we recommend the latest [Visual Studio 2022 IDE](https://visualstudio.microsoft.com/downloads/) with its integrated CMake support.

=== "Linux"

    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Compiler Options:
        *   GCC 5.4 or newer
        *   Clang 11 or newer
    *   [Git client](https://git-scm.com/downloads)

=== "MacOS"

    *   MacOS 10.9.5 or newer
    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Apple Clang 11 or newer
    *   [Git client](https://git-scm.com/downloads)

For the full list of platforms and compilers covered by continuous integration, see [tested platforms](#tested-platforms).

## Cloning the Repository

```
git clone https://github.com/auto-differentiation/xad.git
```

## Building

=== "Visual Studio 2019/2022"

    1. Use "Open Folder" to open the cloned directory
    2. Select the desired configuration from the top toolbar (e.g. `Release x64`), or add a new one using `Manage Configurations...`
    3. Select Project > Build (or press ++f7++)

    See the documentation for [Visual Studio's built-in CMake support](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio) for more information.

=== "Command Line"

    Note: The compiler toolset should be in the path. In Windows, start a Visual Studio Developer command prompt with the desired toolset first.

    1.  Create a directory for the build artefacts
        ```
        cd xad
        mkdir build
        cd build
        ```
    2.  Run cmake to generate the build files
        ```
        cmake ..
        ```
    3.  Build using the native build system or with the generic cmake build command
        ```
        cmake --build .
        ```

=== "CMake GUI"

    1. Open the CMake GUI and select the source and build folders
    2. Click the `Configure` button, selecting the desired compiler toolset
    3. Click `Generate`, which creates the native build files provided.
    4. If the Visual Studio generator was chosen, click `Open Project` to start Visual Studio, where the solution can be built

    More information about how to use the CMake GUI is available from the [official CMake documentation](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#cmake-gui-tool).

## Running the Tests

The tests are managed by CMake's CTest tool. The can be executed as:

=== "Visual Studio Test Explorer"

    Open the Test > Test Explorer to run / debug individual tests or the full suite conveniently.
    Visual Studio discovers the CTest tests after each build automatically.
    Refer to the [documentation for the test explorer](https://learn.microsoft.com/en-us/visualstudio/test/run-unit-tests-with-test-explorer)
    for more information how to use this convenient tool.

=== "Command Line CTest"

    Run the ctest tool inside the build folder:

    ```
    ctest
    ```

    Note that the flag `-j<num_proc>` can be given to run `<num_proc>` parallel processes,
    and there are more [convenient ctest flags](https://cmake.org/cmake/help/latest/manual/ctest.1.html#run-tests) 
    for example for controlling output, ordering, or selecting specific tests.

=== "Test Executable"

    Most tests are combined into a single Google Test executable called `xad_test`,
    which can be found in the `test` directory in the build tree.
    This can also be executed directly:

    ```
    cd test
    ./xad_test
    ```

    It also provides parameters, e.g. for selecting specific tests, control outputs, which
    can be seen when running `./xad_test --help`

## Installing

Run the `install` build target to place the header files, library files, docs, and samples
into the `CMAKE_INSTALL_PREFIX` (configurable with CMake).

=== "Visual Studio 2019/2022"

    Select Build > Install from the menu.

=== "Command Line"

    Run the following from the build folder:

    ```
    cmake --install .
    ```

## Integration Approaches

In order to use XAD as part of other code, we recommend one of the following approaches.

### 1: Submodule + CMake

If your codebase is using CMake, XAD can be integrated easily into your project
by adding it as a git submodule.

To add the submodule in a subdirectory `extern/xad`:

```bash
git submodule add https://github.com/auto-differentiation/xad.git extern/xad
```

Users then need to clone recursively (`git clone --recursive ...`) or initialise and update
the submodules (`git submodule init && git submodule update`).
More [information about submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) can be found in the Git documentation.

To add XAD to the project, all that is needed in one of the `CMakeLists.txt` files is
to add the xad directory, and then link the relevant libraries or executables to `XAD::xad`:

```cmake
add_subdirectory(extern/xad)

add_executable(some_executable ...)
target_link_libraries(some_executable PRIVATE XAD::xad)
```

### 2: FetchContent + CMake

The CMake FetchContent module allows to clone the git repository at configure-time into the
build folder and add it to your project after:

```cmake
include(FetchContent)

FetchContent_Declare(XAD
    GIT_REPOSITORY https://github.com/auto-differentiation/xad.git
    GIT_TAG 1.2.0    # pick a tag, hash, or branch here
)
FetchContent_MakeAvailable(XAD)
```

The relevant libraries or executables can then be linked to `XAD::xad` in the usual way:

```cmake
add_executable(some_executable ...)
target_link_libraries(some_executable PRIVATE XAD::xad)
```

### 3: Install and Link

Another approach is to install XAD into a convenient prefix (e.g. `/usr/local/`) first
(instructions above, setting `CMAKE_INSTALL_PREFIX` appropriately).
Note that the package can also be zipped on one machine and downloaded/extracted on another.

!!! warning

    Since XAD is built as a static library, be careful to use the same compiler and flags 
    for your project and XAD itself. 
    Otherwise the binaries may not be compatible. We therefore recommend the subproject
    approach, building from source within your project.
    The library builds very fast.

**CMake**

Then, when you use CMake, you can setup your project to find the XAD dependency in a `CMakeLists.txt` file as:

```cmake
find_package(XAD REQUIRED)
```

If XAD is installed in a standard location, CMake automatically looks for it there and finds it.
Otherwise, the `CMAKE_PREFIX_PATH` variable can be set at configure-time to
add a different directory to its search path:

```bash
cmake /path/to/src -DCMAKE_PREFIX_PATH=/path/to/xad/installprefix
```

**Other Build Tools**

If your project does not use CMake, an installed package can also be linked by adding the following settings:

- Add `/path/to/xad/include` to the compiler's include search path
- Enable at least C++ 11 support (`-std=c++11` in GCC)
- Enable threading in Linux (requires `-pthread` in GCC for compile and link)
- Add the library path `/path/to/XAD/lib` to the linker search paths
- Link on of the following depending on the configuration:
  - Linux or Mac, Release: `libxad.a`
  - Linux or Mac, Debug: `libxad_d.a`
  - Windows 64bit, Release, toolset 14.3 (VS 2022): `xad64_vc143_md.lib`
  - Windows 64bit, Debug, toolset 14.3 (VS 2022): `xad64_vc143_mdd.lib`
  - Windows 64bit, Release, toolset 14.2 (VS 2019): `xad64_vc142_md.lib`
  - Windows 64bit, Debug, toolset 14.2 (VS 2019): `xad64_vc142_mdd.lib`
  - Windows 64bit, Release, toolset 14.1 (VS 2017): `xad64_vc141_md.lib`
  - Windows 64bit, Debug, toolset 14.1 (VS 2017): `xad64_vc141_mdd.lib`

## Tuning Behaviour and Performance

A number of options are available via CMake configuration variables to control the build
and tune the performance.
They can be configured as:

=== "Visual Studio 2019/2022"

    From the Configurations dropdown in the top toolbar, select `Manage Configurations...` to see the available
    configuration and [adjust the CMake variables](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio#configuring-cmake-projects).

=== "Command Line"

    Within the build directory, variables can be set either with the initial `cmake` command or at a later point as:

    ```
    cmake . -DVARIABLE=value
    ```

=== "CMake GUI"

    The variables with there default settings are listed in the main part of the GUI can can be changed directly.


Influential variables controlling the build are:

| Variable                        | Description                                                                | Default                                      |
| ------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------- |
| `XAD_ENABLE_TESTS`              | Enable building tests and samples.                                         | `ON` if main project<br>`OFF` if sub project |
| `XAD_WARNINGS_PARANOID`         | Enable a high warning level and flag warnings as errors.                   | `ON`                                         |
| `XAD_STATIC_MSVC_RUNTIME`       | Use the static multi-threaded runtime in Visual C++ (default is dynamic)   |
| `XAD_POSITION_INDEPENDENT_CODE` | Generate position-indepent code, i.e. allow linking into a shared library. | `ON`                                         |
| `XAD_ENABLE_ADDRESS_SANITIZER`  | Enable address sanitizer (leak detector) - GCC/Clang only.                 | `OFF`                                        |

Options with an impact on the performance of the tape in adjoint mode (application-specific).
These should not be changed in client code after the tape has been compiled:

| Variable               | Description                                                                                                    | Default |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- | ------- |
| `XAD_SIMD_OPTION`      | Select between `SSE2`, `AVX`, `AVX2`, and `AVX512` instruction sets. Only enable what the target CPU supports. | `AVX`   |
| `XAD_TAPE_REUSE_SLOTS` | Keep track of unused slots in tape and re-use them (less memory, more compute)                                 | `OFF`   |
| `XAD_NO_THREADLOCAL`   | Disable thread-local tapes (use with single-threaded code only                                                 | `OFF`   |

Options that can be set by client code as well, adjusting settings after the
XAD library has already been compiled and installed (in `Config.hpp` or client code pre-processor definitions):

| Variable                   | Description                                                                                                                         | Default |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `XAD_USE_STRONG_INLINE`    | Force inlining expression templates, rather than letting the compiler decide. (faster, slow compilation, possible compiler crashes) | `OFF`   |
| `XAD_ALLOW_INT_CONVERSION` | Add real -> integer conversion operator, similar to `double`. This may result missing some variable dependency tracking for AAD.    | `ON`    |


---
description: >
  Installation, building, configuration, and testing instructions for the XAD automatic differentiation tool.
hide:
  - navigation
---

# Installation

## Building XAD

### Prerequisites

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

### Cloning the Repository

```
git clone https://github.com/auto-differentiation/XAD.git
```

### Building

=== "Visual Studio 2019/2022"

    1. Use "Open Folder" to open the cloned directory
    2. Select the desired configuration from the top toolbar (e.g. `Release x64`), or add a new one using `Manage Configurations...`
    3. Select Project > Build (or press ++f7++)

    See the documentation for [Visual Studio's built-in CMake support](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio) for more information.

=== "Command Line"

    Note: The compiler toolset should be in the path. In Windows, start a Visual Studio Developer command prompt with the desired toolset first.

    1.  Create a directory for the build artefacts
        ```
        cd XAD
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

### Running the Tests

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

### Installing

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

To add the submodule in a subdirectory `extern/XAD`:

```bash
git submodule add https://github.com/auto-differentiation/XAD.git extern/XAD
```

Users then need to clone recursively (`git clone --recursive ...`) or initialise and update
the submodules (`git submodule init && git submodule update`).
More [information about submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) can be found in the Git documentation.

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
cmake /path/to/src -DCMAKE_PREFIX_PATH=/path/to/XAD/installprefix
```

**Other Build Tools**

If your project does not use CMake, an installed package can also be linked by adding the following settings:

- Add `/path/to/XAD/include` to the compiler's include search path
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

## Building the Documentation

The user documentation uses the popular [MkDocs Material tool](https://squidfunk.github.io/mkdocs-material/).
It is entirely generated from the markdown files located in the `docs` folder.
It can be edited with one of the following approaches.

### 1: Docker

The docs can be generated easily [using docker](https://docs.docker.com/get-docker/) by running in the project root:

=== "Windows"

    ```
    docker run --rm -it -p 8000:8000 -v "%cd%":/docs squidfunk/mkdocs-material
    ```

=== "Powershell, Linux, Mac"

    ```
    docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material
    ```

This will serve the documentation on [http://localhost:8000](http://localhost:8000) and watch for changes in the
`docs` folder automatically.
The files can now be modified and conveniently viewed in the browser.

### 2: Local Python

Alternatively, MkDocs can be installed locally into a Python environment and executed as follows
(we recommend a virtual environment as shown below).

Setup the environment and dependencies (first time):

=== "Windows"

    ```
    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install mkdocs-material mkdocs-minify-plugin mkdocs-redirects "pillow<10" "cairosvg>=2.5"
    ```

=== "Powershell"

    ```
    python -m venv .venv
    .venv\Scripts\activate.ps1
    pip install mkdocs-material mkdocs-minify-plugin mkdocs-redirects "pillow<10" "cairosvg>=2.5"
    ```

=== "Linux, Mac"

    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install mkdocs-material mkdocs-minify-plugin mkdocs-redirects "pillow<10" "cairosvg>=2.5"
    ```

Run mkdocs (in the activated environment):

```
mkdocs serve
```

This will serve the documentation on [http://localhost:8000](http://localhost:8000) and watch for changes in the
`docs` folder automatically.
The files can now be modified and conveniently viewed in the browser.

## Getting Help

If you have found an issue, want to report a bug, or have a feature request, please raise a [GitHub issue](https://github.com/auto-differentiation/XAD/issues).

For general questions about XAD, sharing ideas, engaging with community members, etc, please use [GitHub Discussions](https://github.com/auto-differentiation/XAD/discussions).

## Tested Platforms

The following platforms are part of the [continuous integration workflow][ci], i.e. they are tested on each commit. You can use other configurations at your own risk,
or [submit a PR](https://github.com/auto-differentiation/XAD/blob/feature/new-site/CONTRIBUTING.md) to include it in the [CI workflow][ci].

| Operating System    | Compiler                          | Configurations                                      | Test Coverage Recorded |
| ------------------- | --------------------------------- | --------------------------------------------------- | ---------------------- |
| Windows Server 2019 | Visual Studio 2015 (Toolset 14.0) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2017 (Toolset 14.1) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2019 (Toolset 14.2) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2022 (Toolset 14.3) | Debug, Release                                      | no                     |
| Windows Server 2022 | Clang 15.0 (Toolset 14.3)         | Debug, Release                                      | no                     |
| Ubuntu 16.04        | GCC 5.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 17.10        | GCC 6.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 17.10        | GCC 7.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 18.04        | GCC 8.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 19.10        | GCC 9.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 20.04        | GCC 10.3.0                        | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 20.04        | GCC 11.1.0                        | Debug, Release, + both with `XAD_TAPE_REUSE_SLOTS`  | yes                    |
| Ubuntu 20.04        | Clang 11.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 12.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 13.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 14.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 15.0.7                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| MacOS 12.6.5        | AppleClang 14.0.0                 | Debug, Release                                      | yes                    |

[cmake]: https://cmake.org
[ci]: https://github.com/auto-differentiation/XAD/blob/main/.github/workflows/ci.yml

---
title: "C++ QuantLib Integration: QuantLib-Risks"
description: >-
  Efficiently perform financial risk assessments in C++ using QuantLib with XAD's automatic differentiation.
---

# C++ QuantLib Integration: QuantLib-Risks

As a demonstrator of integration with real-world code, the latest release of [QuantLib](https://www.quantlib.org) is AAD-enabled with XAD.
The performance achieved on sample applications is many-fold superior to what has been reported previously with other tools.
This demonstrates production quality use of the XAD library in a code-base
of several hundred thousand lines.

A small adaptor module (also open-source) is required between the two projects, which contains build instructions
as well as XAD-specific tests and examples.

## Getting Started

### Prerequisites

=== "Windows"

    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Compiler/IDE options:
        *   Visual Studio 2017 or newer (for Microsoft compilers)
        *   Visual Studio 2019 or newer with Clang toolset (for Clang)
    *   [Git client](https://git-scm.com/downloads)
    *   A recent version of boost, for example using [Chocolatey](https://chocolatey.org/):
        *   For Visual Studio 2022: `choco install boost-msvc-14.3`
        *   For Visual Studio 2019: `choco install boost-msvc-14.2`
    *   Or using manual installers: [Boost Binaries on SourceForge](https://sourceforge.net/projects/boost/files/boost-binaries/)

    For Windows, we recommend the latest [Visual Studio 2022 IDE](https://visualstudio.microsoft.com/downloads/) with its integrated CMake support.

=== "Linux"

    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Compiler Options:
        *   GCC 5.4 or newer
        *   Clang 11 or newer
    *   [Git client](https://git-scm.com/downloads)
    *   A recent version of boost, for example:
        *   Ubuntu or Debian: `sudo apt install libboost-all-dev`
        *   Fedora or RedHat: `sudo yum install boost-devel`

=== "MacOS"

    *   MacOS 10.9.5 or newer
    *   [CMake](https://cmake.org), version 3.15 or newer
    *   Apple Clang 11 or newer
    *   [Git client](https://git-scm.com/downloads)
    *   A recent version of boost, for example:
        *   using [Homebrew](https://brew.sh/): `brew install boost`
        *   using [Mac Ports](https://www.macports.org/): `sudo port install boost`

### Repository Clone

### 1. Repository Clone/Checkout

Clone these three repositories into separate folders:

*   [https://github.com/auto-differentiation/QuantLib-Risks-Cpp.git](https://github.com/auto-differentiation/QuantLib-Risks-Cpp)
*   [https://github.com/auto-differentiation/xad.git](https://github.com/auto-differentiation/xad)
*   [https://github.com/lballabio/QuantLib.git](https://github.com/lballabio/QuantLib)

It is recommended to either use the latest master branch for all repositories involved,
or use matching tags between QuantLib and QuantLib-Risks-Cpp.

For the remainder of this tutorial, we assume the 3 repositories have been checked out into the following folder structure:

```
QuantLib-Risks-integration/
├─ QuantLib/
├─ xad/
├─ QuantLib-Risks-Cpp/
```

created from the master/main branches as:

```
mkdir QuantLib-Risks-integration
cd QuantLib-Risks-integration
git clone https://github.com/lballabio/QuantLib.git
git clone https://github.com/auto-differentiation/xad.git
git clone https://github.com/auto-differentiation/QuantLib-Risks-Cpp.git
```


### Building

The build is driven from the `QuantLib` directory - XAD and QuantLib-Risks are
inserted using [QuantLib's extension hook](https://www.quantlib.org/install/cmake.shtml#extensions).

QuantLib-Risks ships with a set of [user presets for CMake](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/blob/main/presets/CMakeUserPresets.json) that are designed to work together with 
the [Standard CMake presets](https://github.com/lballabio/QuantLib/blob/master/CMakePresets.json) that come with QuantLib itself. 
It is easiest to copy these user presets into the QuantLib checkout folder, as they contain the required settings to enable AAD in QuantLib via XAD,
and adjust paths and settings as needed. 

The project can then be built as follows:

=== "Visual Studio 2019/2022"

    1. Use "Open Folder" to open the QuantLib directory
    2. Select the desired preset from the top toolbar (e.g. `windows-xad-msvc-release`)
    3. Select Project > Build (or press ++f7++)

    See the documentation for [Visual Studio's built-in CMake support](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio) for more information.

=== "Command Line"

    Note: The compiler toolset should be in the path. In Windows, start a Visual Studio Developer command prompt with the desired toolset first.

    ```
    cd QuantLib
    cmake --preset windows-xad-msvc-release
    cd build/windows-xad-msvc-release
    cmake --build .
    ```

=== "CMake GUI"

    1. Open the CMake GUI and select the source folder for QuantLib
    2. Select a preset which includes XAD, e.g. `windows-xad-msvc-release`
    3. Click `Generate`, which creates the native build files provided (in Windows, you should choose one of the Visual Studio generators when using the GUI).
    4. If the Visual Studio generator was chosen, click `Open Project` to start Visual Studio, where the solution can be built

    More information about how to use the CMake GUI is available from the [official CMake documentation](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#cmake-gui-tool).
    
=== "Manual CMake Variables"

    In case you want to adjust the CMake settings for QuantLib manually, here are the CMake variables that should be set
    in the QuantLib build (adjust the paths according to your checkout location):

    *   `QL_EXTERNAL_SUBDIRECTORIES=/path/to/xad;/path/to/QuantLib-Risks-Cpp`
    *   `QL_EXTRA_LINK_LIBRARIES=QuantLib-Risks`
    *   `QL_NULL_AS_FUNCTIONS=ON`
    *   `XAD_STATIC_MSVC_RUNTIME=ON`  (if using Windows)


### Running the Tests

There are two test executables that get built - the regular QuantLib
test suite with all the standard tests from the mainline repository,
as well as the QuantLib XAD test suite from the QuantLib-Risks-Cpp repository.
Both are using the overloaded XAD type for QuantLib's `Real`,
but only the XAD suite checks for the correctness of the derivatives as well.

Both are regitered with CMake's CTest tool, so they can be run as:

=== "Visual Studio 2019/2022"

    Open the Visual Studio Test Explorer from `Test` -> `Test Explorer` 
    in the menu. It should run and discover all the tests, and they can be
    run directly from this window.

=== "Command Line (CTest)"

    From within the build folder, e.g. `build/linux-xad-gcc-release`,
    the tests can be executed using:

    ```
    ctest .
    ```

=== "Command Line (Executables)"

    The test exectuable can also be run directly. For example, if the build
    folder is `build/linux-xad-gcc-release`, they can be run as:

    QuantLib regular:
    ```
    cd test-suite
    ./quantlib-test-suite --log_level=message
    ```
    QuantLib-Risks
    ```
    cd QuantLib-Risks-Cpp/test-suite
    ./quantlib-risks-test-suite --log_level=message
    ```


### Running the Examples

Apart from the regular QuantLib examples, there are XAD-specific examples
in the QuantLib-Risks-Cpp repository, in the `Examples` folder.
These demonstrate the use of XAD to calculate derivatives using AAD.

They are built into the selected build folder within the `QuantLib` folder, 
e.g. `build/linux-xad-gcc-release/QuantLib-Risks-Cpp/Examples`,
and can be executed directly.

## Benchmarks

Some of the examples in QuantLib-Risks are enabled for benchmarking. 
That is, the performance of the pricing and sensitivity calculation 
is measured, averaged over several iterations, for accurate performance reporting.

Further, setting the CMake option `QLRISKS_DISABLE_AAD` to `ON` builds
QuantLib and QuantLib-Risks with the default `double` datatype,
enabling measurement of the same examples without the overheads involved in using
a custom active data type (this is exposed in the `*-noxad-*` build presets).
The benchmark-enabled examples calculate sensitivities using finite differences 
in that case, 
which also allows verifying correctness of the result against XAD.

Bechmark results:

| QuantLib Example           | Sensitivities | Valuation run (ms) | AAD run (ms) | AAD vs Valuation |
|----------------------------|--------------:|-------------------:|-------------:|-----------------:|
| Equity Option Portfolio    |            98 |               2.83 |         7.00 |            2.47x |
| Barrier Option Replication |            13 |               1.48 |         4.16 |            2.81x |
| Swap Portfolio             |            55 |              26.05 |        36.28 |            1.39x |
| Multicurve Bootstrapping   |            65 |             192.11 |       299.63 |            1.56x |

Benchmark configuration:

-   QuantLib version: 1.30
-   XAD version: 1.2.0
-   XAD configuration: `XAD_USE_STRONG_INLINE=ON`, `XAD_NO_THREADLOCAL=ON`, `XAD_SIMD_OPTION=AVX512`
-   OS: Windows Server 2022 Datacenter
-   Compiler: Visual Studio 2022, 17.6.1
-   RAM: 64GB
-   CPU: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz

## Getting Help

If you have found an issue, want to report a bug, or have a feature request, please raise a [GitHub issue](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/issues).

For general questions about the QuantLib to XAD integration, sharing ideas, engaging with community members, etc, please use [GitHub Discussions](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/discussions).

## Continuous Integration

To ensure continued compatibility with QuantLib's master branch,
[automated CI/CD checks](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/actions/workflows/ci.yaml) are running in the QuantLib-Risks-Cpp repository on a daily basis.
Potential breaks (for example do to changes in QuantLib) are therefore
detected early and fixed quickly.

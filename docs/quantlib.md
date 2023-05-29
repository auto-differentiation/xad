---
description: >-
  As a demonstrator for integration with real-world code, as of version 1.28 (October 2022), QuantLib is AAD-enabled with XAD.
hide:
  - navigation
  - footer
---

# QuantLib Integration

As a demonstrator of integration with real-world code, the latest release of [QuantLib](https://www.quantlib.org) is AAD-enabled with XAD.
The performance achieved on sample applications is many-fold superior to what has been reported previously with other tools.
This demonstrates production quality use of the XAD library in a code-base
of several hundred thousand lines.

A small adaptor module (also open-source) is required between the two projects, which contains build instructions
as well as XAD-specific tests and examples.

## Getting Started

### 1. Repository Clone/Checkout

Clone these three repositories into separate folders:

*   [https://github.com/auto-differentiation/qlxad.git](https://github.com/auto-differentiation/qlxad)
*   [https://github.com/auto-differentiation/XAD.git](https://github.com/auto-differentiation/XAD)
*   [https://github.com/lballabio/QuantLib.git](https://github.com/lballabio/QuantLib)

Note: We recommend either using the lastest master branch for all repositories involved.
These are tested against each other on a daily basis and errors are corrected quickly.

### 2. Install Boost

A recent version of boost is a requirement for building QuantLib.
If you do not have it already, you need to install it into a system path.
You can do that in one of the following ways, depending on your system:

*   Ubuntu or Debian: `sudo apt install libboost-all-dev`
*   Fedora or RedHat: `sudo yum install boost-devel`
*   MacOS using [Homebrew](https://brew.sh/): `brew install boost`
*   MacOS using [Mac Ports](https://www.macports.org/): `sudo port install boost`
*   Windows using [Chocolatey](https://chocolatey.org/):

    *   For Visual Studio 2022: `choco install boost-msvc-14.3`
    *   For Visual Studio 2019: `choco install boost-msvc-14.2`
    *   For Visual Studio 2017: `choco install boost-msvc-14.1`

*   Windows using manual installers: [Boost Binaries on SourceForge](https://sourceforge.net/projects/boost/files/boost-binaries/)

### 3. Install CMake

You will also need a recent version of CMake (minimum version 3.15.0).
You can also install this with your favourite package manager
(e.g. apt, yum, homebrew, chocolatey as above), or obtain it from
the [CMake downloads page](https://cmake.org/download/).

Note that Microsoft ships Visual Studio with a suitable version
command-line only version of CMake since Visual Studio 2019
(the Visual Studio 2017 CMake version is outdated).
It is available in the `PATH` from a Visual Studio command prompt
and can alternatively be used directly from the IDE.

### 4. QuantLib CMake Configuration

The build is driven from the QuantLib directory - XAD and qlxad are
inserted using [QuantLib's extension hook](https://www.quantlib.org/install/cmake.shtml#extensions).

Configure the QuantLib CMake build with setting the following parameters:

*   `QL_EXTERNAL_SUBDIRECTORIES=/path/to/xad;/path/to/qlxad`
*   `QL_EXTRA_LINK_LIBRARIES=qlxad`
*   `QL_NULL_AS_FUNCTIONS=ON`
*   `XAD_STATIC_MSVC_RUNTIME=ON`

For Linux, the command-line for this is:

```shell
cd QuantLib
mkdir build
cd build
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="`pwd`/../../xad;`pwd`/../../qlxad" \
  -DQL_EXTRA_LINK_LIBRARIES=qlxad \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_STATIC_MSVC_RUNTIME=ON
```

In Windows, you can use the CMake GUI to generate the build files,
setting the same variables as above.

### 5. Building

The generated build files can now be built using the regular native
build tools. For example, in Linux `make` can be run,
and in Visual Studio, the solution can be opened and built.
Note that we recommend Release mode for Windows builds.

### 6. Running the Tests

There are two test executables that get built - the regular QuantLib
test suite with all the standard tests from the mainline repository,
as well as the QuantLib XAD test suite from the qlxad repository.
Both are using the overloaded XAD type for `double`,
but only the XAD suite checks for the correctness of the derivatives as well.

These executables can simply be run to execute all the tests.
We recommend to use the parameter `--log_level=message` to see the test
progress.
Alternatively, CTest can also be used to execute them.

### 7. Running the Examples

Apart from the regular QuantLib examples, there are XAD-specific examples
in the qlxad repository, in the `Examples` folder.
These demonstrate the use of XAD to calculate derivatives using AAD.

## Benchmarks

Some of the examples in qlxad are enabled for benchmarking. 
That is, the performance of the pricing and sensitivity calculation 
is measured, averaged over several iterations, for accurate performance reporting.

Further, setting the CMake option `QLXAD_DISABLE_AAD` to `ON` builds
QuantLib and qlxad with the default `double` datatype,
enabling measurement of the same examples without the overheads involved in using
a custom active data type.
The benchmark-enabled examples calculate sensitivities using finite differences 
in that case, 
which also allows verifying correctness of the result against XAD.

Bechmark results:

| Example                    | Sensitivites | Valuation run (ms) | AAD run (ms) | AAD vs Valuation |
|----------------------------|-------------:|-------------------:|-------------:|-----------------:|
| Equity Option Portfolio    |           98 |              2.832 |        7.004 |            2.47x |
| Barrier Option Replication |           13 |              1.481 |        4.162 |            2.81x |
| Swap Portfolio             |           55 |             26.050 |       36.280 |            1.39x |
| Multicurve Bootstrapping   |           65 |            192.107 |      299.631 |            1.56x |

Benchmark configuration:

-   QuantLib version: 1.30
-   XAD version: 1.2.0
-   XAD configuration: `XAD_USE_STRONG_INLINE=ON`, `XAD_NO_THREADLOCAL=ON`, `XAD_SIMD_OPTION=AVX512`
-   OS: Windows Server 2022 Datacenter
-   Compiler: Visual Studio 2022, 17.6.1
-   RAM: 64GB
-   CPU: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz

## Getting Help

If you have found an issue, want to report a bug, or have a feature request, please raise a [GitHub issue](https://github.com/auto-differentiation/qlxad/issues).

For general questions about the QuantLib to XAD integration, sharing ideas, engaging with community members, etc, please use [GitHub Discussions](https://github.com/auto-differentiation/qlxad/discussions).

## Continuous Integration

To ensure continued compatibility with QuantLib's master branch,
[automated CI/CD checks](https://github.com/xcelerit/qlxad/actions/workflows/ci.yaml) are running in the qlxad repository on a daily basis.
Potential breaks (for example do to changes in QuantLib) are therefore
detected early and fixed quickly.

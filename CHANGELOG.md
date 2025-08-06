# Changelog

All notable changes to XAD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed


## [1.8.0] - 2025-08-05

### Added

- **Vector Forward Mode**: Added support for a vector forward mode, where multiple derivatives can be calculated at once ([#177](https://github.com/auto-differentiation/xad/pull/177))
- **Vector Adjoint Mode**: Added support for a vector adjoint mode, where derivatives w.r.t. multiple outputs can be calculated jointly with AAD ([#177](https://github.com/auto-differentiation/xad/pull/177))
- **Direct Mode**: Added a mode for forward and reverse mode without expression templates, to ease integration and debugging ([#177](https://github.com/auto-differentiation/xad/pull/177))
- **Added std::fma Function**: Added support for the standard math function `std::fma` ([#175](https://github.com/auto-differentiation/xad/pull/175))
- **Eigen Compatibility**: Added support and a wide range of tests for using XAD within the Eigen Library ([#174](https://github.com/auto-differentiation/xad/pull/174))
- **Jacobian Performance Optimisation**: Faster discovery of Jacobian co-domain ([#163](https://github.com/auto-differentiation/xad/pull/163/files) by @raneamri)
- **Improved CI/CD Pipelines**: Modern compilers, better tests, windows fixes ([#164](https://github.com/auto-differentiation/xad/pull/164)), and further improvements ([#177](https://github.com/auto-differentiation/xad/pull/177))

### Removed

- Support for Visual Studio 2015 and 2017 (toolchains 14.0 and 14.1)
- Support for GCC 5 and 6

### Fixed

- **libc++ type trait fixes**: Fixes for libc++ (MacOS) for random number type traits ([#170](https://github.com/auto-differentiation/xad/pull/170))
- **copysign function in MSVC**: fixed issues with the `copysign` function with the latest MSVC ([#166](https://github.com/auto-differentiation/xad/pull/166) by @raneamri)
- **Improved testing of chunk containers**: The clear is now tested better ([#157](https://github.com/auto-differentiation/xad/pull/157), by @rghouzra)


## [1.7.0] - 2024-11-30

This release features extensive performance improvements.

### Added

-   **Template Variables**: Added support for C++17 type traits such as `std::is_floating_point_v` ([#127](https://github.com/auto-differentiation/xad/pull/127)).
-   **CI/CD Workflows**: Updated workflows to target the C++17 standard ([#127](https://github.com/auto-differentiation/xad/pull/127)).
-   **New Sample**: Introduced a Monte-Carlo swaption portfolio pricer, including path-wise derivative calculations ([#126](https://github.com/auto-differentiation/xad/pull/126)).

### Changed

-   **Performance Improvements** ([#150](https://github.com/auto-differentiation/xad/pull/150)):

    -   Optimised `OperationsContainer` for handling slots and multipliers.
    -   Enhanced iteration efficiency in `computeAdjoints` by avoiding redundant operations.
    -   Implemented joint tape appending for multipliers and slots in a single operation.
    -   Removed the overhead of `std::fma` calls, relying on compiler-level optimisations.
    -   Added pre-checks to skip derivative calculations when tape is not required.
    -   Provided branch prediction hints using `XAD_LIKELY` and `XAD_UNLIKELY`.
    -   Introduced a paired operations container variant for improved performance at a slight memory cost, with a `XAD_REDUCED_MEMORY` CMake option to toggle memory usage.

-   **Documentation Updates**: Improved and expanded documentation ([#125](https://github.com/auto-differentiation/xad/pull/125)).
-   **CMake Module Renaming**: Renamed helper modules to avoid name clashes (by @raneamri, [#123](https://github.com/auto-differentiation/xad/pull/123)).

### Fixed

-   **Move Behaviour of ChunkContainer** Fixed move behaviour of `ChunkContainer` (by @rghouzra, [#152](https://github.com/auto-differentiation/xad/pull/152))


## [1.6.0] - 2024-07-17

This release mainly adds support for more architectures and compilers and provides higher level derivative functions as well as examples.

### Added

- Support for Mac M1+ architecture (ARM) as well as AppleClang 15 support (by @raneamri [#116](https://github.com/auto-differentiation/xad/pull/116))
- High level functions to compute Jacobian and Hessian matrices (by @raneamri [#117](https://github.com/auto-differentiation/xad/pull/117))

### Removed

- Moved website to its own repository and keeping only the reference manual [#112](https://github.com/auto-differentiation/xad/pull/112)

## [1.5.2] - 2024-04-04

This patch release is for matching versions with
[xad-py](https://github.com/auto-differentiation/xad-py).

### Changed

- Moved Python bindings into its [own repository](https://github.com/auto-differentiation/xad-py)
  (new Python package name is [xad](https://pypi.org/project/xad))
- Reorganised website
- Upgraded CI/CD actions

## [1.5.1] - 2024-03-28

This is a patch release to allow interoperability with the QuantLib-Risks
Python package.

### Added

-   added static functions to `Tape` to activate and deactivate a specific
    tape instance


## [1.5.0] - 2024-03-25

### Added

-   Python bindings as [xad-autodiff](https://pypi.org/project/xad-autodiff/)
-   Added `std::is_signed` trait to `StdCompatibility.hpp` header for consistency
-   Support for enhanced debugger visualisations in Visual Studio (@dholden3)

### Changed

-   Improved documentation for QuantLib-Risks build
-   Cleaned up output of Swap Pricer example

## [1.4.1] - 2024-01-10

This is a patch release to ensure compatibility with QuantLib 1.33.

### Added

-   support for `hypot` math function

### Fixed

-   avoid overflow with complex `abs` function when real / imaginary parts are large

## [1.4.0] - 2024-01-09

### Added

-   Supporting Clang version 16 and added to CI/CD
-   Adding complex arithmetics between complex XAD types and `std::complex<double>`

### Fixed

-   Documentation updates

## [1.3.0] - 2023-08-16

### Added

-   Improved CI/CD workflows with better caching and latest compilers

### Changed

-   Documentation updates

### Fixed

-   Fixed missing include of `<memory>` in `ChunkContainer.hpp`

## [1.2.0] - 2023-05-24

### Added

-   More CI/CD workflows for all supported compiler versions
-   Added math function `copysign`

### Changed

-   Revamped documentation site using mkdocs
-   Improved tests and testing infrastructure

### Fixed

-   Throw exception when no tape is set on `derivative` calls
-   Some test errors with GCC versions not previously tested

## [1.1.0] - 2022-11-17

### Added

-   QuantLib integration by means of the
    [QuantLib-Risks](https://github.com/auto-differentiation/QuantLib-Risks-Cpp)
    integration module
-   Full MacOS support
-   Better CI pipeline with more platforms and compilers tested
-   Code coverage and quality measured on pull requests and reported
    in [README.md](README.md)
-   More tests to improve code coverage
-   Status badges in [README.md](README.md)
-   Documentation updates

### Changed

-   Code quality improvements
-   Better use of caching in CI/CD pipelines for faster builds

## [1.0.0] - 2022-07-07

Initial open-source release

[unreleased]: https://github.com/auto-differentiation/xad/compare/v1.8.0...HEAD

[1.8.0]: https://github.com/auto-differentiation/xad/compare/v1.7.0...v1.8.0

[1.7.0]: https://github.com/auto-differentiation/xad/compare/v1.6.0...v1.7.0

[1.6.0]: https://github.com/auto-differentiation/xad/compare/v1.5.2...v1.6.0

[1.5.2]: https://github.com/auto-differentiation/xad/compare/v1.5.1...v1.5.2

[1.5.1]: https://github.com/auto-differentiation/xad/compare/v1.5.0...v1.5.1

[1.5.0]: https://github.com/auto-differentiation/xad/compare/v1.4.1...v1.5.0

[1.4.1]: https://github.com/auto-differentiation/xad/compare/v1.4.0...v1.4.1

[1.4.0]: https://github.com/auto-differentiation/xad/compare/v1.3.0...v1.4.0

[1.3.0]: https://github.com/auto-differentiation/xad/compare/v1.2.0...v1.3.0

[1.2.0]: https://github.com/auto-differentiation/xad/compare/v1.1.0...v1.2.0

[1.1.0]: https://github.com/auto-differentiation/xad/compare/v1.0.0...v1.1.0

[1.0.0]: https://github.com/auto-differentiation/xad/releases/tag/v1.0.0

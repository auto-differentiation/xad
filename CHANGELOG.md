# Changelog

All notable changes to XAD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.5.2]: https://github.com/auto-differentiation/xad/compare/v1.5.1...v1.5.2

[1.5.1]: https://github.com/auto-differentiation/xad/compare/v1.5.0...v1.5.1

[1.5.0]: https://github.com/auto-differentiation/xad/compare/v1.4.1...v1.5.0

[1.4.1]: https://github.com/auto-differentiation/xad/compare/v1.4.0...v1.4.1

[1.4.0]: https://github.com/auto-differentiation/xad/compare/v1.3.0...v1.4.0

[1.3.0]: https://github.com/auto-differentiation/xad/compare/v1.2.0...v1.3.0

[1.2.0]: https://github.com/auto-differentiation/xad/compare/v1.1.0...v1.2.0

[1.1.0]: https://github.com/auto-differentiation/xad/compare/v1.0.0...v1.1.0

[1.0.0]: https://github.com/auto-differentiation/xad/releases/tag/v1.0.0

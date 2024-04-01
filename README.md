<p align="center" dir="auto">
    <a href="https://auto-differentiation.github.io" rel="nofollow" target="_blank">
        <img src="https://github.com/auto-differentiation/XAD/blob/main/docs/images/logo.svg?raw=true" height="80" alt="XAD" style="max-width:100%">
    </a>
</p>

XAD is a comprehensive library for automatic differentiation, available for both Python and  C++.
It targets production-quality code at any scale, striving for both ease of use and high performance.

<p align="center" dir="auto">
    <a href="https://github.com/auto-differentiation/XAD/releases/latest">
        <img src="https://img.shields.io/github/v/release/auto-differentiation/XAD?label=Download&sort=semver" alt="Download" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/XAD/blob/main/LICENSE.md">
        <img src="https://img.shields.io/github/license/auto-differentiation/XAD?label=License" alt="License" style="max-width: 100%;">
    </a>
    <a href="https://doi.org/10.5281/zenodo.10867823">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10867823.svg" alt="DOI" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/XAD/blob/main/CONTRIBUTING.md">
        <img src="https://img.shields.io/badge/PRs%20-welcome-brightgreen.svg" alt="PRs Welcome" style="max-width: 100%;">
    </a>
    <br>
    <a href="https://github.com/auto-differentiation/XAD/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/auto-differentiation/XAD/ci.yml?label=Build&logo" alt="GitHub Workflow Status" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/XAD/actions?query=workflow%3ACI">
        <img src="https://gist.githubusercontent.com/auto-differentiation-dev/e0eab86863fac6da5e44d20df215f836/raw/badge.svg" alt="Tests Badge" style="max-width: 100%;">
    </a>
    <a href="https://coveralls.io/github/auto-differentiation/XAD?branch=main">
        <img src="https://coveralls.io/repos/github/auto-differentiation/XAD/badge.svg?branch=main" alt="Coverage" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/XAD/actions?query=workflow%3ACodeQL++">
        <img src="https://github.com/auto-differentiation/XAD/actions/workflows/codeql-analysis.yml/badge.svg" alt="GitHub CodeQL Status" style="max-width: 100%;" >
    </a>
    <a href="https://www.codacy.com/gh/auto-differentiation/XAD/dashboard">
        <img src="https://img.shields.io/codacy/grade/1826d0a6c8ce4feb81ef3b482d65c7b4?logo=codacy&label=Quality%20%28Codacy%29" alt="Tests Badge" style="max-width: 100%;">
    </a>
</p>

Automatic differentiation (also called algorithmic differentiation) is a set of techniques for calculating partial derivatives
of functions specified as computer programs. 
Since every program execution is always composed of a sequence of simple operations with known derivatives (arithmetics and mathematical functions like sin, exp, log, etc.),
the chain rule can be applied repeatedly to calculate partial derivatives automatically.
XAD implements this using operator-overloading in C++ and exposes bindings for Python,
allowing to compute derivatives with minimal changes to the program.
See [automatic differentation mathematical background](https://auto-differentiation.github.io/aad/) for more details.

Application areas:

-   Machine Learning and Deep Learning: Training neural networks or other 
    machine learning models.
-   Optimization: Solving optimization problems in engineering and finance.
-   Numerical Analysis: Enhancing numerical solution methods for differential
    equations.
-   Scientific Computing: Simulating physical systems and processes.
-   Risk Management and Quantitative Finance: Assessing and hedging risk in
    financial models.
-   Computer Graphics: Optimizing rendering algorithms.
-   Robotics: Improving control and simulation of robotic systems.
-   Meteorology: Enhancing weather prediction models.
-   Biotechnology: Modeling biological processes and systems.

Key features:

-   Forward and adjoint mode for any order, using operator-overloading
-   Checkpointing support (for tape memory management)
-   External functions interface (to integrate external libraries)
-   Thread-safe tape
-   Formal exception-safety guarantees
-   High performance
-   Battle-tested in large production code bases

## Getting Started
### Python
XAD in Pyhon comes as a PyPi package for all major platforms and operating systems. 

The XAD Python bindings can be installed as usual using pip or any other package manager:
```
pip install xad-autodiff
```
Documentation on usage can be found [here](https://auto-differentiation.github.io/python/#usage).

An example integration with QuantLib, the open source library for quantitative finance, can be found [here](https://auto-differentiation.github.io/quantlib/python/).


### C++
XAD in C++ builds with modern CMake and has no external dependencies. 
For instructions how to build and integrate it into your projects, please refer to the
[Installation Guide](https://auto-differentiation.github.io/installation/).

The documentation site also contains [tutorials](https://auto-differentiation.github.io/tutorials/), 
[examples](https://auto-differentiation.github.io/examples/), 
and information about [integrating XAD into QuantLib](https://auto-differentiation.github.io/quantlib/cxx/).

The sources for the site are located in the [docs](docs) directory in this repository.

## Getting Help

If you have found an issue, want to report a bug, or have a feature request, please raise a [GitHub issue](https://github.com/auto-differentiation/XAD/issues).

For general questions about XAD, sharing ideas, engaging with community members, etc, please use [GitHub Discussions](https://github.com/auto-differentiation/XAD/discussions).

## Planned Features

Please see the [issues list](https://github.com/auto-differentiation/XAD/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement) for planned/open features.
You are very welcome to contribute towards these (or other) features - please contact the project maintainers before to discuss you proposal.
If you have new feature requests, please submit a new issue with a detailed description.

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for the process of contributing to this project.
Please also obey our [Code of Conduct](CODE_OF_CONDUCT.md) in all communication.

## Versioning

We use [SemVer](http://semver.org/) for versioning,
making a new release available as soon as sufficient new features have been merged into `main`.
The final decision about a release and which features are included is with the project maintainers.
For the versions available, see the [releases in GitHub](https://github.com/auto-differentiation/XAD/releases).

Every new version is also added to the [Changelog](CHANGELOG.md),
which needs to be maintained throughout the development.
That is, every pull request should also update the Changelog accordingly.

## Authors

-   Various contributors from Xcelerit
-   See also the list of [contributors](https://github.com/auto-differentiation/XAD/contributors) who participated in the project.


## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE.md](LICENSE.md) file for details.

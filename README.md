<p align="center">
  <a href="https://auto-differentiation.github.io" target="_blank">
    <img src="https://auto-differentiation.github.io/images/logo.svg" height="80" alt="XAD">
  </a>
</p>

# XAD: The fastest automatic differentiation library for C++

Exact derivatives of your existing C++ code, in forward, adjoint and higher-order mode,
with low runtime overhead and a small memory footprint. Python bindings: `pip install xad`.

For Monte Carlo and other repetitive workloads, XAD also provides a JIT backend interface
for record-once / replay-many execution.

[Documentation](https://auto-differentiation.github.io/) ·
[Install](#install) ·
[Samples](samples) ·
[Benchmarks](https://github.com/auto-differentiation/ad-benchmarks) ·
[Python](https://github.com/auto-differentiation/xad-py) ·
[Discussions](https://github.com/auto-differentiation/xad/discussions)

<p align="center" dir="auto">
    <a href="https://github.com/auto-differentiation/xad/releases/latest">
        <img src="https://img.shields.io/github/v/release/auto-differentiation/xad?label=Download&sort=semver" alt="Download" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/xad/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/auto-differentiation/xad/ci.yml?label=Build&logo" alt="Build Status" style="max-width: 100%;">
    </a>
    <a href="https://coveralls.io/github/auto-differentiation/xad?branch=main">
        <img src="https://coveralls.io/repos/github/auto-differentiation/xad/badge.svg?branch=main&v=2" alt="Coverage" style="max-width: 100%;">
    </a>
    <a href="https://pypi.org/project/xad/">
        <img src="https://img.shields.io/pypi/v/xad?label=PyPI" alt="PyPI" style="max-width: 100%;">
    </a>
    <a href="https://doi.org/10.5281/zenodo.7965850">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7965850.svg" alt="DOI" style="max-width: 100%;">
    </a>
    <a href="#license">
        <img src="https://img.shields.io/badge/license-AGPL--3.0%20%7C%20commercial-blue" alt="License: AGPL-3.0 or commercial" style="max-width: 100%;">
    </a>
</p>

## Example

Every derivative of a function in one adjoint sweep:

```c++
#include <XAD/XAD.hpp>
#include <iostream>

using mode = xad::adj<double>;   // adjoint (reverse) mode
using AD = mode::active_type;

template <class T>
T f(T x, T y) { return x * sin(y) + exp(x * y); }   // your code, unchanged

int main() {
    mode::tape_type tape;
    AD x = 1.3, y = 5.2;
    tape.registerInput(x);
    tape.registerInput(y);
    tape.newRecording();
    AD z = f(x, y);
    tape.registerOutput(z);
    derivative(z) = 1.0;
    tape.computeAdjoints();       // every input derivative in one sweep
    std::cout << "dz/dx = " << derivative(x) << ", dz/dy = " << derivative(y) << "\n";
}
```

## Install

With CMake, straight from GitHub:

```cmake
include(FetchContent)
FetchContent_Declare(XAD GIT_REPOSITORY https://github.com/auto-differentiation/xad.git GIT_TAG v2.1.0)
FetchContent_MakeAvailable(XAD)
target_link_libraries(my_app PRIVATE XAD::xad)
```

Or build from source:

```bash
git clone https://github.com/auto-differentiation/xad.git
cmake -S xad -B xad/build
cmake --build xad/build
```

Python: `pip install xad`. More in the [Installation Guide](https://auto-differentiation.github.io/installation/cxx/)
and the [Tutorials](https://auto-differentiation.github.io/tutorials/).

## Benchmarks

Compared against [CppAD](https://github.com/coin-or/CppAD), [Adept 2](https://github.com/rjhogan/Adept-2), [autodiff](https://github.com/autodiff/autodiff), and finite differences on four quant-finance workloads (Intel Xeon Platinum 8488C, GCC 13.3, `-O3 -mavx2 -mfma`):

<p align="center">
  <a href="https://github.com/auto-differentiation/ad-benchmarks">
    <img src="https://raw.githubusercontent.com/auto-differentiation/ad-benchmarks/main/results/chart.svg" alt="Benchmark chart" width="100%">
  </a>
</p>

Full methodology, source code, and reproducible CSV: **[auto-differentiation/ad-benchmarks](https://github.com/auto-differentiation/ad-benchmarks)**

## Key Features

- **Forward & Reverse (Adjoint) Mode**: Supports any order using operator overloading.
- **Vector mode**: Compute multiple derivatives at once.
- **Checkpointing Support**: Efficient tape memory management for large-scale applications.
- **External Function Interface**: Seamlessly connect with external libraries.
- **Eigen support**: Works with the popular linear algebra library [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
- **JIT Backend Support** *(optional)*: Infrastructure for pluggable JIT backends, enabling record-once/replay-many
  workflows. See [samples/jit_tutorial](samples/jit_tutorial).

## Ecosystem

| Repository | Description |
|---|---|
| [xad-py](https://github.com/auto-differentiation/xad-py) | Python bindings for XAD |
| [QuantLibAAD](https://github.com/auto-differentiation/QuantLibAAD) | Full QuantLib integration — compute all Greeks at once, up to 3 orders of magnitude faster than bump-and-reval |
| [QuantLib-Risks-Py](https://github.com/auto-differentiation/QuantLib-Risks-Py) | QuantLib risks from Python |
| [xad-codegen](https://www.xcelerit.com/xad-enterprise-support) | Native code generation backend — maximum throughput (commercial) |
| [AAD Training](https://www.xcelerit.com/solutions/training-aad) | Hands-on AAD training for quants and quant developers - delivered to dozens of tier 1 banks and financial services firms |

## Documentation

Full documentation, including API reference and usage examples, is available at:
[**https://auto-differentiation.github.io/**](https://auto-differentiation.github.io/)

## Contributing

Contributions are welcome. Please see the
[**Contributing Guide**](CONTRIBUTING.md) for details, and feel free to start a
discussion in our
[**GitHub Discussions**](https://github.com/auto-differentiation/xad/discussions).

## Found a Bug?

Please report bugs and issues via the
[**GitHub Issue Tracker**](https://github.com/auto-differentiation/xad/issues).

## License

XAD is free under the [AGPL-3.0](LICENSE.md) for evaluation, research and open-source use.
For production use without AGPL obligations, a [commercial subscription](https://www.xcelerit.com/xad-enterprise-support) is available.

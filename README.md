<p align="center">
  <a href="https://www.xcelerit.com/xad/docs/" target="_blank">
    <img src="https://www.xcelerit.com/img/docs/xad-logo.svg" height="80" alt="XAD">
  </a>
</p>

# XAD: The fastest automatic differentiation library for C++

Exact derivatives of your existing C++ code, in forward, adjoint and higher-order mode,
with low runtime overhead and a small memory footprint. Python bindings: `pip install xad`.

For Monte Carlo and other repetitive workloads, XAD also provides a JIT backend interface
for record-once / replay-many execution.

[Documentation](https://www.xcelerit.com/xad/docs/) ·
[Install](#install) ·
[Samples](samples) ·
[Benchmarks](https://github.com/auto-differentiation/ad-benchmarks) ·
[Python](https://github.com/auto-differentiation/xad-py) ·
[Discussions](https://github.com/auto-differentiation/xad/discussions)

<p align="center" dir="auto">
    <a href="https://github.com/auto-differentiation/xad/releases/latest"><img src="https://img.shields.io/github/v/release/auto-differentiation/xad?label=Download&sort=semver" alt="Download" style="max-width: 100%;"></a>
    <a href="https://github.com/auto-differentiation/xad/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/auto-differentiation/xad/ci.yml?label=Build&logo" alt="Build Status" style="max-width: 100%;"></a>
    <a href="https://coveralls.io/github/auto-differentiation/xad?branch=main"><img src="https://coveralls.io/repos/github/auto-differentiation/xad/badge.svg?branch=main&v=2" alt="Coverage" style="max-width: 100%;"></a>
    <a href="https://pypi.org/project/xad/"><img src="https://img.shields.io/pypi/v/xad?label=PyPI" alt="PyPI" style="max-width: 100%;"></a>
    <a href="https://github.com/microsoft/vcpkg/tree/master/ports/xad"><img src="https://img.shields.io/vcpkg/v/xad?label=vcpkg" alt="vcpkg" style="max-width: 100%;"></a>
    <a href="https://doi.org/10.5281/zenodo.7965850"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7965850.svg" alt="DOI" style="max-width: 100%;"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-AGPL--3.0%20%7C%20commercial-blue" alt="License: AGPL-3.0 or commercial" style="max-width: 100%;"></a>
</p>

> [!IMPORTANT]
> **Using XAD in your organization?** XAD is open source under the AGPL, to try it, for research and for open-source projects. The AGPL is not suitable for enterprise settings: for use in your organization you need a separate commercial license from Xcelerit, by annual XAD subscription. See [XAD licensing](https://www.xcelerit.com/xad/licensing/) for the terms, a quote within one business day, or a free, confidential license check.

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

With [vcpkg](https://github.com/microsoft/vcpkg/tree/master/ports/xad):

```bash
vcpkg install xad
```

```cmake
find_package(XAD CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE XAD::xad)
```

With CMake, straight from GitHub:

```cmake
include(FetchContent)
FetchContent_Declare(XAD GIT_REPOSITORY https://github.com/auto-differentiation/xad.git GIT_TAG v2.2.0)
FetchContent_MakeAvailable(XAD)
target_link_libraries(my_app PRIVATE XAD::xad)
```

Or build from source:

```bash
git clone https://github.com/auto-differentiation/xad.git
cmake -S xad -B xad/build
cmake --build xad/build
```

The defaults suit any code; [Build Options](docs/ref/build-options.md) lists the options that make XAD faster for some kinds of code.

Python: `pip install xad`. More in the [Installation Guide](https://www.xcelerit.com/xad/docs/installation/cxx/)
and the [Tutorials](https://www.xcelerit.com/xad/docs/tutorials/aad/).

## Benchmarks

### Quant-finance workloads

Compared against [CppAD](https://github.com/coin-or/CppAD), [Adept 2](https://github.com/rjhogan/Adept-2) and [autodiff](https://github.com/autodiff/autodiff) on four quant-finance workloads (LMM, Heston, CVA, SABR), with every gradient checked. XAD computes all sensitivities 6.5x to 83x faster than bump-and-reval and is the fastest of the open-source libraries on all four.

<p align="center">
  <a href="https://github.com/auto-differentiation/ad-benchmarks">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/auto-differentiation/ad-benchmarks/main/results/amd/speedup-dark.svg">
      <img src="https://raw.githubusercontent.com/auto-differentiation/ad-benchmarks/main/results/amd/speedup-light.svg" alt="Speed-up over bump-and-reval per workload for XAD, XAD + codegen, Adept 2, CppAD and autodiff" width="600">
    </picture>
  </a>
</p>

XAD v2.2.0, AMD EPYC (Zen 5), one core, GCC 15.2, run 10 November 2026; ad-benchmarks [v1.0.0](https://github.com/auto-differentiation/ad-benchmarks/tree/v1.0.0). Full methodology, source code and [results](https://github.com/auto-differentiation/ad-benchmarks#results): **[auto-differentiation/ad-benchmarks](https://github.com/auto-differentiation/ad-benchmarks)**

### GradBench

GradBench is an independent suite of 11 general AD problems run in the same harness for every tool.

With code generation ([xad-codegen](https://www.xcelerit.com/xad/licensing/), commercial), XAD is the fastest of the C and C++ AD tools on all 11 [GradBench](https://gradben.ch) problems,
and on average 2.4x faster than the next-fastest tool (geometric mean of the ratios to the fastest other AD tool, open-source XAD excluded).
On seven of the nine problems with hand-written derivatives, the generated code beats them.
Without code generation, open-source XAD is the fastest tape-based AD tool on 7 of the 11 and second on the other four.
Measured by Xcelerit with GradBench's harness and tool images, 16 Nov 2026: [GradBench results](https://www.xcelerit.com/xad/benchmarks/#bench-gradbench).

## Key Features

- **Forward & Reverse (Adjoint) Mode**: Supports any order using operator overloading.
- **Vector mode**: Compute multiple derivatives at once.
- **Checkpointing Support**: Efficient tape memory management for large-scale applications.
- **External Function Interface**: Seamlessly connect with external libraries.
- **Eigen support**: Works with the popular linear algebra library [Eigen](https://gitlab.com/libeigen/eigen).
- **JIT Backend Support** *(optional)*: Infrastructure for pluggable JIT backends, enabling record-once/replay-many
  workflows. See [samples/jit_tutorial](samples/jit_tutorial).

## Ecosystem

| Repository | Description |
|---|---|
| [xad-py](https://github.com/auto-differentiation/xad-py) | Python bindings for XAD |
| [QuantLibAAD](https://github.com/auto-differentiation/QuantLibAAD) | Full QuantLib integration — compute all Greeks at once, up to 3 orders of magnitude faster than bump-and-reval |
| [QuantLibAAD-Py](https://github.com/auto-differentiation/QuantLibAAD-Py) | QuantLibAAD from Python |
| [xad-codegen](https://www.xcelerit.com/xad/licensing/) | Native code generation backend — maximum throughput (commercial) |
| [AAD Training](https://www.xcelerit.com/training/) | Hands-on AAD training for quants and quant developers - delivered to dozens of tier 1 banks and financial services firms |

## Documentation

Full documentation, including API reference and usage examples, is available at:
[**https://www.xcelerit.com/xad/docs/**](https://www.xcelerit.com/xad/docs/)

## Contributing

Contributions are welcome. Please see the
[**Contributing Guide**](CONTRIBUTING.md) for details, and feel free to start a
discussion in our
[**GitHub Discussions**](https://github.com/auto-differentiation/xad/discussions).

If XAD is useful to you, a star on GitHub helps others find it.

New here? Pick a [good first issue](https://github.com/auto-differentiation/xad/labels/good%20first%20issue). Each one is small, self-contained and comes with a clear task.

Thanks to everyone who has contributed:

[![Contributors](https://contrib.rocks/image?repo=auto-differentiation/xad)](https://github.com/auto-differentiation/xad/graphs/contributors)

## Found a Bug?

Please report bugs and issues via the
[**GitHub Issue Tracker**](https://github.com/auto-differentiation/xad/issues).

## License

XAD is open source under the [AGPL-3.0](LICENSE.md), to try it, for research and for open-source projects.
For use in your organization you need a separate commercial license from Xcelerit: see [XAD licensing](https://www.xcelerit.com/xad/licensing/).

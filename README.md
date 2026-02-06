<p align="center">
  <a href="https://auto-differentiation.github.io" target="_blank">
    <img src="https://auto-differentiation.github.io/images/logo.svg" height="80" alt="XAD">
  </a>
</p>

# XAD: Fast, easy automatic differentiation in C++

XAD is a high-performance C++ automatic differentiation library designed for large-scale, performance-critical systems.

It provides forward and adjoint (reverse) mode automatic differentiation via operator overloading, with a strong focus on:

* Low runtime overhead
* Minimal memory footprint
* Straightforward integration into existing C++ codebases

For Monte Carlo and other repetitive workloads, XAD also offers optional JIT backend support,
enabling record-once / replay-many execution for additional performance boost.

<p align="center" dir="auto">
    <a href="https://github.com/auto-differentiation/xad/releases/latest">
        <img src="https://img.shields.io/github/v/release/auto-differentiation/xad?label=Download&sort=semver" alt="Download" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/xad/blob/main/CONTRIBUTING.md">
        <img src="https://img.shields.io/badge/PRs%20-welcome-brightgreen.svg" alt="PRs Welcome" style="max-width: 100%;">
    </a>
    <a href="https://github.com/auto-differentiation/xad/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/auto-differentiation/xad/ci.yml?label=Build&logo" alt="Build Status" style="max-width: 100%;">
    </a>
    <a href="https://coveralls.io/github/auto-differentiation/xad?branch=main">
        <img src="https://coveralls.io/repos/github/auto-differentiation/xad/badge.svg?branch=main" alt="Coverage" style="max-width: 100%;">
    </a>
    <a href="https://www.codacy.com/gh/auto-differentiation/xad/dashboard">
        <img src="https://img.shields.io/codacy/grade/1826d0a6c8ce4feb81ef3b482d65c7b4?logo=codacy&label=Quality%20%28Codacy%29" alt="Codacy Quality" style="max-width: 100%;">
    </a>
</p>


## Key Features

- **Forward & Reverse (Adjoint) Mode**: Supports any order using operator overloading.
- **Vector mode**: Compute multiple derivatives at once.
- **Checkpointing Support**: Efficient tape memory management for large-scale applications.
- **External Function Interface**: Seamlessly connect with external libraries.
- **Eigen support**: Works with the popular linear algebra library [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
- **JIT Backend Support** *(optional)*: Infrastructure for pluggable JIT backends, enabling record-once/replay-many.
  workflows - with or without automatic differentiation. See [samples/jit_tutorial](samples/jit_tutorial).

## Example

Calculate first-order derivatives of an arbitrary function with two inputs and one output using XAD in adjoint mode.

```c++
Adouble x0 = 1.3;              // initialise inputs
Adouble x1 = 5.2;
tape.registerInput(x0);        // register independent variables
tape.registerInput(x1);        // with the tape
tape.newRecording();           // start recording derivatives
Adouble y = func(x0, x1);      // run main function
tape.registerOutput(y);        // register the output variable
derivative(y) = 1.0;           // seed output adjoint to 1.0
tape.computeAdjoints();        // roll back adjoints to inputs
cout << "dy/dx0=" << derivative(x0) << "\n"
     << "dy/dx1=" << derivative(x1) << "\n";
```

## Getting Started

Build XAD from source using CMake:

```bash
git clone https://github.com/auto-differentiation/xad.git
cd xad
mkdir build
cd build
cmake ..
make
```

For more detailed guides,
refer to our [**Installation Guide**](https://auto-differentiation.github.io/installation/cxx/)
and explore [**Tutorials**](https://auto-differentiation.github.io/tutorials/).

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

## Related Projects

- [XAD-Py](https://github.com/auto-differentiation/xad-py): XAD in Python.
- [QuantLibAAD](https://github.com/auto-differentiation/QuantLibAAD): AAD integration in [QuantLib](https://github.com/lballabio/QuantLib).
- [xad-forge](https://github.com/da-roth/xad-forge): Forge JIT backends for XAD.

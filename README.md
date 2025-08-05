<p align="center">
  <a href="https://auto-differentiation.github.io" target="_blank">
    <img src="https://auto-differentiation.github.io/images/logo.svg" height="80" alt="XAD">
  </a>
</p>

# 🚀 XAD: Powerful Automatic Differentiation for C++ & Python

**XAD** is the ultimate solution for automatic differentiation, combining **ease of use** with **high performance**.
It's designed to help you differentiate complex applications with speed and precision—whether
you're optimizing neural networks, solving scientific problems, or performing financial risk analysis.

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

## 🌟 Why XAD?

XAD is trusted by professionals for its **speed**, **flexibility**, and **scalability** across various fields:

- **Machine Learning & Deep Learning**: Accelerate neural network training and model optimization.
- **Optimization in Engineering & Finance**: Solve complex problems with high precision.
- **Numerical Analysis**: Improve methods for solving differential equations efficiently.
- **Scientific Computing**: Simulate physical systems and processes with precision.
- **Risk Management & Quantitative Finance**: Assess and hedge risks in sophisticated financial models.
- **Computer Graphics**: Optimize rendering algorithms for high-quality graphics.
- **Robotics**: Enhance control and simulation for robotic systems.
- **Meteorology**: Improve accuracy in weather prediction models.
- **Biotechnology**: Model complex biological processes effectively.

### Key Features

- **Forward & Adjoint Mode**: Supports any order using operator overloading.
- **Checkpointing Support**: Efficient tape memory management for large-scale applications.
- **External Function Interface**: Seamlessly connect with external libraries.
- **Thread-Safe Tape**: Ensure safe, concurrent operations.
- **Exception-Safe**: Formal guarantees for stability and error handling.
- **High Performance**: Optimized for speed and efficiency.
- **Proven in Production**: Battle-tested in large-scale, mission-critical systems.

## 💻 Example

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

## 🚀 Getting Started

```bash
git clone https://github.com/auto-differentiation/xad.git
cd xad
mkdir build
cd build
cmake ..
make
```

For more detailed guides,
refer to our [**Installation Guide**](https://auto-differentiation.github.io/installation/)
and explore [**Tutorials**](https://auto-differentiation.github.io/tutorials/).

## 🤝 Contributing

Want to get involved? We welcome contributors from all backgrounds! Check out
our [**Contributing Guide**](CONTRIBUTING.md) and join the conversation in our
[**Discussions**](https://github.com/auto-differentiation/xad/discussions).

## 🐛 Found a Bug?

Please report any issues through our
[**Issue Tracker**](https://github.com/auto-differentiation/xad/issues).

---

## 📦 Related Projects

- [XAD-Py](https://github.com/auto-differentiation/xad-py): XAD in Python.
- [QuantLib-Risks](https://github.com/auto-differentiation/QuantLib-Risks-Cpp): Fast
  risk evaluations in C++ and Python.

---

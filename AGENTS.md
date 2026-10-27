# AGENTS.md

Guidance for coding agents working on XAD. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md), which this file
summarises and which takes precedence.

## What XAD is

A C++ library for automatic differentiation: forward and adjoint (reverse) mode, higher orders, vector modes,
checkpointing and a JIT interface. Most of it is header code in `src/XAD/`; it must build as C++11.

## Layout

- `src/XAD/`: public headers and sources
- `test/`: unit tests (GoogleTest)
- `samples/`: example programs, each registered as a test `sample_<name>`
- `cmake/`: build modules; `doc/`: Sphinx manual

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -j8 --output-on-failure
```

- Run one test or sample: `ctest --test-dir build -R <name>`.
- Useful options (see `cmake/XADSetupOptions.cmake`): `XAD_ENABLE_TESTS`, `XAD_TAPE_REUSE_SLOTS`,
  `XAD_REDUCED_MEMORY`, `XAD_USE_STRONG_INLINE`, `XAD_ENABLE_ADDRESS_SANITIZER`.
- Before opening a PR, build and run the tests in both C++11 (`-DCMAKE_CXX_STANDARD=11`) and C++17; CI also
  covers several GCC, Clang, MSVC and macOS versions.

## Rules for changes

- Keep a pull request to one change; don't reformat or refactor unrelated code.
- Add or extend tests in `test/` for any behaviour you add or fix, and make sure all tests pass.
- Format with the repository's `.clang-format`; lines at most 100 characters.
- Follow the naming rules in CONTRIBUTING.md (`MixedCase` types, `mixedCase` functions and variables,
  `UPPER_CASE` constants, `member_` for private members, `#pragma once`).
- Every new source file starts with the licence header used by the existing files, plus a short note on
  its purpose.
- Add a line to `CHANGELOG.md` under `[unreleased]` for user-visible changes.
- Don't change public APIs or default behaviour without an issue or discussion first.
- Commit messages: a short one-line summary, a blank line, then details if needed.
- Contributors must sign the Contributor License Agreement before a pull request can be merged.

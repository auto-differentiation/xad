---
title: "Supported Platforms & CI Tests"
description: "Explore XAD's tested platforms across Windows, Ubuntu, MacOS with extensive CI workflow integration. Contribute to broader coverage."
hide:
  - toc
---

# Tested Platforms

The following platforms are part of the [continuous integration workflow][ci], i.e. they are tested on each commit. You can use other configurations at your own risk,
or [submit a PR](https://github.com/auto-differentiation/xad/blob/feature/new-site/CONTRIBUTING.md) to include it in the [CI workflow][ci].

Note that all tested platforms are 64bit Intel/AMD.


| Operating System    | Compiler                          | Configurations                                      | Test Coverage Recorded |
| ------------------- | --------------------------------- | --------------------------------------------------- | ---------------------- |
| Windows Server 2019 | Visual Studio 2015 (Toolset 14.0) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2017 (Toolset 14.1) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2019 (Toolset 14.2) | Debug, Release                                      | no                     |
| Windows Server 2022 | Visual Studio 2022 (Toolset 14.3) | Debug, Release                                      | no                     |
| Windows Server 2022 | Clang 16.0 (Toolset 14.3)         | Debug, Release                                      | no                     |
| Ubuntu 16.04        | GCC 5.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 17.10        | GCC 6.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 17.10        | GCC 7.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 18.04        | GCC 8.4.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 19.10        | GCC 9.2.0                         | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 20.04        | GCC 10.3.0                        | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 20.04        | GCC 11.1.0                        | Debug, Release, + both with `XAD_TAPE_REUSE_SLOTS`  | yes                    |
| Ubuntu 20.04        | Clang 11.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 12.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 13.0.1                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 14.0.0                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| Ubuntu 22.04        | Clang 15.0.7                      | Debug, Release, Release with `XAD_TAPE_REUSE_SLOTS` | no                     |
| MacOS 12.6.5        | AppleClang 14.0.0                 | Debug, Release                                      | yes                    |


[ci]: https://github.com/auto-differentiation/xad/blob/main/.github/workflows/ci.yml

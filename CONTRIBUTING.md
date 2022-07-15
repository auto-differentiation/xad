# How to contribute

We love pull requests from everyone. By participating in this project, you agree to abide
by our [code of conduct](CODE_OF_CONDUCT.md).

1.  Fork, then clone the repository:

```bash
git clone https://github.com/yourusername/xad.git
```

2.  Follow the [Build Instructions](README.md) to setup the dependencies and 
    build the software. Make sure all tests pass.

3.  Create a feature branch, typically based on master, for your change

4.  Create a feature branch, typically based on master, for your change

```bash
git checkout -b feature/my-change main
```

5.  Make your changes, adding tests as you go, and commit. Again, make sure all 
    tests pass.

6.  Push your fork 

7.  [Submit a pull request][pr]. Not that you will have to sign the [Contributer License Agreement][cla] 
    before the PR can be merged.

At this point, you are depending on the core team to review your request. 
We may suggest changes, improvements, or alternatives. 
We strive to at least comment on a pull request within 3 business days. 
After feedback has been given, we expect a response within 2 weeks, 
after which we may close the pull request if it isn't showing activity.

Some things that will highly increase the chance that your pull request gets
accepted:

-   Discuss the change you wish to make via issue or email

-   Write good tests for all added features

-   Follow our [coding style](#coding-style)

-   Write good commit messages (short one-liner, followed by a blank line, 
    followed by a more detailed explanation)

## Source Code Organisation

-   [cmake](cmake): CMake modules and scripts used for the build
-   [doc](doc): Sources to generate the user documentation (manual), using [Sphinx](http://www.sphinx-doc.org)
-   [samples](samples): Example usages of xad
-   [src](src): The C++ sources, including the public headers in [XAD](src/XAD)
-   [test](test): Unit tests

## Coding Style

For convenience, there is a `.clang-format` file in the root of the project which you can (and should) use.

-   **Use Common Sense** - Deviations from these guidelines are explicitly allowed, if they make
    sense in the context of their use.

-   General
    -   Insert a copyright / license note, including short paragraph of the file's 
        purpose, into every source file (look at existing ones for an example)

    -   Use a maximum of 100 characters per source line

-   Naming
    -   Names representing types must be mixed case, starting with upper case `CheckpointCallback`
    -   Variables must be mixed case, starting with lower case `currentRecording`
    -   Named constants must be all upper case, separated by underscores `DEFAULT_LENGTH`
    -   Method and function names must be mixed case, starting with lower case, and must be verbs `pushLhs`
    -   Namespace names must be all lower case `xad`
    -   Template paramters should be single character capitals, or named like types `T`, `Scalar`
    -   Private class members should have an underscore suffix `length_`
    -   All names should be written in English
    -   The name of the object should be avoided in the method name `Line::getLength()`, _not_ `Line::getLineLength()`

-   Includes
    -   Use `#pragma once` instead of include guards

    -   Group includes by libraries, starting with local includes, continuing with third party libraries, 
        and finishing with C++ and C standard libraries

    -   Use `<>` to include globally valid files (including public includes of the current project)

    -   Use `""` to include files relative to the current directory

    -   Includes should only exist on top of the files

-   Comments
    -   Use `//` for comments, and `/*` for disabling large sections of code for debugging

    -   Avoid too many comments within the body of functions - code should be
        self-explanatory

    -   Do comment the functions / methods if their use is not obvious

    -   Use `TODO` inside a comment to flag tasks to attend to in near future

-   Indents, Spacing, Line breaks
    -   Use 4 space to introduce a new scope
    -   Never use tabs
    -   Don't indent namespace scopes
    -   Break before the curly brace in a function definition
    -   Line breaks before curly braces in other contexts are optional
    -   Short loops or functions can be defined in one line

[pr]: https://github.com/xcelerit/xad/compare/

[cla]: https://gist.github.com/xcelerit-dev/4a5c0cf1fbfed7be64308d1c2f47bd25

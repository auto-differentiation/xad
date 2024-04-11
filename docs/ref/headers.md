# Headers and Namespaces

All XAD data types and operations are defined in the `xad` namespace.
For brevity, this namespace has been omitted in the reference section.

XAD provides a general header `XAD/XAD.hpp`,
which includes all headers that are commonly needed to work with XAD.
Typically, this is all that clients need to include.

There are two additional headers provided that can be included on demand:

*   `XAD/Complex.hpp` - for using complex numbers with XAD data types (see [Complex](complex.md)).
    This header should be included wherever [`#!c++ std::complex`](https://en.cppreference.com/w/cpp/numeric/complex) is used.
*   `XAD/StdCompatibility.hpp` - This header imports the XAD math functions
    into the `std` namespace, for compatibility reasons.
    It enables using constructs like [`#!c++ std::sin(x)`](https://en.cppreference.com/w/cpp/numeric/math/sin) where `x` is an XAD type.
    Additionally, it also specialises [`#!c++ std::numeric_limits`](https://en.cppreference.com/w/cpp/types/numeric_limits) for the XAD data types,
    so that it provides traits similar to the standard floating point types.

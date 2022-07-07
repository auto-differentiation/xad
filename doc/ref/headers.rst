.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _ref-headers:

Headers and Namespaces
======================

All XAD data types and operations are defined in the ``xad`` namespace.
For brevity, this namespace has been omitted in the reference section.

XAD provides a general header ``XAD/XAD.hpp``,
which includes all headers that are commonly needed to work with XAD.
Typically, this is all that clients need to include.

There are two additional headers provided that can be included on demand:

* ``XAD/Complex.hpp`` - for using complex numbers with XAD data types (see :ref:`ref-complex`).
  This header should be included wherever :cpp:expr:`std::complex` is used.
* ``XAD/StdCompatibility.hpp`` - This header imports the XAD math functions
  into the ``std`` namespace, for compatibility reasons.
  It enables using constructs like :cpp:expr:`std::sin(x)` where ``x`` is an XAD type.
  Additionally, it also specialises :cpp:expr:`std::numeric_limits` for the XAD data types,
  so that it provides traits similar to the standard floating point types.
.. This file is part of the XAD user manual.
   Copyright (C) 2010-2022 Xcelerit Computing Ltd.
   See the file index.rst for copying conditions. 
   
.. _intro:

Introduction
============

XAD is a fast and comprehensive C++ library for automatic differentiation. 
It targets production-quality code at any scale, 
striving for both ease of use and high performance. 

Key features:

* Forward and adjoint mode for any order, using operator-overloading
* Checkpointing support (for tape memory managment)
* External functions interface (to integrate external libraries)
* Thread-safe tape
* Formal exception-safety guarantees
* High performance 
* Battle-tested in large production code bases

AAD training, consultancy, and commercial licensing is available from 
`Xcelerit <https://www.xcelerit.com/adjoint-algorithmic-differentiation/>`_.


This manual describes how to use the XAD library in your projects. 
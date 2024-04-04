---
title: "Practical Examples"
description: "Explore hands-on examples of XAD for forward and adjoint modes in Python and C++, illustrating easy integration into projects for efficient automatic differentiation."
hide:
  - navigation
---

# Examples

To quickly get an idea how using XAD looks like in practice,
see the examples below.
All code snippets require that the function to be differentiated
is callable with the active data type for algorithmic differentiation.

Note that XAD ships with a set of examples ([Python](https://github.com/auto-differentiation/xad-py/tree/main/samples), [C++](https://github.com/auto-differentiation/xad/tree/main/samples)) that can be used as a starting point for development.
Further, for examples how to use it with a large-scale project,
please see QuantLib Integration ([Python](quantlib-risks/python.md), [C++](quantlib-risks/cxx.md)).

## First Order Forward Mode

=== "Python"

    ```python
    import xad.fwd_1st as xfwd

    # set independent variables
    x0_ad = xfwd.Real(1.0)
    x1_ad = xfwd.Real(1.5)
    x2_ad = xfwd.Real(1.3)
    x3_ad = xfwd.Real(1.2)

    # compute derivative w.r.t. x0
    x0_ad.derivative = 1.0

    # run the algorithm with active variables
    y = 2 * x0_ad + x1_ad - x2_ad * x3_ad

    # output results{
    print(f"y = {y.value}")
    print(f"dy/dx0 = {y.derivative}")
    ```

=== "C++"

    ```c++
    // types for first-order forward mode in double precision
    using mode = xad::fwd<double>;
    using Adouble = mode::active_type;

    // independent variables
    Adouble x0 = 1.0, x1 = 1.5, x2 = 1.3, x3 = 1.2;  
    derivative(x0) = 1.0;   // seed directional derivative
                            // (calculate dy/dx0)
    Adouble y = func(x0, x1, x2, x3); 

    std::cout << "y      = " << value(y) << "\n"
              << "dy/dx0 = " << derivative(y) << "\n";
    ```

## First Order Adjoint Mode


=== "Python"

    ```python
    import xad.adj_1st as xadj

    # set independent variables
    x0_ad = xadj.Real(1.0)
    x1_ad = xadj.Real(1.5)
    x2_ad = xadj.Real(1.3)
    x3_ad = xadj.Real(1.2)

    with xadj.Tape() as tape:
        # ... and register them
        tape.registerInput(x0_ad)
        tape.registerInput(x1_ad)
        tape.registerInput(x2_ad)
        tape.registerInput(x3_ad)

        tape.newRecording()

        y = x0_ad + x1_ad - x2_ad * x3_ad

        tape.registerOutput(y)
        y.derivative = 1.0      # seed output adjoint
        tape.computeAdjoints()  # roll-back tape

        # output results
        print(f"y = {y}")
        print(f"dy/dx0 = {x0_ad.derivative}")
        print(f"dy/dx1 = {x1_ad.derivative}")
        print(f"dy/dx2 = {x2_ad.derivative}")
        print(f"dy/dx3 = {x3_ad.derivative}")
    ```


=== "C++"

    ```c++
    // types for first-order adjoints in double precision
    using mode = xad::adj<double>;
    using Adouble = mode::active_type;
    using Tape = mode::tape_type;

    Tape tape;
    // independent variables and start taping
    std::vector<Adouble> x ={1.0, 1.5, 1.3, 1.2};  
    tape.registerInputs(x);
    tape.newRecording();

    Adouble y = func(x[0], x[1], x[2], x[3]);

    tape.registerOutput(y);
    derivative(y) = 1.0;        // seed output adjoint
    tape.computeAdjoints();     // roll-back tape

    std::cout << "y      = " << value(y) << "\n"
              << "dy/dx0 = " << derivative(x[0]) << "\n"
              << "dy/dx1 = " << derivative(x[1]) << "\n"
              << "dy/dx2 = " << derivative(x[2]) << "\n"
              << "dy/dx3 = " << derivative(x[3]) << "\n";
    ```

## Second Order Forward over Adjoint Mode


=== "C++"

    ```c++
    // types for second-order foward-over-adjoint in double
    using mode = xad::fwd_adj<double>;
    using Adouble = mode::active_type;
    using Tape = mode::tape_type;
      
    Tape tape;
    // independent variables
    std::vector<Adouble> x = {1.0, 1.5, 1.3, 1.2};  
    // seed directional derivative for 2nd order forward
    derivative(value(x0)) = 1.0;  
    // register inputs on tape and record function calls
    tape.registerInputs(x);     
    tape.newRecording();        

    Adouble y = func(x0, x1, x2, x3);

    value(derivative(y)) = 1.0; // seed 1st order adjoint
    tape.computeAdjoints();     // roll-back tape

    std::cout << "y      = " << value(value(y)) << "\n"
              << "\nfirst order derivatives:\n"
              << "dy/dx0 = " << value(derivative(x[0])) << "\n"
              << "dy/dx1 = " << value(derivative(x[1])) << "\n"
              << "dy/dx2 = " << value(derivative(x[2])) << "\n"
              << "dy/dx3 = " << value(derivative(x[3])) << "\n"
              << "\nsecond order derivatives w.r.t. x0:\n"
              << "d2y/dx0dx0 = " << derivative(derivative(x[0])) << "\n"
              << "d2y/dx0dx1 = " << derivative(derivative(x[1])) << "\n"
              << "d2y/dx0dx2 = " << derivative(derivative(x[2])) << "\n"
              << "d2y/dx0dx3 = " << derivative(derivative(x[3])) << "\n";
    ```
---
title: Frequently Asked Questions
description: Practical answers to common issues and best practices around applying automatic differentiation to large code bases.
hide: 
  - navigation
---

# Frequently Asked Questions

## General

??? question "Why did we choose to open-source XAD and QuantLib-Risk?"

    We are committed to the principles of open source. We believe it fosters collaboration, accelerates innovation, and ensures transparency, allowing us to give back to the community while also benefiting from collective expertise. Open sourcing XAD and QuantLib-Risks not only invites peer review and contributions but also demonstrates our confidence in the quality and the robustness of our software.

??? question "How reliable is XAD and QuantLib-Risk?"

    QuantLib-Risk's reliability is evidenced by a comprehensive [CI/CD workflow](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/actions/workflows/ci.yaml), which is rigorously applied across all platforms. All QuantLib unit tests and examples have been adapted for CI/CD and running on each commit. 

    The backbone of QuantLib-Risks, the automatic differentiation tool XAD, features over 1,200 unit tests with a [test coverage nearing 99%](https://coveralls.io/github/auto-differentiation/xad?branch=main). This extensive testing framework underscores our confidence in its reliability and performance.

??? question "Are there control-flow concerns in XAD and QuantLib-Risks?"

    No. XAD's tape (used in QuantLib-Risks) records a single execution path through
    the application, for the provided set of inputs,
    and calculates local derivatives at this point.
    No matter which control flow is used in the program in general,
    for one specific set of inputs the execution path is always linear and control flow issues do not arise.
    If other input values are used, a new tape needs to be recorded.

## Porting Existing Code

??? question "How can I fix compile errors occurring when replacing double with an AD type?"


    Straight replacement of all occurrences of the built-in data type `double`
    with a class like `xad::AReal<double>`, may result in compile errors. 
    This is typically related to function calls, automatic type conversions, 
    template type deductions, or use of libraries. 
    One of the following techniques can typically be used to fix these errors:

    - *Manual Conversions:* Explicit conversions to the active data type may be needed. For example (assuming `Real` is the active XAD type used):
    
        -  ternary (question-mark) expressions:
            ```c++ 
            Real x = condition ? 0.0 : 1.0 - a;
            ```
            may need to be converted as:
            ```c++
            Real x = condition ? Real(0.0) : Real(1.0 - a);
            ```
        - templated function calls:
            ```c++
            Real sum = std::accumulate(vec.begin(), vec.end(), 0.0);
            ```
            need to be converted to:
            ```c++
            Real sum = std::accumulate(vec.begin(), vec.end(), Real(0.0));
            ```
        - deduced return types of lambda expressions:
           ```c++
           auto lambda = [](Real x) { return x * x; };  // return type is an expression type
           ```
           need to be stated explicitly:
           ```c++
           auto lambda = [](Real x) -> Real { return x * x; }; // explict return type
           ```

    - *Convert Called Functions:* If functions called from an AD'ed code do not support 
    using the active data type as parameters, they need to be edited (`double` occurrences replaced) if possible.
    - *External Functions:* If called functions cannot be modified, the 
    [external functions feature](tutorials/cxx.md#external-functions)
    can be used to allow calls in the middle of the AD'ed algorithm. 


??? question "Can I re-use the same code in a pure valuation context?"

    Once a code-base has been AD'ed, it is often the case that the same code base also needs to be
    used in applications that do not require sensitivities.
    In XAD, if variables are not [registered with an active tape](ref/tape.md#registerinput), they can be used in the same 
    fashion as regular `double` types and no tape recording 
    is performed (with the associated overheads).
    However, due to the extra data carried along in the class-based data type,
    the compiler may not optimise the resulting code as aggressively as when `double` is used
    in some cases and a noticeable performance bottleneck can occur for some applications.

    Addressing these challenges necessitates a multifaceted approach. 
    Early steps should include detailed profiling to pinpoint specific performance bottlenecks. 
    This may reveal that the AD version suffers due to the compiler's inability to optimise certain expression templates.
    Performance can often be improved through careful rewriting of expressions 
    and adjustments to loop structures. 
    This optimisation does not compromise the precision of standard double-precision calculations.
    A combined strategy of tool-based AD and manual intervention is frequently the most effective solution, integrating [checkpoints](tutorials/cxx.md#checkpointing) and [external functions](tutorials/cxx.md#external-functions) for sections beyond the tool's support.

    Further, if a global typedef approach is used to replace the active data type,
    it is usually possible to build two versions of the application: one with AD support and another without.
    Then the appropriate version can be used depending if derivatives are needed without code replication.
    

??? question "How can I handle external library calls?"

    If it is not possible (or impractical) to apply XAD to the external library itself (by modifying the 
    library's source code), [external function APIs](tutorials/cxx.md#external-functions) should be used 
    to manually implement the derivatives of the functions and integrate those with the tape.
    These derivatives can be calculated analytically, or estimated using finite differences (bumping).



## Quant Finance Applications

??? question "How can I efficiently apply AD to a Monte-Carlo simulation?"

    **Path-wise**

    Often derivatives of a Monte-Carlo simulation can be calculated in a path-wise fashion. 
    That is, they are calculated individually on every Monte-Carlo path and averaged in the aggregation.
    This not only reduces the tape memory but also allows multi-threaded calculations 
    (if each thread uses a separate tape).

    **Full AD with checkpointing**

    If the mathematical requirements for path-wise AD are not satisfied, 
    a full AD implementation across all paths is needed. 
    This can quickly create memory problems, which can be resolved using [checkpointing](tutorials/cxx.md#checkpointing).


??? question "What is the most efficient way to use adjoint mode for a function with multiple outputs?"

    Adjoint-mode AD is most efficient with an algorithm which maps multiple inputs to a single
    output value. 
    With one application of AD, all sensitivities to the inputs can be obtained. 
    If multiple outputs are needed, 
    it is possible to clear the derivatives stored on the tape, seed the output adjoint of 
    a different output, and roll back the tape again.
    This avoids repeated valuations (forward) executions, as illustrated below:

    ```c++
    tape.registerOutput(output1);
    derivative(output1) = 1.0;   // seed for first output
    tape.computeAdjoints();

    ... // read the input adjoints, which are derivatives of output2

    tape.registerOutput(output2);  
    tape.clearDerivatives();     // clear previous adjoints
    derivative(output2) = 1.0;   // seed for second output
    tape.computeAdjoints();

    ... // read the input adjoints, which are derivatives of output2
    ```


??? question "How can I calculate sensitivities for implicit functions such as an iterative model calibration?"

    Using an adjoint mode AD tape to record the operations across an iterative optimisation (e.g. for model calibration) is impractical and can lead to erroneous results. 
    It is however possible to use the implicit function theorem to deduce 
    the sensitivities of the optimisation inputs (e.g. market parameters) from the 
    sensitivities to the optimisation outputs (e.g. model parameters). 


??? question "How can I handle discontinuous functions?"

    Some functions, e.g. the payoff of a binary option, can't be differentiated in the mathematical sense. 
    However, sensitivities may still be needed and hence approximations need to be used.
    
    Special [smoothed functions](tutorials/smoothed_math.md) can be used to substitute the original discontinuities. 
    For instance, a jump can be replaced with a smooth slope in a short interval around the discontinuity. 
    These smooth functions are differentiable. 
    However, as the original function is modified, 
    caution is required when interpreting the value and sensitivities around the discontinuity.

## Performance


??? question "How can I hand-tune specific performance bottlenecks?"

    To integrate hand-tuned manual AD implementations for a specific bottleneck in an AD'ed algorithm, 
    the [external function feature](tutorials/cxx.md#external-functions) of XAD needs to be used. 
    This allows to fine-tune the performance for a specific part 
    while otherwise still using the convenience of an operator-overloading tool. 


??? question "Is AD safe to use in multi-threaded code?"

    Using multi-threading in tape-less forward mode AD is safe, 
    as no shared state is introduced.
    However, in adjoint mode the operations are recorded on a tape in memory and this can create race conditions if used by multiple threads. 
    Separate tapes must be used in each thread,
    using the [thread-local tape feature](ref/tape.md).


??? question "Can I use GPUs combined with AD?"

    Forward mode AD can be implemented without a tape (provided by many AD tools), 
    adding one extra data item for each `double` to store its derivative. 
    This is straightforward to use on GPUs in principle (using CUDA C++). 
    
    Adjoint mode requires a tape, often dynamically growing in memory as the calculations proceed. 
    As CUDA does not support dynamic memory allocations very well and due to race conditions created by the fine-grained parallelism typical for GPU code, directly using AAD on GPUs is a significant challenge. 
    Currently, no available AD tool supports adjoint mode with a tape on GPUs.

    However, typically GPUs are used for confined sections of the overall application
    which are particularly performance-critical. These sections can be treated as an
    [external function](tutorials/cxx.md#external-functions) and its adjoint can be implemented manually (possibly also using the GPU). 

## Memory Management

??? question "How can I reduce the memory needed for AAD?"

    Several approaches are used to reduce the memory required to store the operations on the XAD tape. They are listed below.

    **Efficient Tape Storage**

    There are vast differences in the memory requirements for different AD tools, depending on how the tape is laid out. 
    The memory can vary up to an order of magnitude. 
    XAD takes particular care to memory efficiency and has a lower memory footprint than
    the majority of the available tools.

    **Use a path-wise approach in a Monte-Carlo simulation**

    In a Monte-Carlo simulation, it is typically possible to exchange the expectation (mean) and differentiation operators. 
    This means, the sensitivities can be calculated along each individual path and then averaged. Instead of storing all paths on tape, only a single path is needed at atime, vastly reducing the memory required.

    **Use checkpointing for block-wise AD**

    By separating the algorithm into multiple stages and only using a tape for each stage separately, the memory requirement can be drastically reduced. 
    This technique is called [checkpointing](tutorials/cxx.md#checkpointing),
    and it trades extra computations for memory. 
    In a practical implementation however, the saved time for memory access may 
    outweigh the time for the added calculations and care must be taken
    how checkpointing is applied.

    **Manually tune sections of the code**

    For specific sections of the algorithm, 
    derivatives may be known analytically or can be implemented more efficiently using manual AD. 
    This can reduce the memory requirements significantly. 


??? question "How can I improve my application's memory access efficiency?"

    The tape required to record operations for adjoint-mode AD often consumes large amounts of memory. 
    As the tape is constantly accessed during the calculations, 
    this can create memory access performance bottlenecks due to frequent cache misses. Using the techniques described in the question "How can I reduce the memory needed for AAD?" reduces the memory for the tape and 
    hence also increases the cache-efficiency significantly.

## QuantLib-Risks

??? question "What performance impact should I expect when calculating risks with QuantLib-Risks?"

    Our observations and feedback from users indicate a cost of up to 3x when calculating an arbitrary number of sensitivities using QuantLib-Risks, compared to the base QuantLib. This efficiency allows for comprehensive risk computation. Should you experience a slowdown exceeding this, we highly encourage sharing a reproducible example on GitHub for community support.

??? question "Where can I find examples of automatic differentiation in QuantLib?"

    We provide a broad selection of examples to illustrate automatic differentiation's application within QuantLib, available in both [C++](https://github.com/auto-differentiation/QuantLib-Risks-Cpp/tree/main/Examples) and [Python](https://github.com/auto-differentiation/QuantLib-Risks-Py/tree/main/Python/examples). 
    These examples are intended for demonstration purposes, aiming to cater to a wide range of applications. As our project is open source, we encourage contributions and feedback through pull requests or open discussions.

??? question "Does the complexity of an example affect its performance in QuantLib-Risks?"

    Generally, the performance of QuantLib-Risks is not inherently dependent on 
    the simplicity of a given example. 
    The underlying mechanism of automatic differentiation, which records arithmetic operations, ensures that the complexity of the original application does not adversely affect performance. 

    Since XAD uses operator-overloading with expression templates, 
    this can place a significant strain on the compiler. 
    This varies with the choice of compiler, the application of compiler flags, and other factors.
    Addressing these challenges necessitates a multifaceted approach,
    based on performance profiling, code tuning, 
    and if necessary combining manually AD'ed code using the external function interface.

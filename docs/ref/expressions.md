# Expressions

## Expression Template

```c++
template <typename T, typename Derived> 
class Expression
```

Represents a mathematical expression in a type.
Active data types, such as [`AReal`](areal.md) and [`FReal`](freal.md),
as well as all mathematical expressions inherit from this class.
All mathematical operations are defined on this type, rather
than any specific derived class.

The derived classes are typically created transparently to the user.

Note that this class uses the CRTP pattern, where `Derived` is
the derived class itself, so that static polymorphism can be used.

All global arithmetic operations defined in C++ are specialized
for `Expression`, so that `double` or `float` can be replaced seamlessly with
an active data type from XAD.
This also includes comparisons.

!!! node "See also"

    [Mathematical Operations](math.md)

## Expression Traits

XAD also defines expression traits to find information about expressions
in a templated context.
This is typically only needed when custom functions dealing with the XAD
expressions are added.

#### `Direction` enum

This enum indicates the direction of algorithmic differentiation associated with a type.

```c++
enum Direction {
    DIR_NONE,       // Not an algorithmic differentiation type
    DIR_FORWARD,    // Forward mode AD type
    DIR_REVERSABLE  // Reverse mode AD type
};
```

#### `ExprTraits`

This is the main traits class to get information on an AD type:

```c++
template <typename T>
struct ExprTraits {
    static const bool isExpr;      // true if an expression of XAD active type
    static const int numVariables; // Number of variables in an expression
    static const bool isForward;   // true if forward-mode AD
    static const bool isReverse;   // true if reverse-mode AD
    static const bool isLiteral;   // true if it's an elementary XAD active type
                                   // and not an expression
    static const Direction direction;  // direction of the expression or type
    
    
    typedef ... nested_type; // underlying type of the expression
                             // e.g. double for AReal<double>
    typedef ... value_type;  // the base active type of a more
                             // complex expression template
    typedef ... scalar_type; // Type when unwrapping a higher order
                             // expression, e.g. FReal<double> for
                             // an expression of AReal<FReal<double>>

};
```

# Direct Adjoint Mode Type `ARealDirect`

## Overview

```c++
template <class Scalar, std::size_t N=1>
struct ARealDirect : public RealDirect<AReal<Scalar, N>, ARealDirect<Scalar, N>>
```

The class `ARealDirect` has the same interface as `AReal`, but does not use expression templates for the operations.
This is to make debugging easier and to fit better with existing code bases with may have
issues with expression templates.

!!! note "See also"

    [AReal](areal.md), [Tape](tape.md), [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

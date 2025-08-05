# Direct Forward Mode Type `FRealDirect`

## Overview

```c++
template <class Scalar, std::size_t N = 1>
struct FRealDirect : public RealDirect<FReal<Scalar, N>, FRealDirect<Scalar, N>>
```

The class `FRealDirect` has the same interface as `FReal`, but does not use expression templates for the operations.
This is to make debugging easier and to fit better with existing code bases,
which may have issues integrating with expression templates.

!!! note "See also"

    [FReal](freal.md), [Tape](tape.md), [Global Functions](global.md), [AD Mode Interface](interface.md),
    [Mathematical Operations](math.md)

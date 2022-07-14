// this test should fail as left and right of : have different types that should not
// be convertible to each other

#include <XAD/XAD.hpp>

int main()
{
    xad::AD value = 1.0;
    xad::AD div = 100.0;
    xad::AD res = value / div < 1.0 ? -(value / div) : (value / div);
    XAD_UNUSED_VARIABLE(res);
}
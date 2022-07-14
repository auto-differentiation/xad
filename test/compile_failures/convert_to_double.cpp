// try conversion to double and make sure that doesn't work implicitly

#include <XAD/XAD.hpp>

int main()
{
    xad::AD value = 1.0;
    double x = value;
    (void)x;
}
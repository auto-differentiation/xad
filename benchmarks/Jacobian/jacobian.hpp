#pragma once

#include <XAD/XAD.hpp>
#include <XAD/Jacobian.hpp>

#include <functional>

template <typename T>
std::function<T(std::vector<T> &)> make_foo() {
    return [](std::vector<T> &x) -> T {
        return {sin(x[0] + x[1]), sin(x[1] + x[2]), cos(x[2] + x[3]), cos(x[3] + x[0])};
    };
}

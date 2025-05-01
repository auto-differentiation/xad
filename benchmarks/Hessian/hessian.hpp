#pragma once

#include <functional>
#include <vector>

#include <XAD/XAD.hpp>
#include <XAD/Hessian.hpp>

// foo function
template <typename T>
std::function<T(std::vector<T> &)> make_foo() {
    return [](std::vector<T> &x) -> T {
        return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0]);
    };
}

// ackley function
template <typename T>
std::function<T(std::vector<T> &)> make_ackley() {
    return [](std::vector<T> &x) -> T {
        T sum = 0.0;
        T cossum = 0.0;
        T n = static_cast<T>(x.size());

        for (auto &xi : x) {
            sum += xi * xi;
            cossum += cos(2.0 * M_PI * xi);
        }

        return -20.0 * exp(-0.2 * sqrt(sum / n)) - exp(cossum / n) + 20.0 + exp(1.0);
    };
}

// neural loss function
template <typename T>
std::function<T(std::vector<T> &)> make_neuralLoss() {
    return [](std::vector<T> &x) -> T {
        T a = 0.0;
        T b = 1.11;

        for (std::size_t i = 0; i < x.size(); ++i) {
            a += x[i] * static_cast<T>(i);
        }

        return log(1 + exp(a + b));
    };
}

// sparse function
template <typename T>
std::function<T(std::vector<T> &)> make_sparse() {
    return [](std::vector<T> &x) -> T {
        T sum = 0.0;

        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T diff = x[i] - x[i + 1];
            sum += diff * diff;
        }

        return sum;
    };
}

// dense function
// the idea is that every variable has a dependency on every other variable
// that way every entry of the hessian is non-zero
template <typename T>
std::function<T(std::vector<T> &)> make_dense() {
    return [](std::vector<T> &x) -> T {
        T t = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            for (std::size_t j = 0; j < x.size(); ++j) {
                if (i == j) {
                    continue;
                }
                t += x[i] * x[j];
            }
        }

        return t;
    };
}
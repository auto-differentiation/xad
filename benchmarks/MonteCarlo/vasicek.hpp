#include <XAD/XAD.hpp>

#include <random>

template <typename T>
double GBM(T& mu, T& sigma, int n, int dt, int x0)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution d{5.0, 2.0};

    double x = exp(
        (mu - pow(sigma, 2.0) / 2.0) * dt
        + sigma * sqrt(dt)
    )
}
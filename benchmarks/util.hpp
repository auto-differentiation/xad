#include <random>
#include <vector>
#include <algorithm>

template <typename T>
std::vector<T> make_vector(size_t size)
{
    std::random_device rd;  
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    auto gen = [&]() {
        return dist(mersenne);
    };

    std::vector<T> vec(size);
    std::generate(vec.begin(), vec.end(), gen);

    return vec;
}
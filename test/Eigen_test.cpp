

#include <XAD/XAD.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(Eigen, Eigen)
{
    Eigen::Matrix<xad::AReal<double>, Eigen::Dynamic, Eigen::Dynamic> M(2, 2);
    M(0, 0) = AReal(1.0);
    M(0, 1) = AReal(2.0);
    M(1, 0) = AReal(3.0);
    M(1, 1) = AReal(4.0);
    auto N = M.transpose();
}
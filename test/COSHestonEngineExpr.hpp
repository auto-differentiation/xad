#pragma once
#include <XAD/StdCompatibility.hpp>

template <class T>
struct TestHeston
{
    T kappa_;
    T rho_;
    T theta_;
    T sigma_;
    T v0_;
    TestHeston() : kappa_(.4), rho_(.8), theta_(1.2), sigma_(.8), v0_(12.) {}

    T c4(T t) const
    {
        const T sigma2 = sigma_ * sigma_;
        const T sigma3 = sigma2 * sigma_;
        const T sigma4 = sigma2 * sigma2;
        const T kappa2 = kappa_ * kappa_;
        const T kappa3 = kappa2 * kappa_;
        const T kappa4 = kappa2 * kappa2;
        const T kappa5 = kappa2 * kappa3;
        const T kappa6 = kappa3 * kappa3;
        const T kappa7 = kappa4 * kappa3;
        const T rho2 = rho_ * rho_;
        const T rho3 = rho2 * rho_;
        const T t2 = t * t;
        const T t3 = t2 * t;

        return (sigma2 *
                (3 * sigma4 * (theta_ - 4 * v0_) +
                 3 * exp(4 * kappa_ * t) *
                     ((-93 * sigma4 + 64 * kappa5 * (t + 4 * rho2 * t) +
                       4 * kappa_ * sigma3 * (176 * rho_ + 5 * sigma_ * t) -
                       32 * kappa2 * sigma2 * (11 + 50 * rho2 + 5 * rho_ * sigma_ * t) +
                       32 * kappa3 * sigma_ *
                           (3 * sigma_ * t + 4 * rho_ * (10 + 8 * rho2 + 3 * rho_ * sigma_ * t)) -
                       32 * kappa4 * (5 + 4 * rho_ * (6 * rho_ + (3 + 2 * rho2) * sigma_ * t))) *
                          theta_ +
                      4 * (4 * kappa2 - 4 * kappa_ * rho_ * sigma_ + sigma2) *
                          (4 * kappa2 * (1 + 4 * rho2) - 20 * kappa_ * rho_ * sigma_ + 5 * sigma2) *
                          v0_) +
                 24 * exp(kappa_ * t) * sigma2 *
                     (-2 * kappa2 * (-1 + rho_ * sigma_ * t) * (theta_ - 3 * v0_) +
                      sigma2 * (theta_ - 2 * v0_) +
                      kappa_ * sigma_ *
                          (-4 * rho_ * theta_ + sigma_ * t * theta_ + 10 * rho_ * v0_ -
                           3 * sigma_ * t * v0_)) +
                 12 * exp(2 * kappa_ * t) *
                     (sigma4 * (7 * theta_ - 4 * v0_) +
                      8 * kappa4 * (1 + 2 * rho_ * sigma_ * t * (-2 + rho_ * sigma_ * t)) *
                          (theta_ - 2 * v0_) +
                      2 * kappa_ * sigma3 *
                          (-24 * rho_ * theta_ + 5 * sigma_ * t * theta_ + 20 * rho_ * v0_ -
                           6 * sigma_ * t * v0_) +
                      4 * kappa2 * sigma2 *
                          ((6 + 20 * rho2 - 14 * rho_ * sigma_ * t + sigma2 * t2) * theta_ -
                           2 * (3 + 12 * rho2 - 10 * rho_ * sigma_ * t + sigma2 * t2) * v0_) +
                      8 * kappa3 * sigma_ *
                          ((3 * sigma_ * t +
                            2 * rho_ * (-4 + sigma_ * t * (4 * rho_ - sigma_ * t))) *
                               theta_ +
                           2 *
                               (-3 * sigma_ * t +
                                2 * rho_ * (3 + sigma_ * t * (-3 * rho_ + sigma_ * t))) *
                               v0_)) -
                 8 * exp(3 * kappa_ * t) *
                     (16 * kappa6 * rho2 * t2 * (-3 + rho_ * sigma_ * t) * (theta_ - v0_) -
                      3 * sigma4 * (7 * theta_ + 2 * v0_) +
                      2 * kappa3 * sigma_ *
                          ((192 * (rho_ + rho3) - 6 * (9 + 40 * rho2) * sigma_ * t +
                            42 * rho_ * sigma2 * t2 - sigma3 * t3) *
                               theta_ +
                           (-48 * rho3 + 18 * (1 + 4 * rho2) * sigma_ * t -
                            24 * rho_ * sigma2 * t2 + sigma3 * t3) *
                               v0_) +
                      12 * kappa4 *
                          ((-4 - 24 * rho2 + 8 * rho_ * (4 + 3 * rho2) * sigma_ * t -
                            (3 + 14 * rho2) * sigma2 * t2 + rho_ * sigma3 * t3) *
                               theta_ +
                           (8 * rho2 - 8 * rho_ * (2 + rho2) * sigma_ * t +
                            (3 + 8 * rho2) * sigma2 * t2 - rho_ * sigma3 * t3) *
                               v0_) -
                      6 * kappa2 * sigma2 *
                          ((15 + 80 * rho2 - 35 * rho_ * sigma_ * t + 2 * sigma2 * t2) * theta_ +
                           (3 + sigma_ * t * (7 * rho_ - sigma_ * t)) * v0_) +
                      24 * kappa5 * t *
                          ((-2 + rho_ * (4 * sigma_ * t +
                                         rho_ * (-8 + sigma_ * t * (4 * rho_ - sigma_ * t)))) *
                               theta_ +
                           (2 + rho_ * (-4 * sigma_ * t +
                                        rho_ * (4 + sigma_ * t * (-2 * rho_ + sigma_ * t)))) *
                               v0_) +
                      3 * kappa_ * sigma3 *
                          (sigma_ * t * (-9 * theta_ + v0_) + 10 * rho_ * (6 * theta_ + v0_))))) /
               (64. * exp(4 * kappa_ * t) * kappa7);
    }
};

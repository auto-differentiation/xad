/*******************************************************************************

   Unit tests for controlled roll-back to specific positions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2025 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include <XAD/StdCompatibility.hpp>
#include <XAD/XAD.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

TEST(PartialRollback, MultiDerivativesInLoop)
{
    xad::AD in = 2.0;
    xad::AD::tape_type tape;
    tape.registerInput(in);
    std::vector<double> values(9);
    std::vector<double> deriv(9);

    tape.newRecording();
    auto pos = tape.getPosition();
    for (std::size_t p = 1; p < 10; ++p)
    {
        xad::AD v = static_cast<double>(p) * in;
        tape.registerOutput(v);
        derivative(v) = 1.0;
        tape.computeAdjointsTo(pos);
        values[p - 1] = value(v);
        deriv[p - 1] = derivative(in);
        tape.resetTo(pos);
        tape.clearDerivatives();
    }

    EXPECT_THAT(values, ElementsAre(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0));
    EXPECT_THAT(deriv, ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
}

namespace
{
template <class T, class U>
T evaluate(U path, const T& val)
{
    return val * val * static_cast<T>(path) + std::exp(val);
    // derivative = 2 * val * path + std:exp(val);
}

}  // namespace

TEST(PartialRollback, MultiDerivativesInComplexLoop)
{
    xad::AD in = 2.0;
    xad::AD::tape_type tape;
    tape.registerInput(in);
    std::vector<double> values(4), expected(4);
    std::vector<double> der(4), der_exp(4);

    tape.newRecording();
    auto pos = tape.getPosition();
    for (std::size_t p = 0; p < 4; ++p)
    {
        xad::AD vt = evaluate(p, in);
        xad::AD v = 2 * vt;
        xad::AD xxx = v * v;
        tape.registerOutput(v);
        derivative(v) = 1.0;
        tape.computeAdjointsTo(pos);
        values[p] = value(v);
        der[p] = derivative(in);
        expected[p] = 2.0 * evaluate(p, value(in));
        der_exp[p] = 2. * (2.0 * value(in) * static_cast<double>(p) + std::exp(value(in)));
        tape.resetTo(pos);
        tape.clearDerivatives();
    }

    EXPECT_THAT(values, ContainerEq(expected));
    EXPECT_THAT(der, ContainerEq(der_exp));
}

TEST(PartialRollback, MultiDerivativesInNestedLoop)
{
    xad::AD r = 0.3;
    xad::AD q = 0.4;
    std::vector<int> intval{1, 2, 3, 4};
    xad::AD::tape_type tape;
    tape.registerInput(r);
    tape.registerInput(q);
    tape.newRecording();

    auto sim_position = tape.getPosition();
    for (int p = 0; p < 5; ++p)
    {
        tape.resetTo(sim_position);
        for (int t = 0; t < 5; ++t)
        {
            // value
            xad::AD rpt = q * p * std::exp(-r * double(t));
            // partial derivatives manual
            double drpt_dq = double(p) * std::exp(-value(r) * double(t));
            double drpt_dr = value(q) * double(p) * -t * std::exp(-value(r) * double(t));

            auto tpos = tape.getPosition();
            for (std::size_t tidx = 0; tidx < intval.size(); ++tidx)
            {
                // value
                xad::AD v = evaluate(intval[tidx], rpt);
                // partial derivative manual
                double dv_drpt = 2. * value(rpt) * double(intval[tidx]) + std::exp(value(rpt));

                // full derivatives AAD
                tape.registerOutput(v);
                derivative(v) = 1.0;
                tape.computeAdjoints();
                double dv_dr_act = derivative(r);
                double dv_dq_act = derivative(q);
                tape.resetTo(tpos);
                tape.clearDerivatives();

                // full derivatives manual
                double dv_dr_exp = dv_drpt * drpt_dr;
                double dv_dq_exp = dv_drpt * drpt_dq;

                // compare
                EXPECT_THAT(dv_dr_act, DoubleEq(dv_dr_exp))
                    << "at (" << p << "," << t << "," << tidx << ")";
                EXPECT_THAT(dv_dq_act, DoubleEq(dv_dq_exp))
                    << "at (" << p << "," << t << "," << tidx << ")";
            }
        }
    }
}

TEST(PartialRollback, ClearDerivativesAfter)
{
    xad::AD::tape_type tape;
    xad::AD x1 = 1.0;
    tape.registerInput(x1);
    xad::AD x2 = 1.2 * x1;
    auto pos = tape.getPosition();
    xad::AD x3 = 1.4 * x2 * x1;
    xad::AD x4 = x2 + x3;
    tape.registerOutput(x4);
    derivative(x4) = 1.0;
    derivative(x3) = 1.0;
    derivative(x2) = 1.0;
    derivative(x1) = 1.0;
    tape.clearDerivativesAfter(pos);

    EXPECT_THAT(derivative(x2), DoubleEq(1.));
    EXPECT_THAT(derivative(x1), DoubleEq(1.));
    EXPECT_THROW(derivative(x3), xad::OutOfRange);
    EXPECT_THROW(derivative(x4), xad::OutOfRange);
}

TEST(PartialRollback, ClearFullTape)
{
    xad::AD r = 0.3;
    xad::AD q = 0.4;
    xad::AD::tape_type tape;
    tape.registerInput(r);
    tape.registerInput(q);
    tape.newRecording();
    xad::AD y = exp(r + q);
    tape.registerOutput(y);
    derivative(y) = 1.0;
    tape.computeAdjoints();

    double yv = value(y);
    double dr = derivative(r);
    double dq = derivative(q);
    auto slotr = r.getSlot();
    auto slotq = q.getSlot();
    auto sloty = y.getSlot();

    tape.clearAll();

    xad::AD r1 = 0.3;
    xad::AD q1 = 0.4;
    tape.registerInput(r1);
    tape.registerInput(q1);
    tape.newRecording();
    xad::AD y1 = exp(r1 + q1);
    tape.registerOutput(y1);
    derivative(y1) = 1.0;
    tape.computeAdjoints();

    double yv1 = value(y1);
    double dr1 = derivative(r1);
    double dq1 = derivative(q1);
    auto slotr1 = r1.getSlot();
    auto slotq1 = q1.getSlot();
    auto sloty1 = y1.getSlot();

    // slot and values should be all the same - it restarts

    EXPECT_THAT(yv, DoubleEq(yv1));
    EXPECT_THAT(dr, DoubleEq(dr1));
    EXPECT_THAT(dq, DoubleEq(dq1));
    EXPECT_THAT(slotr, Eq(slotr1));
    EXPECT_THAT(slotq, Eq(slotq1));
    EXPECT_THAT(sloty, Eq(sloty1));
}

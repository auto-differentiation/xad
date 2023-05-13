/*******************************************************************************

   Unit tests for higher order derivatives.

   This file is part of XAD, a fast and comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2022 Xcelerit Computing Ltd.

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

#include <XAD/XAD.hpp>
#include <gtest/gtest.h>

#include <vector>

using std::vector;

template <typename T>
void f(const std::vector<T>& x, std::vector<T>& y)
{
    T v = tan(x[2] * x[3]);
    T w = x[1] - v;
    y[0] = x[0] * v / w;
    y[1] = y[0] * x[1];
}

void driver_fwd_adj(const vector<double>& xv, const vector<double>& xt2, vector<double>& xa1,
                    vector<double>& xa1t2, vector<double>& yv, vector<double>& yt2,
                    vector<double>& ya1, vector<double>& ya1t2)
{
    typedef xad::FReal<double> base_type;
    typedef xad::AReal<base_type> ad_type;
    typedef xad::Tape<base_type> tape_type;

    tape_type tape;
    const auto n = xv.size();
    const auto m = yv.size();
    // setup inputs and set their forward derivatives
    vector<ad_type> x(n), y(m);
    for (std::size_t i = 0; i < n; i++) tape.registerInput(x[i]);

    for (std::size_t i = 0; i < n; i++)
    {
        x[i] = xv[i];

        derivative(value(x[i])) = xt2[i];
    }
    tape.newRecording();

    f(x, y);

    for (std::size_t i = 0; i < m; i++) tape.registerOutput(y[i]);
    for (std::size_t i = 0; i < n; i++)
    {
        value(derivative(x[i])) = xa1[i];
        derivative(derivative(x[i])) = xa1t2[i];
    }

    for (std::size_t i = 0; i < m; i++)
    {
        yv[i] = value(value(y[i]));
        yt2[i] = derivative(value(y[i]));
        value(derivative(y[i])) = ya1[i];
        derivative(derivative(y[i])) = ya1t2[i];
    }
    tape.computeAdjoints();

    for (std::size_t i = 0; i < n; i++)
    {
        xa1t2[i] = derivative(derivative(x[i]));
        xa1[i] = value(derivative(x[i]));
    }
    for (std::size_t i = 0; i < m; i++)
    {
        ya1t2[i] = derivative(derivative(y[i]));
        ya1[i] = value(derivative(y[i]));
    }
}

template <class T>
inline void compareLimitedPrecision(T ref, T act, int prec, const std::string& msg)
{
    // using decimal precision on output stream for comparison
    std::stringstream sstr1, sstr2;
    sstr1.precision(prec);
    sstr2.precision(prec);
    sstr1 << ref;
    sstr2 << act;
    sstr1.flush();
    sstr2.flush();
    EXPECT_EQ(sstr1.str(), sstr2.str()) << msg;
}

TEST(HigherOrder, fwd_adj)
{
    const std::size_t n = 4, m = 2;
    std::cout.precision(15);
    vector<double> xv(n), xa1(n), xt2(n), xa1t2(n);
    vector<double> yv(m), ya1(m), yt2(m), ya1t2(m);
    for (std::size_t i = 0; i < n; i++)
    {
        xv[i] = 1;
        xt2[i] = 1;
        xa1[i] = 1;
        xa1t2[i] = 0;
    }
    for (std::size_t i = 0; i < m; i++)
    {
        ya1[i] = 1;
        ya1t2[i] = 0;
    }

    driver_fwd_adj(xv, xt2, xa1, xa1t2, yv, yt2, ya1, ya1t2);

    compareLimitedPrecision(-2.794018912492, yv[0], 13, "y[0]");
    compareLimitedPrecision(-2.794018912492, yv[1], 13, "y[0]");
    compareLimitedPrecision(-4.588037824984, xa1[0], 13, "x_(1)[0]");
    compareLimitedPrecision(-11.81906445423, xa1[1], 13, "x_(1)[1]");
    compareLimitedPrecision(23.05009108348, xa1[2], 13, "x_(1)[2]");
    compareLimitedPrecision(23.05009108348, xa1[3], 13, "x_(1)[3]");
    compareLimitedPrecision(14.24354940012, yt2[0], 13, "y^(2)[0]");
    compareLimitedPrecision(11.44953048763, yt2[1], 13, "y^(2)[1]");
    compareLimitedPrecision(31.28111771273, xa1t2[0], 13, "x_(1)^(2)[0]");
    compareLimitedPrecision(165.5690423573, xa1t2[1], 13, "x_(1)^(2)[1]");
    compareLimitedPrecision(-248.3747280974, xa1t2[2], 13, "x_(1)^(2)[2]");
    compareLimitedPrecision(-248.3747280974, xa1t2[3], 13, "x_(1)^(2)[3]");
}

void driver_fwd_fwd(const vector<double>& xv, const vector<double>& xt1, const vector<double>& xt2,
                    const vector<double>& xt1t2, vector<double>& yv, vector<double>& yt1,
                    vector<double>& yt2, vector<double>& yt1t2)
{
    typedef xad::FReal<double> base_type;
    typedef xad::FReal<base_type> ad_type;

    const auto n = xv.size(), m = yv.size();
    vector<ad_type> x(n), y(m);
    for (std::size_t i = 0; i < n; i++)
    {
        value(value(x[i])) = xv[i];
        derivative(value(x[i])) = xt1[i];
        value(derivative(x[i])) = xt2[i];
        derivative(derivative(x[i])) = xt1t2[i];
    }

    f(x, y);

    for (std::size_t i = 0; i < m; i++)
    {
        yv[i] = value(value(y[i]));
        yt1[i] = derivative(value(y[i]));
        yt2[i] = value(derivative(y[i]));
        yt1t2[i] = derivative(derivative(y[i]));
    }
}

TEST(HigherOrder, fwd_fwd)
{
    using std::cout;
    using std::endl;
    const std::size_t n = 4, m = 2;
    cout.precision(15);
    vector<double> xv(n), xt1(n), xt2(n), xt1t2(n);
    vector<double> yv(m), yt1(m), yt2(m), yt1t2(m);
    for (std::size_t i = 0; i < n; i++)
    {
        xv[i] = 1;
        xt1[i] = 1;
        xt2[i] = 1;
        xt1t2[i] = 1;
    }
    driver_fwd_fwd(xv, xt1, xt2, xt1t2, yv, yt1, yt2, yt1t2);

    compareLimitedPrecision(-2.794018912492, yv[0], 13, "y[0]");
    compareLimitedPrecision(-2.794018912492, yv[1], 13, "y[1]");
    compareLimitedPrecision(14.24354940012, yt1[0], 13, "y^(1)[0]");
    compareLimitedPrecision(11.44953048763, yt1[1], 13, "y^(1)[1]");
    compareLimitedPrecision(14.24354940012, yt2[0], 13, "y^(2)[0]");
    compareLimitedPrecision(11.44953048763, yt2[1], 13, "y^(2)[1]");
    compareLimitedPrecision(-149.9496480624, yt1t2[0], 13, "y^(1,2)[0]");
    compareLimitedPrecision(-124.2565681746, yt1t2[1], 13, "y^(1,2)[1]");
}

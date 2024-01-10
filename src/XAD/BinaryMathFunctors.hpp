/*******************************************************************************

   Functors for binary math functions.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2023 Xcelerit Computing Ltd.

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

#pragma once

#include <XAD/Macros.hpp>
#include <XAD/MathFunctions.hpp>

namespace xad
{
////////////// Pow

template <class Scalar>
struct pow_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return pow(a, b); }
    XAD_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b, const Scalar&) const
    {
        return b * pow(a, b - Scalar(1));
    }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar&, const Scalar& v) const
    {
        return log(a) * v;
    }
};

template <class Scalar>
struct OperatorTraits<pow_op<Scalar> >
{
    enum
    {
        useResultBasedDerivatives = 1
    };
};

/// smooth ABS

template <class Scalar>
struct smooth_abs_op
{
    XAD_INLINE Scalar operator()(const Scalar& x, const Scalar& c) const
    {
        if (abs(x) > c)
            return abs(x);
        if (x < Scalar())
        {
            return x * x * (Scalar(2) / c + x / (c * c));
        }
        else
        {
            return x * x * (Scalar(2) / c - x / (c * c));
        }
    }

    XAD_INLINE Scalar derivative_a(const Scalar& x, const Scalar& c) const
    {
        if (x > c)
            return Scalar(1);
        else if (x < -c)
            return Scalar(-1);
        else if (x < Scalar())
        {
            return x / (c * c) * (Scalar(3) * x + Scalar(4) * c);
        }
        else
            return -x / (c * c) * (Scalar(3) * x - Scalar(4) * c);
    }

    XAD_INLINE Scalar derivative_b(const Scalar& x, const Scalar& c) const
    {
        if (x > c || x < -c)
            return Scalar();
        else if (x < Scalar())
        {
            return -Scalar(2) * x * x * (c + x) / (c * c * c);
        }
        else
        {
            return -Scalar(2) * x * x * (c - x) / (c * c * c);
        }
    }
};

//////// max

// need this complicated expression to have a kind-of smooth 2nd derivative
template <class Scalar>
struct max_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
            return (a < b) ? b : a;
        else
            return (a + b + abs(a - b)) / Scalar(2);
    }
    XAD_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (b < a)
                return Scalar(1);
            else if (a < b)
                return Scalar();
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) + (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (b < a)
                return Scalar();
            else if (a < b)
                return Scalar(1);
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) - (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
};

////// min

template <class Scalar>
struct min_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
            return (a < b) ? a : b;
        else
            return (a + b - abs(a - b)) / Scalar(2);
    }
    XAD_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (a < b)
                return Scalar(1);
            else if (b < a)
                return Scalar();
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) - (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (a < b)
                return Scalar();
            else if (b < a)
                return Scalar(1);
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) + (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
};

///////// fmax / fmin

template <class Scalar>
struct fmax_op : max_op<Scalar>
{
};
template <class Scalar>
struct fmin_op : min_op<Scalar>
{
};

/////////// fmod

template <class Scalar>
struct fmod_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return fmod(a, b); }
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -floor(a / b); }
};

template <class Scalar>
struct atan2_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return atan2(a, b); }
    XAD_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        return b / (a * a + b * b);
    }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        return -a / (a * a + b * b);
    }
};

template <class Scalar>
struct hypot_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return hypot(a, b); }
    XAD_INLINE Scalar derivative_a(const Scalar& a, const Scalar&, const Scalar& v) const
    {
        return a / v;
    }
    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar& b, const Scalar& v) const
    {
        return b / v;
    }
};

template <class Scalar>
struct OperatorTraits<hypot_op<Scalar> >
{
    enum
    {
        useResultBasedDerivatives = 1
    };
};

template <class Scalar>
struct remainder_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return remainder(a, b); }
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        // function is rare enough that there's no need to optimize this better
        int n_;
        using std::remquo;
        XAD_UNUSED_VARIABLE(remquo(a, b, &n_));
        return Scalar(-n_);
    }
};

template <class Scalar>
struct remquo_op
{
    XAD_INLINE explicit remquo_op(int* quo) : quo_(quo), q_() {}
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        using std::remquo;
        Scalar v = remquo(a, b, &q_);
        *quo_ = q_;
        return v;
    }
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-q_); }
    int* quo_;
    mutable int q_;
};

template <class Scalar>
struct nextafter_op
{
    XAD_INLINE explicit nextafter_op() {}
    XAD_INLINE Scalar operator()(const Scalar& from, const Scalar& to) const
    {
        return nextafter(from, to);
    }
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(0); }
};

}  // namespace xad

/*******************************************************************************

   Declaration of traits classes.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

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

#include <XAD/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

// Forward mode: a zero tangent adds nothing, even with an infinite or NaN partial (as for a
// constant operand). GCC on x86-64 compiles a compare for that to branches per op (and per
// lane), so there tangentTerm masks with an SSE compare and AND instead.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__SSE2__)
#define XAD_BRANCH_FREE_TANGENTS
#include <emmintrin.h>
#endif

namespace xad
{

template <class, std::size_t>
struct FReal;

// How an expression node holds an operand: FReal leaves larger than two doubles (five with
// clang-cl), or nested ones larger than four, by reference (as ADVar holds an AReal), so they
// must outlive the full expression; smaller ones, which copy at less cost, and everything else
// by value.
template <class Expr>
struct ExprStorage
{
    typedef Expr type;
};

template <class Scalar, std::size_t N>
struct ExprStorage<FReal<Scalar, N>>
{
#if defined(_MSC_VER) && defined(__clang__)
    // clang-cl outlines the tangent of by-reference expressions up to this size
    static const std::size_t byValueDoubles = 5;
#else
    static const std::size_t byValueDoubles = 2;
#endif
    typedef typename std::conditional<(sizeof(FReal<Scalar, N>) > byValueDoubles * sizeof(double)),
                                      const FReal<Scalar, N>&, FReal<Scalar, N>>::type type;
};

// nested FReal up to four doubles by value: gcc outlines the by-reference form, reloads the
// operands after each partial store and splits the result into scalars, which costs more than
// the copy
template <class Scalar, std::size_t M, std::size_t N>
struct ExprStorage<FReal<FReal<Scalar, M>, N>>
{
    typedef FReal<FReal<Scalar, M>, N> T;
    typedef typename std::conditional<(sizeof(T) > 4 * sizeof(double)), const T&, T>::type type;
};

enum Direction
{
    DIR_NONE,
    DIR_FORWARD,
    DIR_REVERSE
};

template <class T>
struct ExprTraits
{
    static const bool isExpr = false;
    static const int numVariables = 0;
    static const bool isForward = false;
    static const bool isReverse = false;
    static const bool isLiteral = false;
    static const Direction direction = Direction::DIR_NONE;
    static const std::size_t vector_size = 1;

    typedef T nested_type;
    typedef T value_type;
    typedef T scalar_type;
};

template <class T>
struct ExprTraits<const T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<volatile T> : ExprTraits<T>
{
};
template <class T>
struct ExprTraits<const volatile T> : ExprTraits<T>
{
};

template <class Op>
struct OperatorTraits
{
    enum
    {
        useResultBasedDerivatives = 0
    };
};

template <class T>
struct float_or_double : public std::false_type
{
};
template <>
struct float_or_double<float> : public std::true_type
{
};
template <>
struct float_or_double<double> : public std::true_type
{
};

template <class, std::size_t>
struct ARealDirect;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct AReal;

namespace detail
{
// Whether recording writes a T straight into tape memory: trivially copyable types, and FReal /
// FRealDirect over them. Not built on the std traits of XAD's types, which StdCompatibility.hpp
// specialises.
template <class T>
struct is_direct_recordable : std::is_trivially_copyable<T>
{
};
template <class T, std::size_t N>
struct is_direct_recordable<FReal<T, N>> : is_direct_recordable<T>
{
};
template <class T, std::size_t N>
struct is_direct_recordable<FRealDirect<T, N>> : is_direct_recordable<T>
{
};
template <class T, std::size_t N>
struct is_direct_recordable<AReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_direct_recordable<ARealDirect<T, N>> : std::false_type
{
};

// Whether a directly recorded partial is assigned (trivially copyable non-XAD types) or
// constructed in place (FReal, FRealDirect)
template <class T>
struct is_assigned_partial : std::is_trivially_copyable<T>
{
};
template <class T, std::size_t N>
struct is_assigned_partial<FReal<T, N>> : std::false_type
{
};
template <class T, std::size_t N>
struct is_assigned_partial<FRealDirect<T, N>> : std::false_type
{
};
}  // namespace detail

namespace detail
{

template <class T>
struct is_direct : public std::false_type
{
};

template <class T, std::size_t N>
struct is_direct<ARealDirect<T, N>> : public std::true_type
{
};

template <class T, std::size_t N>
struct is_direct<FRealDirect<T, N>> : public std::true_type
{
};

// built-in arithmetic types (is_fundamental: StdCompatibility.hpp makes active types arithmetic)
template <class T>
struct is_builtin_arithmetic
    : public std::integral_constant<bool,
                                    std::is_arithmetic<T>::value && std::is_fundamental<T>::value>
{
};

// operands forwarded to the underlying AReal / FReal: built-in arithmetic types, expressions
template <class T>
struct direct_operand
    : public std::integral_constant<bool, is_builtin_arithmetic<T>::value || ExprTraits<T>::isExpr>
{
};

// other operands implicitly convertible to D (e.g. QuantLib's Null<Real>); not enums
template <class D, class T>
struct direct_convertible_operand
    : public std::integral_constant<bool, !direct_operand<T>::value && !is_direct<T>::value &&
                                              !std::is_enum<T>::value &&
                                              std::is_convertible<const T&, D>::value>
{
};

// float, double and active types based on them (higher-order direct mode)
template <class T>
struct direct_scalar : public float_or_double<typename ExprTraits<T>::nested_type>
{
};

// active types: expressions (AReal, FReal, ...) and direct mode types
template <class T>
struct is_active : public std::integral_constant<bool, ExprTraits<T>::isExpr || is_direct<T>::value>
{
};

// Forward mode: derivatives are computed as usual and redone with tangentTerm if NaN, except
// for ops whose partials are finite where their arguments are, or which give the tangent
// themselves (op.tangent). With XAD_BRANCH_FREE_TANGENTS, tangentTerm is used directly.

template <class Op>
struct HasFinitePartials : std::false_type
{
};

template <class Op>
struct HasTangentRule : std::false_type
{
};

// An op's partial in its first (A) or second (B) operand where it is the constant +1 or -1, else 0.
// Recording passes the multiplier on for these instead of multiplying by them, which costs nested
// products for higher-order types.
template <class Op>
struct UnitPartialA : std::integral_constant<int, 0>
{
};

template <class Op>
struct UnitPartialB : std::integral_constant<int, 0>
{
};

template <class D>
XAD_INLINE typename std::enable_if<float_or_double<D>::value, bool>::type isNaNTangent(const D& d)
{
    return std::isnan(d);
}

// active tangents (higher order) can carry dependencies with a zero value, so they are kept
template <class D>
XAD_INLINE typename std::enable_if<!float_or_double<D>::value, bool>::type isNaNTangent(const D&)
{
    return false;
}

template <class T, std::size_t N>
XAD_INLINE bool isNaNTangent(const Vec<T, N>& d)
{
    for (std::size_t i = 0; i < N; ++i)
        if (isNaNTangent(d[i]))
            return true;
    return false;
}

// isNaNTangent(d), or false for ops with finite partials (a tag: MSVC warns on constant conditions)
template <class D>
XAD_INLINE bool isNaNTangent(const D&, std::true_type)
{
    return false;
}

template <class D>
XAD_INLINE bool isNaNTangent(const D& d, std::false_type)
{
    return isNaNTangent(d);
}

#ifdef XAD_BRANCH_FREE_TANGENTS
// d, or 0 where t is +-0, in xmm registers (a bit mask on integers moves each lane xmm -> gpr)
XAD_INLINE double maskedTerm(double t, double d)
{
    const __m128d m = _mm_cmpneq_sd(_mm_set_sd(t), _mm_setzero_pd());
    return _mm_cvtsd_f64(_mm_and_pd(m, _mm_set_sd(d)));
}

XAD_INLINE float maskedTerm(float t, float d)
{
    const __m128 m = _mm_cmpneq_ss(_mm_set_ss(t), _mm_setzero_ps());
    return _mm_cvtss_f32(_mm_and_ps(m, _mm_set_ss(d)));
}
#endif

// partial * tangent, but 0 where the tangent is +-0
template <class P, class D>
XAD_INLINE typename std::enable_if<float_or_double<D>::value, D>::type tangentTerm(
    const P& partial, const D& tangent)
{
#ifdef XAD_BRANCH_FREE_TANGENTS
    return maskedTerm(tangent, D(partial * tangent));
#else
    return tangent == D() ? D() : D(partial * tangent);
#endif
}

template <class P, class D>
XAD_INLINE auto tangentTerm(const P& partial, const D& tangent) ->
    typename std::enable_if<!float_or_double<D>::value, decltype(partial * tangent)>::type
{
    return partial * tangent;
}

template <class P, class T, std::size_t N>
XAD_INLINE Vec<T, N> tangentTerm(const P& partial, const Vec<T, N>& tangent)
{
    Vec<T, N> r(tangent);
    for (std::size_t i = 0; i < N; ++i) r[i] = tangentTerm(partial, tangent[i]);
    return r;
}

#ifdef XAD_BRANCH_FREE_TANGENTS
// B-byte vectors of doubles (V) and of lane masks (M) that may be unaligned. Written out per size:
// GCC 7 drops aligned(8) when vector_size depends on a template parameter, giving aligned loads.
template <std::size_t B>
struct PackedLanes;

template <>
struct PackedLanes<16>
{
    typedef double V __attribute__((vector_size(16), aligned(8), may_alias));
    typedef long long M __attribute__((vector_size(16), aligned(8), may_alias));
};

#ifdef __AVX__
template <>
struct PackedLanes<32>
{
    typedef double V __attribute__((vector_size(32), aligned(8), may_alias));
    typedef long long M __attribute__((vector_size(32), aligned(8), may_alias));
};
#endif

// lanes in blocks of B bytes from i on, with packed compares and ANDs; returns the next lane
template <std::size_t B, class P, std::size_t N>
XAD_INLINE std::size_t maskedTerms(const P& partial, const Vec<double, N>& tangent,
                                   Vec<double, N>& r, std::size_t i)
{
    typedef typename PackedLanes<B>::V V;
    typedef typename PackedLanes<B>::M M;
    const std::size_t L = B / sizeof(double);
    for (; i + L <= N; i += L)
    {
        const V t = *reinterpret_cast<const V*>(&tangent[i]);
        *reinterpret_cast<V*>(&r[i]) = (V)((t != 0) & (M)(double(partial) * t));
    }
    return i;
}

template <class P, std::size_t N>
XAD_INLINE Vec<double, N> tangentTerm(const P& partial, const Vec<double, N>& tangent)
{
    Vec<double, N> r(tangent);
    std::size_t i = 0;
#ifdef __AVX__
    i = maskedTerms<32>(partial, tangent, r, i);
#endif
    i = maskedTerms<16>(partial, tangent, r, i);
    if (i < N)
        r[i] = maskedTerm(tangent[i], double(partial * tangent[i]));
    return r;
}
#endif

// tangentTerm, or partial * tangent for ops with finite partials (HasFinitePartials<Op>())
template <class P, class D>
XAD_INLINE auto tangentTerm(const P& partial, const D& tangent, std::true_type)
    -> decltype(partial * tangent)
{
    return partial * tangent;
}

template <class P, class D>
XAD_INLINE auto tangentTerm(const P& partial, const D& tangent, std::false_type)
    -> decltype(tangentTerm(partial, tangent))
{
    return tangentTerm(partial, tangent);
}

// the tangent of v = a / b: (ta - v * tb) / b, where 1 / b * ta would be 0 * inf for b = 0
template <class S, class D>
XAD_INLINE D quotientTangent(const S& b, const S& v, const D& ta, const D& tb)
{
    return D((ta - v * tb) / b);
}

template <class S, class T, std::size_t N>
XAD_INLINE typename std::enable_if<!float_or_double<S>::value, Vec<T, N>>::type quotientTangent(
    const S& b, const S& v, const Vec<T, N>& ta, const Vec<T, N>& tb)
{
    Vec<T, N> r(ta);
    for (std::size_t i = 0; i < N; ++i) r[i] = quotientTangent(b, v, ta[i], tb[i]);
    return r;
}

// 2^64 for a subnormal b (1 / b overflows), else 1. GCC compiles a ?: here to a branch, so there it
// is a packed compare and blend (which it still folds for a constant b, unlike SSE intrinsics).
template <class S>
XAD_INLINE S reciprocalScale(S b)
{
#ifdef XAD_BRANCH_FREE_TANGENTS
    typedef S V __attribute__((vector_size(16)));
    typedef decltype(V() < V()) M;
    const M m = V{std::fabs(b)} < V{std::numeric_limits<S>::min()};
    return V((m & M(V{S(18446744073709551616.0)})) | (~m & M(V{S(1)})))[0];
#else
    return std::fabs(b) < std::numeric_limits<S>::min() ? S(18446744073709551616.0) : S(1);
#endif
}

// one division for all lanes: (ta - v * tb) * (1 / b), matching the per-lane division for
// b = 0, inf and NaN; a subnormal b (1 / b overflows) is scaled by 2^64 first, the result back
template <class S, class T, std::size_t N>
XAD_INLINE typename std::enable_if<float_or_double<S>::value, Vec<T, N>>::type quotientTangent(
    const S& b, const S& v, const Vec<T, N>& ta, const Vec<T, N>& tb)
{
    Vec<T, N> r(ta);
    const S scale = reciprocalScale(b);
    const S rb = S(1) / (b * scale);
    for (std::size_t i = 0; i < N; ++i) r[i] = (ta[i] - v * tb[i]) * rb * scale;
    return r;
}

}  // namespace detail

template <class T, std::size_t N>
struct DerivativesTraits
{
    using type = Vec<T, N>;
};

template <class T>
struct DerivativesTraits<T, 1>
{
    using type = T;
};

}  // namespace xad

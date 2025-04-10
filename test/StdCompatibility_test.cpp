#include <XAD/Complex.hpp>
#include <XAD/StdCompatibility.hpp>
#include <XAD/XAD.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;

namespace
{

template <class T>
struct is_complex : public std::false_type
{
};

template <class T>
struct is_complex<std::complex<T>> : public std::true_type
{
};

}  // namespace

TEST(StdCompatibility, canUseStdMath)
{
    xad::AD x = 0.2;
    auto xd = x.getValue();
    xad::AD x2 = 1.2;
    double x2d = 1.2;
    int n = 0;

    // check that there is a getValue member, i.e. it keeps the XAD type, and that
    // it matches the standard double version
    EXPECT_THAT(std::abs(x).getValue(), DoubleNear(std::abs(xd), 1e-9));
    EXPECT_THAT(std::fabs(x).getValue(), DoubleNear(std::fabs(xd), 1e-9));
    EXPECT_THAT(std::min(x, x).getValue(), DoubleNear(std::min(xd, xd), 1e-9));
    EXPECT_THAT(std::fmin(x, x).getValue(), DoubleNear(std::fmin(xd, xd), 1e-9));
    EXPECT_THAT(std::max(x, x).getValue(), DoubleNear(std::max(xd, xd), 1e-9));
    EXPECT_THAT(std::fmax(x, x).getValue(), DoubleNear(std::fmax(xd, xd), 1e-9));

    EXPECT_THAT(std::ceil(x).getValue(), DoubleNear(std::ceil(xd), 1e-9));
    EXPECT_THAT(std::floor(x).getValue(), DoubleNear(std::floor(xd), 1e-9));
    EXPECT_THAT(std::trunc(x).getValue(), DoubleNear(std::trunc(xd), 1e-9));
    EXPECT_THAT(std::round(x).getValue(), DoubleNear(std::round(xd), 1e-9));
    EXPECT_THAT(std::lround(x), Eq(std::lround(xd)));
    EXPECT_THAT(std::lround(x * 2.1), Eq(std::lround(xd * 2.1)));
    EXPECT_THAT(std::lround(-x), Eq(std::lround(-xd)));
    EXPECT_THAT(std::remainder(x, x).getValue(), DoubleNear(std::remainder(xd, xd), 1e-9));

    EXPECT_THAT(std::fmod(x, x).getValue(), DoubleNear(std::fmod(xd, xd), 1e-9));
    EXPECT_THAT(std::remquo(x, x, &n).getValue(), DoubleNear(std::remquo(xd, xd, &n), 1e-9));
    EXPECT_THAT(std::nextafter(x, 2.0 * x).getValue(),
                DoubleNear(std::nextafter(xd, 2.0 * xd), 1e-9));

    EXPECT_THAT(std::sin(x).getValue(), DoubleNear(std::sin(xd), 1e-9));
    EXPECT_THAT(std::cos(x).getValue(), DoubleNear(std::cos(xd), 1e-9));
    EXPECT_THAT(std::tan(x).getValue(), DoubleNear(std::tan(xd), 1e-9));
    EXPECT_THAT(std::asin(x).getValue(), DoubleNear(std::asin(xd), 1e-9));
    EXPECT_THAT(std::acos(x).getValue(), DoubleNear(std::acos(xd), 1e-9));
    EXPECT_THAT(std::atan(x).getValue(), DoubleNear(std::atan(xd), 1e-9));
    EXPECT_THAT(std::cosh(x).getValue(), DoubleNear(std::cosh(xd), 1e-9));
    EXPECT_THAT(std::sinh(x).getValue(), DoubleNear(std::sinh(xd), 1e-9));
    EXPECT_THAT(std::tanh(x).getValue(), DoubleNear(std::tanh(xd), 1e-9));
    EXPECT_THAT(std::acosh(x2).getValue(), DoubleNear(std::acosh(x2d), 1e-9));
    EXPECT_THAT(std::asinh(x).getValue(), DoubleNear(std::asinh(xd), 1e-9));
    EXPECT_THAT(std::atanh(x).getValue(), DoubleNear(std::atanh(xd), 1e-9));
    EXPECT_THAT(std::atan2(x, x).getValue(), DoubleNear(std::atan2(xd, xd), 1e-9));
    EXPECT_THAT(std::hypot(x, x).getValue(), DoubleNear(std::hypot(xd, xd), 1e-9));

    EXPECT_THAT(std::exp(x).getValue(), DoubleNear(std::exp(xd), 1e-9));
    EXPECT_THAT(std::log(x).getValue(), DoubleNear(std::log(xd), 1e-9));
    EXPECT_THAT(std::log10(x).getValue(), DoubleNear(std::log10(xd), 1e-9));
    EXPECT_THAT(std::log2(x).getValue(), DoubleNear(std::log2(xd), 1e-9));
    EXPECT_THAT(std::expm1(x).getValue(), DoubleNear(std::expm1(xd), 1e-9));
    EXPECT_THAT(std::exp2(x).getValue(), DoubleNear(std::exp2(xd), 1e-9));
    EXPECT_THAT(std::log1p(x).getValue(), DoubleNear(std::log1p(xd), 1e-9));
    EXPECT_THAT(std::sqrt(x).getValue(), DoubleNear(std::sqrt(xd), 1e-9));
    EXPECT_THAT(std::cbrt(x).getValue(), DoubleNear(std::cbrt(xd), 1e-9));
    EXPECT_THAT(std::pow(x, x).getValue(), DoubleNear(std::pow(xd, xd), 1e-9));
    EXPECT_THAT(std::erf(x).getValue(), DoubleNear(std::erf(xd), 1e-9));
    EXPECT_THAT(std::erfc(x).getValue(), DoubleNear(std::erfc(xd), 1e-9));
    EXPECT_THAT(std::scalbn(x, 2).getValue(), DoubleNear(std::scalbn(xd, 2), 1e-9));

    EXPECT_THAT(std::ldexp(x, 3).getValue(), DoubleNear(std::ldexp(xd, 3), 1e-9));
    int ex = 0, ex2 = 0;
    EXPECT_THAT(std::frexp(x, &ex).getValue(), DoubleNear(std::frexp(xd, &ex2), 1e-9));
    EXPECT_THAT(ex, Eq(ex2));

    double iptr = -1.0, iptr2 = -1.0;
    EXPECT_THAT(std::modf(x, &iptr).getValue(), DoubleNear(std::modf(xd, &iptr2), 1e-9));
    EXPECT_THAT(iptr, Eq(iptr2));

    EXPECT_THAT(std::isfinite(x), Eq(std::isfinite(xd)));
    EXPECT_THAT(std::isinf(x), Eq(std::isinf(xd)));
    EXPECT_THAT(std::isnan(x), Eq(std::isnan(xd)));
    EXPECT_THAT(std::isnormal(x), Eq(std::isnormal(xd)));
    EXPECT_THAT(std::signbit(x), Eq(std::signbit(xd)));
    EXPECT_THAT(std::fpclassify(x), Eq(std::fpclassify(xd)));
    EXPECT_THAT(std::ilogb(x), Eq(std::ilogb(xd)));
    EXPECT_THAT(std::copysign(x, -x), Eq(std::copysign(xd, -xd)));

    // complex
    EXPECT_THAT(std::real(x).getValue(), DoubleNear(std::real(xd), 1e-9));
    EXPECT_THAT(std::imag(x).getValue(), DoubleNear(std::imag(xd), 1e-9));
    EXPECT_THAT(std::arg(x).getValue(), DoubleNear(std::arg(xd), 1e-9));
    EXPECT_THAT(std::norm(x).getValue(), DoubleNear(std::norm(xd), 1e-9));

    // complex stuff that can have conflicts with std:: version when used on doubles,
    // Note: VS 2017 makes std::proj(double) and std::conj(double) a double rather than complex,
    // which is not according to the standard. But we have to put the same behaviour here
    auto zp = std::proj(x);
    auto zp_s = std::proj(xd);
    static_assert(is_complex<decltype(zp)>::value == is_complex<decltype(zp_s)>::value,
                  "std::proj result type not matching in their complex/real types");
    EXPECT_THAT(std::real(zp).getValue(), DoubleNear(std::real(zp_s), 1e-9));
    EXPECT_THAT(std::imag(zp).getValue(), DoubleNear(std::imag(zp_s), 1e-9));

    auto zc = std::conj(x);
    auto zc_s = std::conj(xd);
    static_assert(is_complex<decltype(zc)>::value == is_complex<decltype(zc_s)>::value,
                  "std::conj result type not matching in their complex/real types");
    EXPECT_THAT(std::real(zc).getValue(), DoubleNear(std::real(zc_s), 1e-9));
    EXPECT_THAT(std::imag(zc).getValue(), DoubleNear(std::imag(zc_s), 1e-9));

    auto zpol = std::polar(x, x);
    auto zpol_s = std::polar(xd, xd);
    EXPECT_THAT(zpol.real().getValue(), DoubleNear(zpol_s.real(), 1e-9));
    EXPECT_THAT(zpol.imag().getValue(), DoubleNear(zpol_s.imag(), 1e-9));
}

template <class T>
class StdCompatibilityTempl : public ::testing::Test
{
};

typedef ::testing::Types<xad::AD, xad::FAD, xad::AReal<xad::AReal<double>>,
                         xad::FReal<xad::AReal<double>>, xad::AReal<xad::FReal<double>>,
                         xad::FReal<xad::FReal<double>>>
    test_types;

TYPED_TEST_SUITE(StdCompatibilityTempl, test_types);

TYPED_TEST(StdCompatibilityTempl, NumericLimits)
{
    typedef typename xad::ExprTraits<TypeParam>::nested_type nested;

    // make sure it can be used in constexpr
    constexpr auto tmin = std::numeric_limits<TypeParam>::min();
    constexpr auto tmax = std::numeric_limits<TypeParam>::max();
    constexpr auto tlowest = std::numeric_limits<TypeParam>::lowest();
    constexpr auto teps = std::numeric_limits<TypeParam>::epsilon();
    constexpr auto tround = std::numeric_limits<TypeParam>::round_error();
    constexpr auto tden = std::numeric_limits<TypeParam>::denorm_min();

    XAD_UNUSED_VARIABLE(tmin);
    XAD_UNUSED_VARIABLE(tmax);
    XAD_UNUSED_VARIABLE(tlowest);
    XAD_UNUSED_VARIABLE(teps);
    XAD_UNUSED_VARIABLE(tround);
    XAD_UNUSED_VARIABLE(tden);

    static_assert(std::is_same<decltype(tmin), const nested>::value,
                  "numeric limits return underlying floating point type");

    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::min())),
                Eq(std::numeric_limits<nested>::min()));
    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::max())),
                Eq(std::numeric_limits<nested>::max()));
    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::lowest())),
                Eq(std::numeric_limits<nested>::lowest()));
    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::epsilon())),
                Eq(std::numeric_limits<nested>::epsilon()));
    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::round_error())),
                Eq(std::numeric_limits<nested>::round_error()));
    EXPECT_THAT(xad::value(xad::value(std::numeric_limits<TypeParam>::denorm_min())),
                Eq(std::numeric_limits<nested>::denorm_min()));

    EXPECT_THAT(std::numeric_limits<TypeParam>::is_specialized,
                Eq(std::numeric_limits<nested>::is_specialized));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_signed,
                Eq(std::numeric_limits<nested>::is_signed));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_integer,
                Eq(std::numeric_limits<nested>::is_integer));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_exact,
                Eq(std::numeric_limits<nested>::is_exact));
    EXPECT_THAT(std::numeric_limits<TypeParam>::has_infinity,
                Eq(std::numeric_limits<nested>::has_infinity));
    EXPECT_THAT(std::numeric_limits<TypeParam>::has_quiet_NaN,
                Eq(std::numeric_limits<nested>::has_quiet_NaN));
    EXPECT_THAT(std::numeric_limits<TypeParam>::has_signaling_NaN,
                Eq(std::numeric_limits<nested>::has_signaling_NaN));
    EXPECT_THAT(std::numeric_limits<TypeParam>::has_denorm,
                Eq(std::numeric_limits<nested>::has_denorm));
    EXPECT_THAT(std::numeric_limits<TypeParam>::has_denorm_loss,
                Eq(std::numeric_limits<nested>::has_denorm_loss));
    EXPECT_THAT(std::numeric_limits<TypeParam>::round_style,
                Eq(std::numeric_limits<nested>::round_style));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_iec559,
                Eq(std::numeric_limits<nested>::is_iec559));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_bounded,
                Eq(std::numeric_limits<nested>::is_bounded));
    EXPECT_THAT(std::numeric_limits<TypeParam>::is_modulo,
                Eq(std::numeric_limits<nested>::is_modulo));
    EXPECT_THAT(std::numeric_limits<TypeParam>::digits, Eq(std::numeric_limits<nested>::digits));
    EXPECT_THAT(std::numeric_limits<TypeParam>::digits10,
                Eq(std::numeric_limits<nested>::digits10));
    EXPECT_THAT(std::numeric_limits<TypeParam>::max_digits10,
                Eq(std::numeric_limits<nested>::max_digits10));
    EXPECT_THAT(std::numeric_limits<TypeParam>::radix, Eq(std::numeric_limits<nested>::radix));
    EXPECT_THAT(std::numeric_limits<TypeParam>::min_exponent,
                Eq(std::numeric_limits<nested>::min_exponent));
    EXPECT_THAT(std::numeric_limits<TypeParam>::min_exponent10,
                Eq(std::numeric_limits<nested>::min_exponent10));
    EXPECT_THAT(std::numeric_limits<TypeParam>::max_exponent10,
                Eq(std::numeric_limits<nested>::max_exponent10));
    EXPECT_THAT(std::numeric_limits<TypeParam>::traps, Eq(std::numeric_limits<nested>::traps));
    EXPECT_THAT(std::numeric_limits<TypeParam>::tinyness_before,
                Eq(std::numeric_limits<nested>::tinyness_before));
}

TYPED_TEST(StdCompatibilityTempl, Hashing)
{
    using T = TypeParam;
    using Tbase = typename xad::ExprTraits<T>::nested_type;

    T x = 42.0;
    Tbase xbase = xad::value(xad::value(x));

    auto hash = std::hash<T>{}(x);
    auto hash_base = std::hash<Tbase>{}(xbase);

    EXPECT_THAT(hash, Eq(hash_base));
}

// https://github.com/auto-differentiation/xad/pull/164#issuecomment-2775730529
#if !defined(_MSC_VER ) || _MSC_VER < 1941
TYPED_TEST(StdCompatibilityTempl, Traits)
{
    static_assert(std::is_floating_point<TypeParam>::value, "active real should be floating point");
    static_assert(std::is_arithmetic<TypeParam>::value, "active real should be arithmetic");
    static_assert(!std::is_pod<TypeParam>::value, "active type is not POD");
    static_assert(std::is_convertible<TypeParam, TypeParam>::value, "convertible to itself");
    static_assert(std::is_convertible<double, TypeParam>::value, "doubles are convertible");
    static_assert(std::is_convertible<int, TypeParam>::value, "integers are convertible");
    static_assert(!std::is_convertible<TypeParam, int>::value, "not implicitly convertible to int");
    static_assert(!std::is_convertible<TypeParam, long long>::value,
                  "not implicitly convertible to long long");
    static_assert(!std::is_convertible<TypeParam, char>::value,
                  "not implicitly convertible to char");
    static_assert(std::is_integral<TypeParam>::value == false, "not an integral type");
    static_assert(std::is_fundamental<TypeParam>::value == false, "not fundamental");
    static_assert(!std::is_scalar<TypeParam>::value,
                  "it's not a scalar type - would cause issues with constexpr etc");
    static_assert(std::is_object<TypeParam>::value, "it's an object type");
    static_assert(std::is_compound<TypeParam>::value, "it's compound");
    static_assert(!std::is_trivial<TypeParam>::value, "it's not a trivial type");
    // forward or forward over forward is trivally copyable
    constexpr bool fwd =
        xad::ExprTraits<TypeParam>::isForward &&
        (xad::ExprTraits<typename xad::ExprTraits<TypeParam>::scalar_type>::isForward ||
         !xad::ExprTraits<typename xad::ExprTraits<TypeParam>::scalar_type>::isExpr);
#if !(defined(__GNUC__) && __GNUC__ < 5) || defined(__clang__)
    static_assert(std::is_trivially_copyable<TypeParam>::value == fwd, "trivially copyable");
#endif
    static_assert(std::is_trivially_destructible<TypeParam>::value == fwd,
                  "trivially destructable for fwd mode");
}
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
TYPED_TEST(StdCompatibilityTempl, TraitsTemplateVars)
{
    static_assert(std::is_floating_point_v<TypeParam>, "active real should be floating point");
    static_assert(std::is_arithmetic_v<TypeParam>, "active real should be arithmetic");
    static_assert(!std::is_pod_v<TypeParam>, "active type is not POD");
    static_assert(std::is_convertible_v<TypeParam, TypeParam>, "convertible to itself");
    static_assert(std::is_convertible_v<double, TypeParam>, "doubles are convertible");
    static_assert(std::is_convertible_v<int, TypeParam>, "integers are convertible");
    static_assert(!std::is_convertible_v<TypeParam, int>, "not implicitly convertible to int");
    static_assert(!std::is_convertible_v<TypeParam, long long>,
                  "not implicitly convertible to long long");
    static_assert(!std::is_convertible_v<TypeParam, char>, "not implicitly convertible to char");
    static_assert(std::is_integral_v<TypeParam> == false, "not an integral type");
    static_assert(std::is_fundamental_v<TypeParam> == false, "not fundamental");
    static_assert(!std::is_scalar_v<TypeParam>,
                  "it's not a scalar type - would cause issues with constexpr etc");
    static_assert(std::is_object_v<TypeParam>, "it's an object type");
    static_assert(std::is_compound_v<TypeParam>, "it's compound");
    static_assert(!std::is_trivial_v<TypeParam>, "it's not a trivial type");
    // forward or forward over forward is trivally copyable
    constexpr bool fwd =
        xad::ExprTraits<TypeParam>::isForward &&
        (xad::ExprTraits<typename xad::ExprTraits<TypeParam>::scalar_type>::isForward ||
         !xad::ExprTraits<typename xad::ExprTraits<TypeParam>::scalar_type>::isExpr);
#if !(defined(__GNUC__) && __GNUC__ < 5) || defined(__clang__)
    static_assert(std::is_trivially_copyable_v<TypeParam> == fwd, "trivially copyable");
#endif
    static_assert(std::is_trivially_destructible_v<TypeParam> == fwd,
                  "trivially destructable for fwd mode");
}
#endif

template <class T>
class StdCompatibilityConstexprTempl : public ::testing::Test
{
};

typedef ::testing::Types<xad::FAD, xad::FReal<xad::FReal<double>>> constexpr_test_types;

#if !(_MSC_VER && _MSC_VER <= 1900)  // VS 2015 doesn't implement constexpr objects correctly

TYPED_TEST_SUITE(StdCompatibilityConstexprTempl, constexpr_test_types);

TYPED_TEST(StdCompatibilityConstexprTempl, NumericLimitsConstexpr)
{
    constexpr TypeParam t_xx = 1.0;
    constexpr TypeParam t_min = std::numeric_limits<TypeParam>::min();
    constexpr TypeParam t_max = std::numeric_limits<TypeParam>::max();
    constexpr TypeParam t_lowest = std::numeric_limits<TypeParam>::lowest();
    constexpr TypeParam t_eps = std::numeric_limits<TypeParam>::epsilon();
    constexpr TypeParam t_den = std::numeric_limits<TypeParam>::denorm_min();
    constexpr TypeParam t_inf = std::numeric_limits<TypeParam>::infinity();
    constexpr TypeParam t_nan = std::numeric_limits<TypeParam>::quiet_NaN();
    constexpr TypeParam t_snan = std::numeric_limits<TypeParam>::signaling_NaN();
    constexpr TypeParam t_round = std::numeric_limits<TypeParam>::round_error();

    XAD_UNUSED_VARIABLE(t_xx);
    XAD_UNUSED_VARIABLE(t_min);
    XAD_UNUSED_VARIABLE(t_max);
    XAD_UNUSED_VARIABLE(t_lowest);
    XAD_UNUSED_VARIABLE(t_eps);
    XAD_UNUSED_VARIABLE(t_den);
    XAD_UNUSED_VARIABLE(t_inf);
    XAD_UNUSED_VARIABLE(t_nan);
    XAD_UNUSED_VARIABLE(t_snan);
    XAD_UNUSED_VARIABLE(t_round);
}

#endif

TEST(StdCompatibility, StdMinWithSizeWorks)
{
    // this has been added as the call to hashtable_policy.h in GCC showed a failure with this,
    // it called xad::min, which instantiates an AD type with int scalar type,
    // which is not supported
    const auto max_width = std::min<size_t>(sizeof(size_t), 8);
    const auto min_width = std::max<size_t>(sizeof(size_t), 8);

    static_assert(std::is_same<decltype(max_width), const size_t>::value, "mismatch in type");
    static_assert(std::is_same<decltype(min_width), const size_t>::value, "mismatch in type");

    EXPECT_THAT(max_width, Le(8));
    EXPECT_THAT(min_width, Ge(8));
}

TEST(StdCompatibility, UseWithRandomDistribution)
{
    std::normal_distribution<xad::AReal<double>> dst(1.0, 2.0);

    EXPECT_THAT(dst.mean().value(), DoubleEq(1.0));
}

TEST(StdCompatibility, UseInVectorAndFill)
{
    std::vector<xad::AReal<double>> v(3);
    std::fill(v.begin(), v.end(), xad::AReal<double>(1.));

    EXPECT_THAT(v, ElementsAre(1., 1., 1.));
}

// https://github.com/auto-differentiation/xad/issues/158
TEST(StdCompatibility, CopysignWindowsAReal)
{
    xad::AD x(1.2);
    xad::AD y(-0.5);
    xad::AD one(1.);

    auto r1 = std::copysign(1.2, y);
    auto r2 = std::copysign(x, -0.5);
    auto r3 = std::copysign(x, y);
    // with expressions
    auto r4 = std::copysign(1.2, y * one);
    auto r5 = std::copysign(x * one, -0.5);
    auto r6 = std::copysign(x * one, y);
    auto r7 = std::copysign(x * one, y * one);
    auto r8 = std::copysign(x, y * one);

    EXPECT_EQ(xad::value(r1), -1.2);
    EXPECT_EQ(xad::value(r2), -1.2);
    EXPECT_EQ(xad::value(r3), -1.2);
    EXPECT_EQ(xad::value(r4), -1.2);
    EXPECT_EQ(xad::value(r5), -1.2);
    EXPECT_EQ(xad::value(r6), -1.2);
    EXPECT_EQ(xad::value(r7), -1.2);
    EXPECT_EQ(xad::value(r8), -1.2);
}

TEST(StdCompatibility, CopysignWindowsFReal)
{
    xad::FAD x(1.2);
    xad::FAD y(-0.5);
    xad::FAD one(1.);

    auto r1 = std::copysign(1.2, y);
    auto r2 = std::copysign(x, -0.5);
    auto r3 = std::copysign(x, y);
    // with expressions
    auto r4 = std::copysign(1.2, y * one);
    auto r5 = std::copysign(x * one, -0.5);
    auto r6 = std::copysign(x * one, y);
    auto r7 = std::copysign(x * one, y * one);
    auto r8 = std::copysign(x, y * one);

    EXPECT_EQ(xad::value(r1), -1.2);
    EXPECT_EQ(xad::value(r2), -1.2);
    EXPECT_EQ(xad::value(r3), -1.2);
    EXPECT_EQ(xad::value(r4), -1.2);
    EXPECT_EQ(xad::value(r5), -1.2);
    EXPECT_EQ(xad::value(r6), -1.2);
    EXPECT_EQ(xad::value(r7), -1.2);
    EXPECT_EQ(xad::value(r8), -1.2);
}

template <typename T, bool IsArithmetic>
    class TestTemplate {
      public:
        TestTemplate() {
            if constexpr (IsArithmetic) {
                lambda_ = [](T x) { return x + 1; };
            } else {
                lambda_ = [](T x) { return x; };
            }
        }

        T apply(T value) const {
            return lambda_(value);
        }

      private:
        std::function<T(T)> lambda_;
    };

    TEST(StdCompatibility, MSVCLambdaTemplateSpecialization)
    {
        TestTemplate<int, std::is_arithmetic<int>::value> arithmeticTest;
        EXPECT_EQ(arithmeticTest.apply(1), 2);
    
        TestTemplate<std::vector<int>, std::is_arithmetic<std::vector<int>>::value> nonArithmeticTest;
        std::vector<int> input = {1, 2, 3};
        EXPECT_EQ(nonArithmeticTest.apply(input), input);
    
        TestTemplate<xad::AReal<double>, std::is_arithmetic<xad::AReal<double>>::value> aRealTest;
        xad::AReal<double> x(1.0);
        EXPECT_EQ(aRealTest.apply(x), x + 1.0);
    
        TestTemplate<xad::FReal<double>, std::is_arithmetic<xad::FReal<double>>::value> fRealTest;
        xad::FReal<double> y(1.0);
        EXPECT_EQ(fRealTest.apply(y), y + 1.0);
    }
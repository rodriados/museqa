/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Generic n-dimensional geometry data structures and functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/guard.hpp>

#include <museqa/thirdparty/fmtlib.h>
#include <museqa/thirdparty/reflector.h>
#include <museqa/thirdparty/supertuple.h>

MUSEQA_BEGIN_NAMESPACE

namespace geometry
{

// Ignoring warnings for anonymous structs which are prohibited by ISO C++ standard,
// but is supported by all major compilers. We exploit these anonymous structs to
// allow coordinates to be accessed by their different axis-names within distinct
// coordinated systems at the same time.
MUSEQA_DISABLE_GCC_WARNING_BEGIN("-Wpedantic")
MUSEQA_DISABLE_CLANG_WARNING_BEGIN("-Wgnu-anonymous-struct")

    /**
     * The representation of a generic point in a D-dimensional coordinate system.
     * Due to the purposedly arbitrary number of dimensions, there are no special
     * axis or dimension identifiers to coordinates higher than 4 dimensions.
     * @tparam D The point's coordinate system dimensionality.
     * @tparam T The type for the coordinate system dimensions' values.
     * @since 1.0
     */
    template <size_t D, typename T = int64_t>
    union point_t
    {
        typedef T element_t;

        T dim[D] {};

        MUSEQA_CONSTEXPR static size_t dimensionality = D;

        static_assert(dimensionality > 0, "a point must be at least 1-dimensional");
        static_assert(std::is_arithmetic_v<T>, "a point must have arithmetic dimension type");

        MUSEQA_CONSTEXPR point_t() noexcept = default;
        MUSEQA_CONSTEXPR point_t(const point_t&) noexcept = default;
        MUSEQA_CONSTEXPR point_t(point_t&&) noexcept = default;

        MUSEQA_INLINE point_t& operator=(const point_t&) noexcept = default;
        MUSEQA_INLINE point_t& operator=(point_t&&) noexcept = default;

        /**
         * Initializes a point from a generic list of values.
         * @tparam U The list of given value types.
         * @param values The values to initialize a point with.
         */
        template <
            typename ...U
          , typename = std::enable_if_t<utility::all(std::is_convertible_v<U, T>...)>>
        MUSEQA_CUDA_CONSTEXPR point_t(U&&... values) noexcept
          : dim {static_cast<T>(values)...}
        {}

        /**
         * Retrieves a dimension value from a point's coordinate value.
         * @param id The axis id of the requested dimension value.
         * @return The point's dimension value.
         */
        MUSEQA_CUDA_CONSTEXPR T operator[](ptrdiff_t id) const MUSEQA_SAFE_EXCEPT
        {
            guard((size_t) id < dimensionality, "point dimension is out of range");
            return dim[id];
        }
    };

    /**
     * The representation of a point in a generic 1-dimensional coordinate system.
     * As a 1-dimensional value is a scalar, such coordinate system can be interpreted
     * as and seamlessly converted to its underlying scalar type.
     * @tparam T The type for the coordinate system dimensions' values.
     * @since 1.0
     */
    template <typename T>
    union point_t<1, T>
    {
        typedef T element_t;

        T dim[1] {};
        struct { T a; };
        struct { T i; };
        struct { T x; };

        MUSEQA_CONSTEXPR static size_t dimensionality = 1;

        static_assert(std::is_arithmetic_v<T>, "a point must have arithmetic dimension type");

        MUSEQA_CONSTEXPR point_t() noexcept = default;
        MUSEQA_CONSTEXPR point_t(const point_t&) noexcept = default;
        MUSEQA_CONSTEXPR point_t(point_t&&) noexcept = default;

        MUSEQA_INLINE point_t& operator=(const point_t&) noexcept = default;
        MUSEQA_INLINE point_t& operator=(point_t&&) noexcept = default;

        /**
         * Initializes a point from a generic value.
         * @tparam U The type of the given value.
         * @param value The value to initialize a point with.
         */
        template <
            typename U
          , typename = std::enable_if_t<std::is_convertible_v<U, T>>>
        MUSEQA_CUDA_CONSTEXPR point_t(U&& value) noexcept
          : dim {static_cast<T>(value)}
        {}

        /**
         * Retrieves a dimension value from a point's coordinate value.
         * @param id The axis id of the requested dimension value.
         * @return The point's dimension value.
         */
        MUSEQA_CUDA_CONSTEXPR T operator[](ptrdiff_t id) const MUSEQA_SAFE_EXCEPT
        {
            guard((size_t) id < dimensionality, "point dimension is out of range");
            return dim[id];
        }

        /**
         * Seamlessly converts the point into its scalar value, a particularity
         * of points on 1-dimensional coordinate systems.
         * @return The point's 1-dimensional scalar value.
         */
        MUSEQA_CUDA_CONSTEXPR operator T() const noexcept
        {
            return dim[0];
        }
    };

    /**
     * The representation of a point in a generic 2-dimensional coordinate system.
     * At this dimensionality, the coordinates in the system might be referred to
     * as or named with a pair of different 2-letter combinations.
     * @tparam T The type for the coordinate system dimensions' values.
     * @since 1.0
     */
    template <typename T>
    union point_t<2, T>
    {
        typedef T element_t;

        T dim[2] {};
        struct { T a, b; };
        struct { T i, j; };
        struct { T x, y; };

        MUSEQA_CONSTEXPR static size_t dimensionality = 2;

        static_assert(std::is_arithmetic_v<T>, "a point must have arithmetic dimension type");

        MUSEQA_CONSTEXPR point_t() noexcept = default;
        MUSEQA_CONSTEXPR point_t(const point_t&) noexcept = default;
        MUSEQA_CONSTEXPR point_t(point_t&&) noexcept = default;

        MUSEQA_INLINE point_t& operator=(const point_t&) noexcept = default;
        MUSEQA_INLINE point_t& operator=(point_t&&) noexcept = default;

        /**
         * Initializes a point from a generic list of values.
         * @tparam U The list of given value types.
         * @param values The values to initialize a point with.
         */
        template <
            typename ...U
          , typename = std::enable_if_t<utility::all(std::is_convertible_v<U, T>...)>>
        MUSEQA_CUDA_CONSTEXPR point_t(U&&... values) noexcept
          : dim {static_cast<T>(values)...}
        {}

        /**
         * Retrieves a dimension value from a point's coordinate value.
         * @param id The axis id of the requested dimension value.
         * @return The point's dimension value.
         */
        MUSEQA_CUDA_CONSTEXPR T operator[](ptrdiff_t id) const MUSEQA_SAFE_EXCEPT
        {
            guard((size_t) id < dimensionality, "point dimension is out of range");
            return dim[id];
        }
    };

    /**
     * The representation of a point in a generic 3-dimensional coordinate system.
     * At this dimensionality, the coordinates in the system might be referred to
     * as or named with a triple of different 3-letter combinations.
     * @tparam T The type for the coordinate system dimensions' values.
     * @since 1.0
     */
    template <typename T>
    union point_t<3, T>
    {
        typedef T element_t;

        T dim[3] {};
        struct { T a, b, c; };
        struct { T i, j, k; };
        struct { T x, y, z; };

        MUSEQA_CONSTEXPR static size_t dimensionality = 3;

        static_assert(std::is_arithmetic_v<T>, "a point must have arithmetic dimension type");

        MUSEQA_CONSTEXPR point_t() noexcept = default;
        MUSEQA_CONSTEXPR point_t(const point_t&) noexcept = default;
        MUSEQA_CONSTEXPR point_t(point_t&&) noexcept = default;

        MUSEQA_INLINE point_t& operator=(const point_t&) noexcept = default;
        MUSEQA_INLINE point_t& operator=(point_t&&) noexcept = default;

        /**
         * Initializes a point from a generic list of values.
         * @tparam U The list of given value types.
         * @param values The values to initialize a point with.
         */
        template <
            typename ...U
          , typename = std::enable_if_t<utility::all(std::is_convertible_v<U, T>...)>>
        MUSEQA_CUDA_CONSTEXPR point_t(U&&... values) noexcept
          : dim {static_cast<T>(values)...}
        {}

        /**
         * Retrieves a dimension value from a point's coordinate value.
         * @param id The axis id of the requested dimension value.
         * @return The point's dimension value.
         */
        MUSEQA_CUDA_CONSTEXPR T operator[](ptrdiff_t id) const MUSEQA_SAFE_EXCEPT
        {
            guard((size_t) id < dimensionality, "point dimension is out of range");
            return dim[id];
        }
    };

MUSEQA_DISABLE_CLANG_WARNING_END("-Wgnu-anonymous-struct")
MUSEQA_DISABLE_GCC_WARNING_END("-Wpedantic")

    /*
     * Deduction guides for a generic multi-dimensional point.
     * @since 1.0
     */
    template <typename ...T> point_t(T...) -> point_t<sizeof...(T), std::common_type_t<T...>>;

    /**
     * The equality operator for two points of equal dimensionality.
     * @tparam D The points' coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point instance.
     * @param q The second point instance.
     * @return Are both points equal?
     */
    template <size_t D, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator==(
        const point_t<D, T>& p
      , const point_t<D, U>& q
    ) noexcept {
        return supertuple::foldl(
            supertuple::zipwith(
                supertuple::tie(p.dim)
              , supertuple::tie(q.dim)
              , utility::equal)
          , utility::andL);
    }

    /**
     * The equality operator for points of different dimensionalities.
     * @tparam P The first point's coordinate system dimensionality.
     * @tparam Q The second point's coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @return Are both points equal?
     */
    template <size_t P, size_t Q, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator==(
        const point_t<P, T>&
      , const point_t<Q, U>&
    ) noexcept {
        return false;
    }

    /**
     * The inequality operator for two generic points.
     * @tparam P The first point's coordinate system dimensionality.
     * @tparam Q The second point's coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point instance.
     * @param q The second point instance.
     * @return Are the points different?
     */
    template <size_t P, size_t Q, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator!=(
        const point_t<P, T>& p
      , const point_t<Q, U>& q
    ) noexcept {
        return !geometry::operator==(p, q);
    }

    /**
     * The operator for the sum of two vector-like points.
     * @tparam D The points' coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point's instance.
     * @param q The second point's instance.
     * @return The resulting vector-like point sum.
     */
    template <size_t D, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR auto operator+(
        const point_t<D, T>& p
      , const point_t<D, U>& q
    ) noexcept {
        return supertuple::forward(
            supertuple::zipwith(
                supertuple::tie(p.dim)
              , supertuple::tie(q.dim)
              , utility::add)
          , SUPERTUPLE_FORWARD_CONSTRUCTOR(point_t));
    }

    /**
     * The operator for the subtraction of two vector-like points.
     * @tparam D The points' coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point's instance.
     * @param q The second point's instance.
     * @return The resulting vector-like point subtraction.
     */
    template <size_t D, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR auto operator-(
        const point_t<D, T>& p
      , const point_t<D, U>& q
    ) noexcept {
        return supertuple::forward(
            supertuple::zipwith(
                supertuple::tie(p.dim)
              , supertuple::tie(q.dim)
              , utility::sub)
          , SUPERTUPLE_FORWARD_CONSTRUCTOR(point_t));
    }

    /**
     * The operator for scalar product of a vector-like point.
     * @tparam P The points' coordinate system dimensionality.
     * @tparam T The point's dimension type.
     * @tparam S The scalar type.
     * @param p The point's instance.
     * @param scalar The scalar value.
     * @return The resulting vector-like point.
     */
    template <size_t P, typename T, typename S>
    MUSEQA_CUDA_CONSTEXPR auto operator*(
        const point_t<P, T>& p
      , const S& scalar
    ) noexcept {
        return supertuple::forward(
            supertuple::apply(
                supertuple::tie(p.dim)
              , utility::mul, scalar)
          , SUPERTUPLE_FORWARD_CONSTRUCTOR(point_t));
    }

    /**
     * The commutative operator for scalar product of a vector-like point.
     * @tparam P The points' coordinate system dimensionality.
     * @tparam T The point's dimension type.
     * @tparam S The scalar type.
     * @param scalar The scalar value.
     * @param p The point's instance.
     * @return The resulting vector-like point.
     */
    template <size_t P, typename T, typename S>
    MUSEQA_CUDA_CONSTEXPR auto operator*(
        const S& scalar
      , const point_t<P, T>& p
    ) noexcept {
        return supertuple::forward(
            supertuple::apply(
                supertuple::tie(p.dim)
              , utility::rmul, scalar)
          , SUPERTUPLE_FORWARD_CONSTRUCTOR(point_t));
    }

    /**
     * The distance operator for two generic points.
     * @tparam D The points' coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point instance.
     * @param q The second point instance.
     * @return The euclidean distance between the points.
     */
    template <size_t D, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR double distance(
        const point_t<D, T>& p
      , const point_t<D, U>& q
    ) noexcept {
        return sqrt(
            supertuple::foldl(
                supertuple::zipwith(
                    supertuple::tie(p.dim)
                  , supertuple::tie(q.dim)
                  , [](T a, U b) { return pow(b - a, 2.0); })
              , utility::add, double(0)));
    }

    /**
     * The dot-product operator for two vector-like points.
     * @tparam D The points' coordinate system dimensionality.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point instance.
     * @param q The second point instance.
     * @return The resulting dot-product value.
     */
    template <size_t D, typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR auto dot(
        const point_t<D, T>& p
      , const point_t<D, U>& q
    ) noexcept {
        return supertuple::foldl(
            supertuple::zipwith(
                supertuple::tie(p.dim)
              , supertuple::tie(q.dim)
              , utility::mul)
          , utility::add, T(0));
    }

    /**
     * The cross-product operator for 3-dimensional vector-like points.
     * @tparam T The first point's dimension type.
     * @tparam U The second point's dimension type.
     * @param p The first point instance.
     * @param q The second point instance.
     * @return The resulting cross-product vector-like point.
     */
    template <typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR auto cross(
        const point_t<3, T>& p
      , const point_t<3, U>& q
    ) noexcept {
        return point_t {
            (p.y * q.z - p.z * q.y)
          , (p.z * q.x - p.x * q.z)
          , (p.x * q.y - p.y * q.x)
        };
    }

    /**
     * The length operator for a vector-like point.
     * @tparam P The point's coordinate system dimensionality.
     * @tparam T The point's dimension type.
     * @param p The point instance.
     * @return The length of the vector-like point.
     */
    template <size_t P, typename T>
    MUSEQA_CUDA_CONSTEXPR double length(const point_t<P, T>& p) noexcept
    {
        return sqrt(
            supertuple::foldl(
                supertuple::apply(
                    supertuple::tie(p.dim)
                  , pow, double(2.0))
              , utility::add, double(0)));
    }

    /**
     * The operator for normalization of a vector-like point.
     * @tparam P The point's coordinate system dimensionality.
     * @tparam T The point's dimension type.
     * @param p The point instance.
     * @return The resulting normalized vector-like point.
     */
    template <size_t P, typename T>
    MUSEQA_CUDA_CONSTEXPR auto normalize(const point_t<P, T>& p) noexcept
    {
        constexpr auto lambda = [](double x, double len) -> double
            { return len > 0 ? x / len : 0; };
        return supertuple::forward(
            supertuple::apply(
                supertuple::tie(p.dim)
              , lambda, geometry::length(p))
          , SUPERTUPLE_FORWARD_CONSTRUCTOR(point_t));
    }
}

MUSEQA_END_NAMESPACE

#ifndef MUSEQA_AVOID_REFLECTOR

/**
 * Explicitly defines the reflection for a generic point.
 * @tparam P The point's coordinate system dimensionality.
 * @tparam T The point's dimension type.
 * @since 1.0
 */
template <size_t P, typename T>
struct reflector::provider_t<MUSEQA_NAMESPACE::geometry::point_t<P, T>>
{
    typedef MUSEQA_NAMESPACE::geometry::point_t<P, T> target_t;

    /**
     * Provides the internal members of a point for reflection.
     * @return The point members reflection configuration.
     */
    MUSEQA_CONSTEXPR static auto provide() noexcept
    {
        return reflector::provide(&target_t::dim);
    }
};

#endif

#ifndef MUSEQA_AVOID_FMTLIB

/**
 * Implements a string formatter for a generic point type.
 * @tparam P The point's coordinate system dimensionality.
 * @tparam T The point's dimension type.
 * @since 1.0
 */
template <size_t P, typename T>
struct fmt::formatter<MUSEQA_NAMESPACE::geometry::point_t<P, T>>
{
    typedef MUSEQA_NAMESPACE::geometry::point_t<P, T> target_t;

    /**
     * Evaluates the formatter's parsing context.
     * @tparam C The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename C>
    MUSEQA_CONSTEXPR auto parse(C& ctx) const -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the point into a printable string.
     * @tparam F The formatting context type.
     * @param point The point to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_t& point, F& ctx) const -> decltype(ctx.out())
    {
        return fmt::format_to(
            ctx.out(), "({})"
          , fmt::join(point.dim, point.dim + target_t::dimensionality, ", ")
        );
    }
};

#endif

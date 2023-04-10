/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A vector data structure in a n-dimensional geometric space.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cmath>
#include <cstdint>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/utility/reflection.hpp>
#include <museqa/geometry/coordinate.hpp>
#include <museqa/geometry/point.hpp>

#include <museqa/thirdparty/fmtlib.h>

MUSEQA_BEGIN_NAMESPACE

namespace geometry
{
    /**
     * Represents a simple generic geometric vector in a D-dimensional space.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's coordinates' type.
     * @since 1.0
     */
    template <size_t D, typename T = int64_t>
    struct vector_t : public geometry::point_t<D, T>
    {
        __host__ __device__ inline constexpr vector_t() noexcept = default;
        __host__ __device__ inline constexpr vector_t(const vector_t&) noexcept = default;
        __host__ __device__ inline constexpr vector_t(vector_t&&) noexcept = default;

        using geometry::point_t<D, T>::point_t;

        /**
         * Instantiates a new vector from a point instance.
         * @tparam U The foreign point's coordinates' type.
         * @param point The foreign point instance to create a new vector from.
         */
        template <typename U>
        __host__ __device__ inline constexpr vector_t(const geometry::point_t<D, U>& point) noexcept
          : geometry::point_t<D, T> (point)
        {}

        __host__ __device__ inline vector_t& operator=(const vector_t&) noexcept = default;
        __host__ __device__ inline vector_t& operator=(vector_t&&) noexcept = default;
    };

    /*
     * Deduction guides for generic vector types.
     * @since 1.0
     */
    template <typename T, typename ...U> vector_t(T, U...) -> vector_t<sizeof...(U) + 1, T>;
    template <typename T, typename ...U> vector_t(utility::tuple_t<T, U...>) -> vector_t<sizeof...(U) + 1, T>;
    template <typename T, size_t N> vector_t(utility::ntuple_t<T, N>) -> vector_t<N, T>;

    /**
     * The operator for the sum of two vectors.
     * @tparam D The vectors' dimensionality.
     * @tparam T The first vector's coordinates' type.
     * @tparam U The second vector's coordinates' type.
     * @param a The first vector's instance.
     * @param b The second vector's instance.
     * @return The resulting vector sum.
     */
    template <size_t D, typename T, typename U>
    __host__ __device__ inline constexpr auto operator+(
        const vector_t<D, T>& a
      , const vector_t<D, U>& b
    ) noexcept {
        return vector_t(utility::zipwith(utility::add, utility::tie(a.value), utility::tie(b.value)));
    }

    /**
     * The operator for the subtraction of two vectors.
     * @tparam D The vectors' dimensionality.
     * @tparam T The first vector's coordinates' type.
     * @tparam U The second vector's coordinates' type.
     * @param a The first vector's instance.
     * @param b The second vector's instance.
     * @return The resulting vector subtraction.
     */
    template <size_t D, typename T, typename U>
    __host__ __device__ inline constexpr auto operator-(
        const vector_t<D, T>& a
      , const vector_t<D, U>& b
    ) noexcept {
        return vector_t(utility::zipwith(utility::sub, utility::tie(a.value), utility::tie(b.value)));
    }

    /**
     * The operator for a vector's scalar product.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's coordinates' type.
     * @tparam S The scalar type.
     * @param v The vector's instance.
     * @param scalar The scalar value.
     * @return The resulting vector.
     */
    template <size_t D, typename T, typename S>
    __host__ __device__ inline constexpr auto operator*(
        const vector_t<D, T>& v
      , const S& scalar
    ) noexcept {
        return vector_t(utility::apply(utility::mul, utility::tie(v.value), scalar));
    }

    /**
     * The operator for a vector's scalar product with commutativity assumed.
     * @tparam S The scalar type.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's coordinates' type.
     * @param scalar The scalar value.
     * @param v The vector's instance.
     * @return The resulting vector.
     */
    template <typename S, size_t D, typename T>
    __host__ __device__ inline constexpr auto operator*(
        const S& scalar
      , const vector_t<D, T>& v
    ) noexcept {
        return vector_t(utility::apply(utility::rmul, utility::tie(v.value), scalar));
    }

    /**
     * The operator for the dot product of two vectors.
     * @tparam D The vectors' dimensionality.
     * @tparam T The first vector's coordinates' type.
     * @tparam U The second vector's coordinates' type.
     * @param a The first vector's instance.
     * @param b The second vector's instance.
     * @return The resulting dot product value.
     */
    template <size_t D, typename T, typename U>
    __host__ __device__ inline constexpr auto dot(
        const vector_t<D, T>& a
      , const vector_t<D, U>& b
    ) noexcept {
        return utility::foldl(
            utility::add, T(0)
          , utility::zipwith(utility::mul, utility::tie(a.value), utility::tie(b.value))
        );
    }

    /**
     * The operator for the cross product of two 3-dimensional vectors.
     * @tparam T The first vector's coordinates' type.
     * @tparam U The second vector's coordinates' type.
     * @param a The first vector's instance.
     * @param b The second vector's instance.
     * @return The resulting vector.
     */
    template <typename T, typename U>
    __host__ __device__ inline constexpr auto cross(
        const vector_t<3, T>& a
      , const vector_t<3, U>& b
    ) noexcept {
        return vector_t {
            (a.y * b.z - a.z * b.y)
          , (a.z * b.x - a.x * b.z)
          , (a.x * b.y - a.y * b.x)
        };
    }

    /**
     * The operator for the length of a vector.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's coordinates' type.
     * @param v The vector's instance.
     * @return The resulting length value.
     */
    template <size_t D, typename T>
    __host__ __device__ inline constexpr double length(const vector_t<D, T>& v) noexcept
    {
        return sqrt(utility::foldl(
            utility::add, double(0)
          , utility::apply(pow, utility::tie(v.value), 2.0)
        ));
    }

    /**
     * The operator for the normalization of a vector.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's coordinates' type.
     * @param v The vector's instance.
     * @return The resulting normalized vector.
     */
    template <size_t D, typename T>
    __host__ __device__ inline constexpr auto normalize(const vector_t<D, T>& v) noexcept
    {
        return vector_t(utility::apply(
            [](double value, double length) { return length > 0.0 ? (value / length) : 0.0; }
          , utility::tie(v.value)
          , geometry::length(v)
        ));
    }
}

#if !defined(MUSEQA_AVOID_REFLECTION)

/**
 * Explicitly defines the reflector for a vector. Although a trivial type, a vector
 * cannot be automatically reflected over due to its inheritance.
 * @tparam D The vector's dimensionality.
 * @tparam T The vector's coordinates' type.
 * @since 1.0
 */
template <size_t D, typename T>
class utility::reflector_t<geometry::vector_t<D, T>>
  : public utility::reflector_t<geometry::coordinate_t<D, T>> {};

#endif

MUSEQA_END_NAMESPACE

#if !defined(MUSEQA_AVOID_FMTLIB)

/**
 * Implements a string formatter for a generic vector type, thus allowing vectors
 * to be seamlessly printed as scalar types.
 * @tparam D The vector's dimensionality.
 * @tparam T The vector's coordinates' type.
 * @since 1.0
 */
template <size_t D, typename T>
struct fmt::formatter<museqa::geometry::vector_t<D, T>>
  : fmt::formatter<museqa::geometry::coordinate_t<D, T>> {};

#endif

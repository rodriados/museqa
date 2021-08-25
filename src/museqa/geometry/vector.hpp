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

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma push
  #pragma diag_suppress = unrecognized_gcc_pragma
#endif

#include <fmt/format.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma pop
#endif

#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/geometry/point.hpp>

#if !defined(MUSEQA_AVOID_REFLECTION)
  #include <museqa/utility/reflection.hpp>
#endif

namespace museqa
{
    namespace geometry
    {
        /**
         * Represents a simple generic geometric vector in a D-dimensional space.
         * @tparam D The vector's dimensionality.
         * @tparam T The vector's dimensions' type.
         * @since 1.0
         */
        template <size_t D, typename T = int64_t>
        struct vector : public geometry::point<D, T>
        {
            __host__ __device__ inline constexpr vector() noexcept = default;
            __host__ __device__ inline constexpr vector(const vector&) noexcept = default;
            __host__ __device__ inline constexpr vector(vector&&) noexcept = default;

            using geometry::point<D, T>::point;

            __host__ __device__ inline vector& operator=(const vector&) noexcept = default;
            __host__ __device__ inline vector& operator=(vector&&) noexcept = default;
        };

        /**
         * The operator for the sum of two vectors.
         * @tparam D The vectors' dimensionality.
         * @tparam T The first vector's dimensions type.
         * @tparam U The second vector's dimensions type.
         * @param a The first vector's instance.
         * @param b The second vector's instance.
         * @return The resulting vector sum.
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr auto operator+(const vector<D, T>& a, const vector<D, U>& b) noexcept
        -> vector<D, decltype(utility::add(a[0], b[0]))>
        {
            return utility::zipwith(utility::add<T, U>, utility::tie(a.dim), utility::tie(b.dim));
        }

        /**
         * The operator for the subtraction of two vectors.
         * @tparam D The vectors' dimensionality.
         * @tparam T The first vector's dimensions type.
         * @tparam U The second vector's dimensions type.
         * @param a The first vector's instance.
         * @param b The second vector's instance.
         * @return The resulting vector subtraction.
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr auto operator-(const vector<D, T>& a, const vector<D, U>& b) noexcept
        -> vector<D, decltype(utility::sub(a[0], b[0]))>
        {
            return utility::zipwith(utility::sub<T, U>, utility::tie(a.dim), utility::tie(b.dim));
        }

        /**
         * The operator for a vector's scalar product.
         * @tparam D The vector's dimensionality.
         * @tparam T The vector's dimensions type.
         * @tparam S The scalar type.
         * @param v The vector's instance.
         * @param scalar The scalar value.
         * @return The resulting vector.
         */
        template <size_t D, typename T, typename S>
        __host__ __device__ inline constexpr auto operator*(const vector<D, T>& v, const S& scalar) noexcept
        -> vector<D, decltype(utility::mul(v[0], scalar))>
        {
            return utility::apply(utility::mul<T, S>, utility::tie(v.dim), scalar);
        }

        /**
         * The operator for a vector's scalar product with commutativity assumed.
         * @tparam S The scalar type.
         * @tparam D The vector's dimensionality.
         * @tparam T The vector's dimensions type.
         * @param scalar The scalar value.
         * @param v The vector's instance.
         * @return The resulting vector.
         */
        template <typename S, size_t D, typename T>
        __host__ __device__ inline constexpr auto operator*(const S& scalar, const vector<D, T>& v) noexcept
        -> vector<D, decltype(utility::mul(scalar, v[0]))>
        {
            return geometry::operator*(v, scalar);
        }

        /**
         * The operator for the dot product of two vectors.
         * @tparam D The vectors' dimensionality.
         * @tparam T The first vector's dimensions type.
         * @tparam U The second vector's dimensions type.
         * @param a The first vector's instance.
         * @param b The second vector's instance.
         * @return The resulting dot product value.
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr auto dot(const vector<D, T>& a, const vector<D, U>& b) noexcept
        -> decltype(utility::mul(a[0], b[0]))
        {
            return utility::foldl(
                utility::add<decltype(utility::mul(a[0], b[0]))>, 0
              , utility::zipwith(utility::mul<T, U>, utility::tie(a.dim), utility::tie(b.dim))
            );
        }

        /**
         * The operator for the cross product of two 3-dimensional vectors.
         * @tparam T The first vector's dimensions type.
         * @tparam U The second vector's dimensions type.
         * @param a The first vector's instance.
         * @param b The second vector's instance.
         * @return The resulting vector.
         */
        template <typename T, typename U>
        __host__ __device__ inline constexpr auto cross(const vector<3, T>& a, const vector<3, U>& b) noexcept
        -> vector<3, decltype(a[0] * b[0])>
        {
            return {(a.y * b.z - a.z * b.y), (a.z * b.x - a.x * b.z), (a.x * b.y - a.y * b.x)};
        }

        /**
         * The operator for the length of a vector.
         * @tparam D The vector's dimensionality.
         * @tparam T The vector's dimensions type.
         * @param v The vector's instance.
         * @return The resulting length value.
         */
        template <size_t D, typename T>
        __host__ __device__ inline constexpr auto length(const vector<D, T>& v) noexcept -> double
        {
            return sqrt(utility::foldl(utility::add<double>, 0.0, utility::apply(pow, utility::tie(v.dim), 2.0)));
        }

        /**
         * The operator for the normalization of a vector.
         * @tparam D The vector's dimensionality.
         * @tparam T The vector's dimensions type.
         * @param v The vector's instance.
         * @return The resulting normalized vector.
         */
        template <size_t D, typename T>
        __host__ __device__ inline constexpr auto normalize(const vector<D, T>& v) noexcept -> vector<D, double>
        {
            return utility::apply(
                [](double dim, double length) { return length > 0.0 ? (dim / length) : 0.0; }
              , utility::tie(v.dim)
              , geometry::length(v)
            );
        }
    }

  #if !defined(MUSEQA_AVOID_REFLECTION)
    /**
     * Explicitly defines the reflector for a vector. Although a trivial type, a
     * vector cannot be automatically reflected over due to its inheritance.
     * @tparam D The vector's dimensionality.
     * @tparam T The vector's dimensions' type.
     * @since 1.0
     */
    template <size_t D, typename T>
    class utility::reflector<geometry::vector<D, T>>
      : public utility::reflector<geometry::point<D, T>>
    {};
  #endif
}

/**
 * Implements a string formatter for a generic vector type, thus allowing vectors
 * to be seamlessly printed as scalar types.
 * @tparam D The vector's dimensionality.
 * @tparam T The vector's dimensions' type.
 * @since 1.0
 */
template <size_t D, typename T>
struct fmt::formatter<museqa::geometry::vector<D, T>> : fmt::formatter<museqa::geometry::point<D, T>>
{};
/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An n-dimensional geometric point data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

#include <museqa/environment.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma push
  #pragma diag_suppress = unrecognized_gcc_pragma
#endif

#include <fmt/format.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #pragma pop
#endif

#include <museqa/exception.hpp>
#include <museqa/utility.hpp>
#include <museqa/utility/tuple.hpp>
#include <museqa/utility/indexer.hpp>

#include <museqa/geometry/coordinate.hpp>

#if !defined(MUSEQA_AVOID_REFLECTION)
  #include <museqa/utility/reflection.hpp>
#endif

namespace museqa
{
    namespace geometry
    {
        /**
         * Represents a simple point for a generic D-dimensional space.
         * @tparam D The point's dimensionality.
         * @tparam T The point's coordinates' type.
         * @since 1.0
         */
        template <size_t D, typename T = int64_t>
        class point : public geometry::coordinate<D, T>
        {
          private:
            typedef geometry::coordinate<D, T> underlying_type;

          public:
            using typename underlying_type::coordinate_type;
            using underlying_type::dimensionality;

          public:
            __host__ __device__ inline constexpr point() noexcept = default;
            __host__ __device__ inline constexpr point(const point&) noexcept = default;
            __host__ __device__ inline constexpr point(point&&) noexcept = default;

            /**
             * Instantiates a new point from a general list of coordinate values.
             * @tparam U The list of coordinate parameter types.
             * @param value The point's coordinates' values.
             */
            template <
                class ...U
              , class = typename std::enable_if<utility::all(std::is_convertible<U, coordinate_type>{}...)>::type
              , class = typename std::enable_if<dimensionality == sizeof...(U)>::type
            >
            __host__ __device__ inline constexpr point(const U&... value)
              : underlying_type {static_cast<coordinate_type>(value)...}
            {}

            /**
             * Instantiates a new point from a tuple.
             * @tparam I The sequence tuple's indeces.
             * @tparam U The tuple's contents types.
             * @param tuple The tuple to build a point from.
             */
            template <size_t ...I, typename ...U>
            __host__ __device__ inline constexpr point(const utility::tuple<utility::indexer<I...>, U...>& tuple)
              : point {tuple.template get<I>()...}
            {}

            /**
             * Instantiates a new point from a foreign point instance.
             * @tparam U The foreign point's coordinates' type.
             * @param other The foreign point instance.
             */
            template <typename U>
            __host__ __device__ inline constexpr point(const point<D, U>& other)
              : point {typename utility::indexer<D>::type (), other}
            {}

            __host__ __device__ inline point& operator=(const point&) noexcept = default;
            __host__ __device__ inline point& operator=(point&&) noexcept = default;

            /**
             * Gives direct access to a point's coordinate value.
             * @param offset The requested point coordinate offset.
             * @return The point's requested coordinate value.
             */
            __host__ __device__ inline coordinate_type& operator[](ptrdiff_t offset) noexcept(!safe)
            {
                museqa::ensure((size_t) offset < dimensionality, "point coordinate out of range");
                return this->value[offset];
            }

            /**
             * Gives direct access to a const-qualified point's coordinate value.
             * @param offset The requested point coordinate offset.
             * @return The point's requested const-qualified coordinate value.
             */
            __host__ __device__ inline const coordinate_type& operator[](ptrdiff_t offset) const noexcept(!safe)
            {
                museqa::ensure((size_t) offset < dimensionality, "point coordinate out of range");
                return this->value[offset];
            }

          private:
            /**
             * Instantiates a new point from a foreign point and an indexer helper.
             * @tparam I The foreign point's coordinates sequence indeces.
             * @tparam P The foreign point's instance type.
             * @param other The foreign point to build a new point from.
             */
            template <size_t ...I, typename P>
            __host__ __device__ inline constexpr point(utility::indexer<I...>, const P& other)
              : point {other.value[I]...}
            {}
        };

        /**
         * The distance operator for two generic points.
         * @tparam D The points' dimensionality value.
         * @tparam T The first point's coordinate type.
         * @tparam U The second point's coordinate type.
         * @param a The first point instance.
         * @param b The second point instance.
         * @return The Euclidean distance between the points.
         */
        template <size_t D, typename T, typename U>
        __host__ __device__ inline constexpr double distance(const point<D, T>& a, const point<D, U>& b) noexcept
        {
            return sqrt(utility::foldl(
                utility::add<double>, 0.0
              , utility::zipwith(
                    [](const T& a, const U& b) { return pow(b - a, 2.0); }
                  , utility::tie(a.value)
                  , utility::tie(b.value)
                )
            ));
        }
    }

  #if !defined(MUSEQA_AVOID_REFLECTION)
    /**
     * Explicitly defines the reflector for a point. Although a trivial type, a
     * point cannot be automatically reflected over due to its customized constructors.
     * @tparam D The point's dimensionality.
     * @tparam T The point's coordinates' type.
     * @since 1.0
     */
    template <size_t D, typename T>
    class utility::reflector<geometry::point<D, T>>
      : public utility::reflector<geometry::coordinate<D, T>>
    {};
  #endif
}

/**
 * Implements a string formatter for a generic point type, thus allowing points
 * to be seamlessly printed as scalar types.
 * @tparam D The point's dimensionality.
 * @tparam T The point's coordinates' type.
 * @since 1.0
 */
template <size_t D, typename T>
struct fmt::formatter<museqa::geometry::point<D, T>> : fmt::formatter<museqa::geometry::coordinate<D, T>>
{};

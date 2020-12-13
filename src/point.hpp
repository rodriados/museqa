/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a n-dimensional point data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include "tuple.hpp"
#include "utils.hpp"
#include "exception.hpp"

namespace museqa
{
    /**
     * Represents a D-dimensional point.
     * @tparam D The point's dimensionality.
     * @tparam T The type of dimension values.
     * @since 0.1.1
     */
    template <size_t D, typename T = size_t>
    union point
    {
        using dimension_type = T;                       /// The point's dimensions type.
        static constexpr size_t dimensionality = D;     /// The point's dimensionality.

        static_assert(dimensionality > 0, "points must be at least 1-dimensional");

        dimension_type dim[dimensionality];             /// The point's dimensions values.

        __host__ __device__ inline point() noexcept = default;
        __host__ __device__ inline point(const point&) noexcept = default;
        __host__ __device__ inline point(point&&) noexcept = default;

        /**
         * Constructs a new point instance.
         * @tparam U The point's dimension values' types.
         * @param value The point's dimension values.
         */
        template <
                class ...U
            ,   class = typename std::enable_if<utils::all(std::is_convertible<U, dimension_type>()...)>::type
            ,   class = typename std::enable_if<sizeof...(U) == dimensionality>::type
            >
        __host__ __device__ inline constexpr point(const U&... value)
        :   dim {static_cast<dimension_type>(value)...}
        {}

        /**
         * Constructs a new point instance from a tuple.
         * @tparam I The sequence of tuple indeces.
         * @tparam U The tuple types.
         * @param tup The tuple from which the new point will be created.
         * @return The new point instance.
         */
        template <size_t ...I, typename ...U>
        __host__ __device__ inline constexpr point(const detail::tuple::base<indexer<I...>, U...>& tup) noexcept
        :   point {detail::tuple::get<I>(tup)...}
        {}

        __host__ __device__ inline point& operator=(const point&) noexcept = default;
        __host__ __device__ inline point& operator=(point&&) noexcept = default;

        /**
         * Allows direct access to the value of one of the point's dimensions.
         * @param offset The requested dimension offset identifier.
         * @return The requested dimension's value.
         */
        __host__ __device__ inline dimension_type operator[](ptrdiff_t offset) const
        {
            enforce(size_t(offset) < dimensionality, "requested dimension does not exist");
            return dim[offset];
        }
    };

    /**
     * Represents a 1-dimensional point.
     * @tparam T The type of dimension values.
     * @since 0.1.1
     */
    template <typename T>
    union point<1, T>
    {
        using dimension_type = T;                       /// The point's dimensions type.
        static constexpr size_t dimensionality = 1;     /// The point's dimensionality.

        dimension_type x;                               /// The point's dimension alias.
        dimension_type value;                           /// The point's uni-dimensional value.
        dimension_type dim[dimensionality];             /// The point's dimension value.

        __host__ __device__ inline point() noexcept = default;
        __host__ __device__ inline point(const point&) noexcept = default;
        __host__ __device__ inline point(point&&) noexcept = default;

        /**
         * Constructs a new point instance.
         * @tparam U The point's dimension values' types.
         * @param value The point's dimension values.
         */
        template <
                class U
            ,   class = typename std::enable_if<std::is_convertible<U, dimension_type>::value>::type
            >
        __host__ __device__ inline constexpr point(const U& value)
        :   dim {static_cast<dimension_type>(value)}
        {}

        /**
         * Constructs a new point instance from a tuple.
         * @tparam U The tuple's content type.
         * @param tup The tuple from which the new point will be created.
         * @return The new point instance.
         */
        template <typename U>
        __host__ __device__ inline constexpr point(const tuple<U>& tup) noexcept
        :   point {tup.template get<0>()}
        {}

        __host__ __device__ inline point& operator=(const point&) noexcept = default;
        __host__ __device__ inline point& operator=(point&&) noexcept = default;

        /**
         * Allows direct access to the value of one of the point's dimensions.
         * @param offset The requested dimension offset identifier.
         * @return The requested dimension's value.
         */
        __host__ __device__ inline dimension_type operator[](ptrdiff_t offset) const
        {
            enforce(size_t(offset) < dimensionality, "requested dimension does not exist");
            return dim[offset];
        }

        /**
         * Allows a uni-dimensional point value to be represented as a value.
         * @return The uni-dimensional point value.
         */
        __host__ __device__ inline operator dimension_type() const noexcept
        {
            return value;
        }
    };

    /**
     * Represents a 2-dimensional point.
     * @tparam T The type of dimension values.
     * @since 0.1.1
     */
    template <typename T>
    union point<2, T>
    {
        using dimension_type = T;                       /// The point's dimensions type.
        static constexpr size_t dimensionality = 2;     /// The point's dimensionality.

        struct { dimension_type x, y; };                /// The point's dimensions aliases.
        dimension_type dim[dimensionality];             /// The point's dimensions values.

        __host__ __device__ inline point() noexcept = default;
        __host__ __device__ inline point(const point&) noexcept = default;
        __host__ __device__ inline point(point&&) noexcept = default;

        /**
         * Constructs a new point instance.
         * @tparam U The point's first dimension value type.
         * @tparam V The point's second dimension value type.
         * @param x The point's first dimension value.
         * @param y The point's second dimension value.
         */
        template <
                class U, class V
            ,   class = typename std::enable_if<std::is_convertible<U, dimension_type>::value>::type
            ,   class = typename std::enable_if<std::is_convertible<V, dimension_type>::value>::type
            >
        __host__ __device__ inline constexpr point(const U& x, const V& y)
        :   dim {static_cast<dimension_type>(x), static_cast<dimension_type>(y)}
        {}

        /**
         * Constructs a new point instance from a tuple.
         * @tparam U The point's first dimension value type.
         * @tparam V The point's second dimension value type.
         * @param tup The tuple from which the new point will be created.
         * @return The new point instance.
         */
        template <typename U, typename V>
        __host__ __device__ inline constexpr point(const tuple<U, V>& tup) noexcept
        :   point {detail::tuple::get<0>(tup), detail::tuple::get<1>(tup)}
        {}

        __host__ __device__ inline point& operator=(const point&) noexcept = default;
        __host__ __device__ inline point& operator=(point&&) noexcept = default;

        /**
         * Allows direct access to the value of one of the point's dimensions.
         * @param offset The requested dimension offset identifier.
         * @return The requested dimension's value.
         */
        __host__ __device__ inline dimension_type operator[](ptrdiff_t offset) const
        {
            enforce(size_t(offset) < dimensionality, "requested dimension does not exist");
            return dim[offset];
        }
    };

    /**#@+
     * Aliases for points of common dimensionalities.
     * @since 0.1.1
     */
    template <typename T = ptrdiff_t> using point1 = point<1, T>;
    template <typename T = ptrdiff_t> using point2 = point<2, T>;
    /**#@-*/
}

/**
 * Multiple Sequence Alignment space header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <point.hpp>
#include <tuple.hpp>
#include <utils.hpp>
#include <exception.hpp>
#include <transform.hpp>

namespace msa
{
    namespace detail
    {
        namespace space
        {
            /*
             * Forward-declaration of collapse and volume proxy functions. These
             * functions are available for use inside the space class.
             */
            template <size_t D, typename T, typename U>
            __host__ __device__ inline constexpr T collapse(const point<D, T>&, const point<D, U>&) noexcept;

            template <size_t D, typename T>
            __host__ __device__ inline constexpr T volume(const point<D, T>&) noexcept;
        }
    }

    /**
     * Represents a D-dimensional space.
     * @tparam D The space's dimensionality.
     * @tparam T The type of dimension values.
     * @tparam M The space coordinates transformer.
     * @since 0.1.1
     */
    template <size_t D, typename T = size_t, typename M = transform::linear<D>>
    struct space
    {
        using dimension_type = T;                       /// The space's dimensions type.
        using transformer_type = M;                     /// The space's coordinates transform.
        using point_type = point<D, T>;                 /// The space's point type.
        static constexpr size_t dimensionality = D;     /// The space's dimensionality.

        static_assert(dimensionality > 0, "spaces must be at least 1-dimensional");

        point_type dim;                                 /// The space's dimensions.

        __host__ __device__ inline constexpr space() noexcept = default;
        __host__ __device__ inline constexpr space(const space&) noexcept = default;
        __host__ __device__ inline constexpr space(space&&) noexcept = default;

        /**
         * Constructs a new space instance.
         * @tparam U The space's dimension values' types.
         * @param value The space's dimension values.
         */
        template <typename ...U>
        __host__ __device__ inline constexpr space(const U&... value) noexcept
        :   dim {transformer_type::shape(point_type {value...})}
        {}

        __host__ __device__ inline space& operator=(const space&) noexcept = default;
        __host__ __device__ inline space& operator=(space&&) noexcept = default;

        /**
         * Allows direct access to the value of one of the space's dimensions.
         * @param offset The requested dimension offset identifier.
         * @return The requested dimension's value.
         */
        __host__ __device__ inline dimension_type operator[](ptrdiff_t offset) const
        {
            enforce(size_t(offset) < dimensionality, "requested dimension does not exist");
            return dim[offset];
        }

        /**
         * Collapses a multidimensional space onto an 1-dimensional point value,
         * effectively mapping every point in space to a linear value.
         * @param target The target point to collapse the space onto.
         * @return The 1-dimensional collapsed linear value.
         */
        __host__ __device__ inline constexpr dimension_type collapse(const point_type& target) const noexcept
        {
            return direct(transformer_type::transform(dim, target));
        }

        /**
         * Collapses a multidimensional space onto an 1-dimensional untransformed
         * point value. This allows any space to be mapped linearly.
         * @param target The target point to collapse the space onto.
         * @return The 1-dimensional collapsed linear value.
         */
        __host__ __device__ inline constexpr dimension_type direct(const point_type& target) const noexcept
        {
            return detail::space::collapse(dim, target);
        }

        /**
         * Informs the space's projection dimensions. These dimensions, though,
         * may be different from those represented internally by this space.
         * @return The space's dimensions.
         */
        __host__ __device__ inline constexpr point_type dimension() const noexcept
        {
            return transformer_type::projection(dim);
        }

        /**
         * Informs the space's internal representation's shape dimensions.
         * @return The space's internal representation's dimensions.
         */
        __host__ __device__ inline constexpr point_type reprdim() const noexcept
        {
            return dim;
        }

        /**
         * Calculates the space's total volume.
         * @return The space volume.
         */
        __host__ __device__ inline constexpr dimension_type volume() const noexcept
        {
            return detail::space::volume(dim);
        }
    };

    namespace detail
    {
        namespace space
        {
            /**#@+
             * Maps a multidimensional space onto an 1-dimensional point value,
             * effectively assigning a linear value to every point in space.
             * @tparam D The point and space dimensionality.
             * @tparam T The space's dimensions type.
             * @tparam U The point's dimensions type.
             * @param space The dimensions of the space being collapsed.
             * @param target The point to collapse the space onto.
             * @return The resulting 1-dimensional point value.
             */
            template <typename T, typename U>
            __host__ __device__ inline constexpr T map(const point1<T>&, const point1<U>& target) noexcept
            {
                return T {target.x};
            }

            template <typename T, typename U>
            __host__ __device__ inline constexpr T map(const point2<T>& space, const point2<U>& target) noexcept
            {
                return T {target.x * space.y + target.y};
            }

            template <size_t D, typename T, typename U>
            __host__ __device__ inline constexpr T map(const point<D, T>& space, const point<D, U>& target) noexcept
            {
                using namespace utils;
                return foldl(add<T>, 0, zipwith(mul<T>, tie(target.dim), scanr(mul<T>, 1, tail(tie(space.dim)))));
            }
            /**#@-*/

            /**#@+
             * Calculates the total number of elements in a space of given dimensions.
             * @tparam D The space dimensionality.
             * @tparam T The space's dimensions types.
             * @return The space's total volume.
             */
            template <typename T>
            __host__ __device__ inline constexpr T count(const point1<T>& target) noexcept
            {
                return target.x;
            }

            template <typename T>
            __host__ __device__ inline constexpr T count(const point2<T>& target) noexcept
            {
                return target.x * target.y;
            }

            template <size_t D, typename T>
            __host__ __device__ inline constexpr T count(const point<D, T>& target) noexcept
            {
                using namespace utils;
                return foldl(mul<T>, 1, tie(target.dim));
            }
            /**#@-*/

            /**
             * Collapses a multidimensional space onto an 1-dimensional point value.
             * @tparam D The point and space dimensionality.
             * @tparam T The space's dimensions type.
             * @tparam U The point's dimensions type.
             * @param space The dimensions of the space being collapsed.
             * @param target The point to collapse the space onto.
             * @return The resulting 1-dimensional point value.
             */
            template <size_t D, typename T, typename U>
            __host__ __device__ inline constexpr T collapse(
                    const point<D, T>& space
                ,   const point<D, U>& target
                ) noexcept
            {
                return map(space, target);
            }

            /**
             * Calculates the given space's total volume.
             * @tparam D The space dimensionality.
             * @tparam T The space's dimensions types.
             * @param space The dimensions of the target space.
             * @return The space's total volume.
             */
            template <size_t D, typename T>
            __host__ __device__ inline constexpr T volume(const point<D, T>& space) noexcept
            {
                return count(space);
            }          
        }
    }
}

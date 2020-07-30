/**
 * Multiple Sequence Alignment space transformation header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include <point.hpp>
#include <utils.hpp>

namespace msa
{
    namespace transform
    {
        /**
         * Implements a spacial linear transformation. Effectively, this transformation
         * does not affect or change the space in any way.
         * @tparam D The space's dimensionality.
         * @since 0.1.1
         */
        template <size_t D>
        struct linear
        {
            /**
             * Applies the transformation to the shape of a space.
             * @tparam T The space's dimensions type.
             * @param target The original space shape.
             * @return The transformed space.
             */
            template <typename T>
            __host__ __device__ inline constexpr static auto shape(const point<D, T>& target) noexcept
            -> point<D, T>
            {
                return target;
            }

            /**
             * Applies the transformation to a given point on the given space.
             * @tparam T The space's dimensions type.
             * @param (ignored) The shape of the space applying the transformation.
             * @param target The target point to be transformed.
             * @return The transformed shape in space.
             */
            template <typename T>
            __host__ __device__ inline static auto transform(const point<D, T>&, const point<D, T>& target) noexcept
            -> point<D, T>
            {
                return target;
            }

            /**
             * Retrieves the space's projected shape from its transformed shape.
             * @tparam T The space's dimensions type.
             * @param shape The space's transformed shape.
             * @return The projected space shape.
             */
            template <typename T>
            __host__ __device__ inline constexpr static auto projection(const point<D, T>& shape) noexcept
            -> point<D, T>
            {
                return shape;
            }
        };

        /**
         * Implements a symmetric transformation. Effectively, this transformation
         * saves up almost half of the space occupied by a symmetric matrix. This
         * transformation can only be applied to 2-dimensional spaces.
         * @since 0.1.1
         */
        struct symmetric
        {
            /**
             * Applies the transformation to the shape of a space.
             * @tparam T The space's dimensions type.
             * @param target The original space shape.
             * @return The transformed space.
             */
            template <typename T>
            __host__ __device__ inline constexpr static point2<T> shape(const point2<T>& target) noexcept
            {
                return {(target.y >> 1) + (target.y & 1), target.y};
            }

            /**
             * Applies the transformation to a given point on the given space.
             * @tparam T The space's dimensions type.
             * @param space The shape of the space applying the transformation.
             * @param target The target point to be transformed.
             * @return The transformed shape in space.
             */
            template <typename T>
            __host__ __device__ inline static point2<T> transform(
                    const point2<T>& space
                ,   const point2<T>& target
                ) noexcept
            {
                const auto i = utils::max(target.x, target.y);
                const auto j = utils::min(target.x, target.y);

                const auto x = (i < space.x) ? i : space.y - i - 1;
                const auto y = (i < space.x) ? j : space.y - j - 1;
                return {x, y};
            }

            /**
             * Retrieves the space's projected shape from its transformed shape.
             * @tparam T The space's dimensions type.
             * @param shape The space's transformed shape.
             * @return The projected space shape.
             */
            template <typename T>
            __host__ __device__ inline constexpr static point2<T> projection(const point2<T>& shape) noexcept
            {
                return {shape.y, shape.y};
            }
        };
    }
}

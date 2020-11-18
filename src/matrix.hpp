/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a generic spatial storage matrix data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include "point.hpp"
#include "space.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "transform.hpp"

namespace museqa
{
    /**
     * Creates a general-purpose bi-dimensional buffer. The matrix stores all
     * of its data contiguously in memory.
     * @tparam E The matrix's buffer element type.
     * @tparam T The matrix's spacial transformation.
     * @since 0.1.1
     */
    template <typename E, typename T = transform::linear<2>>
    class matrix : protected buffer<E>
    {
        protected:
            using underlying_buffer = buffer<E>;
            using space_type = museqa::space<2, size_t, T>;

        public:
            using element_type = typename underlying_buffer::element_type;
            using pointer_type = typename underlying_buffer::pointer_type;
            using point_type = typename space_type::point_type;

        protected:
            space_type m_space;                         /// The matrix's space.

        public:
            __host__ __device__ inline matrix() noexcept = default;
            __host__ __device__ inline matrix(const matrix&) noexcept = default;
            __host__ __device__ inline matrix(matrix&&) noexcept = default;

            /**
             * Acquires the ownership of a raw matrix buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param space The space dimensions of buffer to acquire.
             */
            inline explicit matrix(element_type *ptr, const space_type& space)
            :   underlying_buffer {ptr, space.volume()}
            ,   m_space {space}
            {}

            /**
             * Acquires the ownership of a matrix buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param space The space dimensions of buffer to acquire.
             */
            __host__ __device__ inline explicit matrix(pointer_type&& ptr, const space_type& space)
            :   underlying_buffer {std::forward<decltype(ptr)>(ptr), space.volume()}
            ,   m_space {space}
            {}

            /**
             * Instantiates a new matrix from an already allocated buffer.
             * @param buf The pre-allocated matrix buffer.
             * @param space The matrix's space dimensions.
             */
            __host__ __device__ inline explicit matrix(const underlying_buffer& buf, const space_type& space)
            :   underlying_buffer {buf}
            ,   m_space {space}
            {}

            __host__ __device__ inline matrix& operator=(const matrix&) = default;
            __host__ __device__ inline matrix& operator=(matrix&&) = default;

            /**
             * Gives access to an element in the matrix.
             * @param offset The element's offset value.
             * @return The requested element.
             */
            __host__ __device__ inline element_type& operator[](const point_type& offset)
            {
                return underlying_buffer::operator[](m_space.collapse(offset));
            }

            /**
             * Gives access to a const-qualified element in the matrix.
             * @param offset The element's offset value.
             * @return The requested const-qualified element.
             */
            __host__ __device__ inline const element_type& operator[](const point_type& offset) const
            {
                return underlying_buffer::operator[](m_space.collapse(offset));
            }

            /**
             * Gives direct linear access to an element in matrix.
             * @param offset The element's offset value.
             * @return The requested element.
             */
            __host__ __device__ inline element_type& linear(const point_type& offset)
            {
                return underlying_buffer::operator[](m_space.direct(offset));
            }

            /**
             * Gives direct linear access to a const-qualified element in matrix.
             * @param offset The element's offset value.
             * @return The requested const-qualified element.
             */
            __host__ __device__ inline const element_type& linear(const point_type& offset) const
            {
                return underlying_buffer::operator[](m_space.direct(offset));
            }

            /**
             * Informs the matrix's projection dimensions.
             * @return The matrix's projected size.
             */
            __host__ __device__ inline point_type dimension() const noexcept
            {
                return m_space.dimension();
            }

            /**
             * Informs the matrix's internal representation dimensions.
             * @return The matrix's shape in memory.
             */
            __host__ __device__ inline point_type reprdim() const noexcept
            {
                return m_space.reprdim();
            }

            /**
             * Copies data from an existing matrix instance.
             * @param mat The target matrix to copy data from.
             * @return A newly created matrix instance.
             */
            static inline matrix copy(const matrix& mat) noexcept
            {
                return matrix {underlying_buffer::copy(mat), mat.m_space};
            }

            /**
             * Creates a new matrix of given space dimensions.
             * @param space The matrix's space dimensions.
             * @return The newly created matrix instance.
             */
            static inline matrix make(const space_type& space) noexcept
            {
                return matrix {pointer_type::make(space.volume()), space};
            }

            /**
             * Creates a new matrix of given dimensions with an allocator.
             * @param allocator The allocator to be used to new matrix.
             * @param space The matrix's space dimensions.
             * @return The newly created matrix instance.
             */
            static inline matrix make(const museqa::allocator& allocator, const space_type& space) noexcept
            {
                return matrix {pointer_type::make(allocator, space.volume()), space};
            }
    };

    /**
     * Implements a 2-dimensional symmetric matrix.
     * @tparam E The matrix's contents type.
     * @since 0.1.1
     */
    template <typename E>
    using symmatrix = matrix<E, transform::symmetric>;
}

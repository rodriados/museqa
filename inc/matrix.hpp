/**
 * Multiple Sequence Alignment matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-2020 Rodrigo Siqueira
 */
#pragma once

#include <buffer.hpp>
#include <cartesian.hpp>

namespace msa
{
    /**
     * Creates a general-purpose multi-dimensional buffer. The matrix stores all
     * its data contiguously in memory.
     * @tparam T The matrix's buffer element type.
     * @tparam D The matrix's dimensionality.
     * @since 0.1.1
     */
    template <typename T, size_t D = 2>
    class matrix : protected buffer<T>
    {
        static_assert(D >= 2, "matrices must be at least 2-dimensional");

        protected:
            using underlying_buffer = buffer<T>;    /// The matrix's underlying buffer.

        public:
            using element_type = typename buffer<T>::element_type;  /// The matrix's element type.
            using pointer_type = typename buffer<T>::pointer_type;  /// The matrix's pointer type.
            using cartesian_type = cartesian<D, size_t>;            /// The matrix's dimension value type.

            static constexpr size_t dimensionality = D;             /// The matrix's dimensionality.

        protected:
            cartesian_type m_dim;                                   /// The matrix's dimensional size.

        public:
            __host__ __device__ inline matrix() noexcept = default;
            __host__ __device__ inline matrix(const matrix&) noexcept = default;
            __host__ __device__ inline matrix(matrix&&) noexcept = default;

            /**
             * Acquires the ownership of a raw matrix buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param dim The size dimensions of buffer to acquire.
             */
            inline explicit matrix(element_type *ptr, const cartesian_type& dim)
            :   underlying_buffer {ptr, dim.volume()}
            ,   m_dim {dim}
            {}

            /**
             * Acquires the ownership of a matrix buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param dim The size dimensions of buffer to acquire.
             */
            __host__ __device__ inline explicit matrix(pointer_type&& ptr, const cartesian_type& dim)
            :   underlying_buffer {std::forward<decltype(ptr)>(ptr), dim.volume()}
            ,   m_dim {dim}
            {}

            /**
             * Instantiates a new matrix from an already allocated buffer.
             * @param buf The pre-allocated matrix buffer.
             * @param dim The matrix's dimensions.
             */
            __host__ __device__ inline explicit matrix(const underlying_buffer& buf, const cartesian_type& dim)
            :   underlying_buffer {buf}
            ,   m_dim {dim}
            {}

            inline matrix& operator=(const matrix&) = default;
            inline matrix& operator=(matrix&&) = default;

            /**
             * Gives access to an element in the matrix.
             * @param offset The offset's values.
             * @return The requested element.
             */
            __host__ __device__ inline element_type& operator[](const cartesian_type& offset)
            {
                return underlying_buffer::operator[](m_dim.collapse(offset));
            }

            /**
             * Gives access to a const-qualified element in the matrix.
             * @param offset The offset's values.
             * @return The requested constant element.
             */
            __host__ __device__ inline const element_type& operator[](const cartesian_type& offset) const
            {
                return underlying_buffer::operator[](m_dim.collapse(offset));
            }

            /**
             * Informs the matrix's dimensional sizes.
             * @return The matrix's dimensions.
             */
            __host__ __device__ inline const cartesian_type& dimension() const noexcept
            {
                return m_dim;
            }

            /**
             * Copies data from an existing matrix instance.
             * @param mat The target matrix to copy data from.
             * @return A newly created matrix instance.
             */
            static inline matrix copy(const matrix& mat) noexcept
            {
                return matrix {underlying_buffer::copy(mat), mat.m_dim};
            }

            /**
             * Creates a new matrix of given size.
             * @param dim The matrix's dimension sizes.
             * @return The newly created matrix instance.
             */
            static inline matrix make(const cartesian_type& dim) noexcept
            {
                return matrix {pointer_type::make(dim.volume()), dim};
            }

            /**
             * Creates a new matrix of given size with an allocator.
             * @param allocator The allocator to be used to new matrix.
             * @param dim The matrix's dimension sizes.
             * @return The newly created matrix instance.
             */
            static inline matrix make(const msa::allocator& allocator, const cartesian_type& dim) noexcept
            {
                return matrix {pointer_type::make(allocator, dim.volume()), dim};
            }
    };
}

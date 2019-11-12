/**
 * Multiple Sequence Alignment matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#include <buffer.hpp>
#include <cartesian.hpp>

template <typename T, size_t D = 2>
class matrix : protected buffer<T>
{
    static_assert(D >= 2, "matrices must be at least 2-dimensional");

    public:
        using element_type = T;                             /// The buffer's element size.
        using cartesian_type = cartesian<D>;                /// The matrix's dimension type.
        using underlying_buffer = buffer<T>;                /// The matrix's underlying buffer.
        using pointer_type = pointer<element_type[]>;       /// The buffer's pointer type.
        using allocator_type = allocatr<element_type[]>;    /// The buffer's allocator type.
        static constexpr size_t dimensionality = D;         /// The matrix's dimensionality.

    protected:
        cartesian_type mdim;                                /// The matrix's dimensions sizes.

    public:
        inline matrix() noexcept = default;
        inline matrix(const matrix&) noexcept = default;
        inline matrix(matrix&&) noexcept = default;

        /**
         * Acquires the ownership of a raw matrix buffer pointer.
         * @param ptr The buffer pointer to acquire.
         * @param dim The size dimensions of buffer to acquire.
         */
        inline explicit matrix(element_type *ptr, const cartesian_type& dim)
        :   underlying_buffer {ptr, dim.volume()}
        ,   mdim {dim}
        {}

        /**
         * Acquires the ownership of a matrix buffer pointer.
         * @param ptr The buffer pointer to acquire.
         * @param dim The size dimensions of buffer to acquire.
         */
        inline explicit matrix(pointer_type&& ptr, const cartesian_type& dim)
        :   underlying_buffer {std::forward<decltype(ptr)>(ptr), dim.volume()}
        ,   mdim {dim}
        {}

        /**
         * Instantiates a new matrix from an already allocated buffer.
         * @param buf The pre-allocated matrix buffer.
         * @param dim The matrix's dimensions.
         */
        inline explicit matrix(const underlying_buffer& buf, const cartesian_type& dim)
        :   underlying_buffer {std::forward<decltype(buf)>(buf)}
        ,   mdim {dim}
        {}

        inline matrix& operator=(const matrix&) = default;
        inline matrix& operator=(matrix&&) = default;

        /**
         * Gives access to a element in the matrix.
         * @param offset The offset's values.
         * @return The requested element.
         */
        __host__ __device__ inline element_type& operator[](const cartesian_type& offset)
        {
            return underlying_buffer::operator[](mdim.collapse(offset));
        }

        /**
         * Gives access to a constant element in the matrix.
         * @param offset The offset's values.
         * @return The requested constant element.
         */
        __host__ __device__ inline const element_type& operator[](const cartesian_type& offset) const
        {
            return underlying_buffer::operator[](mdim.collapse(offset));
        }

        /**
         * Informs the matrix's size and dimensions.
         * @return The matrix's dimensions.
         */
        __host__ __device__ inline const cartesian_type& dimension() const noexcept
        {
            return mdim;
        }

        /**
         * Creates a new matrix of given size.
         * @param dim The matrix's dimension sizes.
         * @return The newly created matrix instance.
         */
        static inline matrix make(const cartesian_type& dim) noexcept
        {
            return make(allocator_type {}, dim);
        }

        /**
         * Creates a new matrix of given size with an allocator.
         * @param alloc The allocator to be used to new matrix.
         * @param dim The matrix's dimension sizes.
         * @return The newly created matrix instance.
         */
        static inline matrix make(const allocator_type& alloc, const cartesian_type& dim) noexcept
        {
            return matrix {pointer_type::make(alloc, dim.volume()), dim};
        }
};

#endif
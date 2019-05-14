/**
 * Multiple Sequence Alignment matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#include "buffer.hpp"
#include "cartesian.hpp"
#include "exception.hpp"

/**
 * A general purpose matrix with given contents type.
 * @tparam T The matrix's contents type.
 * @since 0.1.1
 */
template <typename T>
class Matrix : protected Buffer<T>
{
    protected:
        Cartesian<2> dimension;         /// The matrix's dimensions.

    public:
        inline Matrix() noexcept = default;
        inline Matrix(const Matrix&) = default;
        inline Matrix(Matrix&&) = default;

        /**
         * Instantiates a new empty matrix of given side dimensions.
         * @param dimension The number of columns and lines in new matrix.
         */
        inline Matrix(const Cartesian<2>& dimension)
        :   Buffer<T> {dimension.getVolume()}
        ,   dimension {dimension}
        {}

        /**
         * Instantiates a new empty matrix of given side dimensions.
         * @param lines The number of lines in the matrix.
         * @param columns The number of columns in the matrix.
         */
        inline Matrix(size_t lines, size_t columns)
        :   Buffer<T> {lines * columns}
        ,   dimension {lines, columns}
        {}

        inline Matrix& operator=(const Matrix&) noexcept = default;
        inline Matrix& operator=(Matrix&&) noexcept = default;

        /**
         * Informs the matrix's size and dimensions.
         * @return The matrix's dimensions.
         */
        __host__ __device__ inline Cartesian<2> getSize() const noexcept
        {
            return dimension;
        }

        /**
         * Gives access to an element in the matrix.
         * @param point The point to access in matrix.
         * @return The recovered matrix element.
         */
        __host__ __device__ inline T& at(const Cartesian<2>& point) const
        {
#if defined(msa_compile_cython) && !defined(msa_compile_cuda)
            if(point >= dimension)
                throw Exception("matrix offset out of range");
#endif
            return this->ptr[dimension.collapseTo(point)];
        }

        /**
         * Gives access to an element in the matrix.
         * @param x The requested element line index.
         * @param y The requested element column index.
         * @return The recovered matrix element.
         */
        __host__ __device__ inline T& at(ptrdiff_t x, ptrdiff_t y) const
        {
            return at({x, y});
        }
};

#endif
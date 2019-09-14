/**
 * Multiple Sequence Alignment matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#include <utility>

#include "tuple.hpp"
#include "utils.hpp"
#include "buffer.hpp"
#include "cartesian.hpp"
#include "exception.hpp"

/**
 * A general purpose matrix with given contents type.
 * @tparam T The matrix's contents type.
 * @tparam D The matrix's dimensionality.
 * @since 0.1.1
 */
template <typename T, size_t D = 2>
class Matrix : protected Buffer<T>
{
    static_assert(D >= 2, "matrices must be at least 2-dimensional");

    protected:
        Cartesian<D> dimension;         /// The matrix's dimensions.

    public:
        inline Matrix() noexcept = default;
        inline Matrix(const Matrix&) noexcept = default;
        inline Matrix(Matrix&&) noexcept = default;

        /**
         * Instantiates a new empty matrix of given side dimensions.
         * @param dimension The number of columns and lines in new matrix.
         */
        inline Matrix(const Cartesian<D>& dimension)
        :   Buffer<T> {dimension.getVolume()}
        ,   dimension {dimension}
        {}

        /**
         * Instantiates a new empty matrix of given side dimensions.
         * @param lines The number of lines in the matrix.
         * @param columns The number of columns in the matrix.
         */
        template <typename ...U, typename = typename std::enable_if<
                utils::all(std::is_convertible<U, size_t>::value...) &&
                sizeof...(U) == D
            >::type >
        inline Matrix(U&&... dimension)
        :   Matrix {Cartesian<D> {std::forward<U>(dimension)...}}
        {}

        /**
         * Instantiates a new matrix from an already allocated buffer.
         * @param buffer The pre-allocated matrix buffer.
         * @param dimension The matrix's dimensions.
         */
        inline Matrix(const Buffer<T>& buffer, const Cartesian<D>& dimension) noexcept
        :   Buffer<T> {buffer}
        ,   dimension {dimension}
        {}

        inline Matrix& operator=(const Matrix&) = default;
        inline Matrix& operator=(Matrix&&) = default;

        /**
         * Gives access to a element in the matrix.
         * @param offset The offset's values.
         * @return The requested element.
         */
        template <typename ...U, typename = typename std::enable_if<
                utils::add(std::is_convertible<U, size_t>::value...) &&
                sizeof...(U) == D
            >::type >
        __host__ __device__ inline T& operator()(U&&... offset) noexcept
        {
            return getOffset({offset...});
        }

        /**
         * Gives access to a constant element in the matrix.
         * @param offset The offset's values.
         * @return The requested constant element.
         */
        template <typename ...U, typename = typename std::enable_if<
                utils::add(std::is_convertible<U, size_t>::value...) &&
                sizeof...(U) == D
            >::type >
        __host__ __device__ inline const T& operator()(U&&... offset) const noexcept
        {
            return getOffset({offset...});
        }

        /**
         * Informs the matrix's size and dimensions.
         * @return The matrix's dimensions.
         */
        __host__ __device__ inline const Cartesian<D>& getDimension() const noexcept
        {
            return dimension;
        }

    protected:
        using Buffer<T>::Buffer;

        /**
         * Gives direct access to elements, bypassing the matrix's organization.
         * @param offset The offset to be accessed.
         * @return The element in given offset.
         */
        __host__ __device__ inline T& getOffset(const Cartesian<D>& offset)
        {
            return this->Buffer<T>::operator[](dimension.collapseTo(offset));
        }

        /**
         * Gives direct constant access to elements, bypassing the matrix's organization.
         * @param offset The offset to be accessed.
         * @return The constant element in given offset.
         */
        __host__ __device__ inline const T& getOffset(const Cartesian<D>& offset) const
        {
            return this->Buffer<T>::operator[](dimension.collapseTo(offset));
        }
};

#endif
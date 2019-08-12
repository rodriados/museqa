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
        /**
         * The auxiliary accessor class, allowing the usage of array operator.
         * @tparam I The current accessed level in the matrix's topology.
         * @since 0.1.1
         */
        template <size_t I = 0>
        class Accessor;

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

        inline Matrix& operator=(const Matrix&) = default;
        inline Matrix& operator=(Matrix&&) = default;

        /**
         * Gives access to an element in the matrix.
         * @param offset The first dimension offset.
         * @return The composed operator instance.
         */
        __host__ __device__ inline auto operator[](ptrdiff_t offset) noexcept
        -> decltype(std::declval<Accessor<>>()[offset])
        {
            return (Accessor<> {*this})[offset];
        }

        /**
         * Gives access to a constant element in the matrix.
         * @param offset The first dimension offset.
         * @return The composed operator instance.
         */
        __host__ __device__ inline auto operator[](ptrdiff_t offset) const noexcept
        -> const decltype(std::declval<Accessor<>>()[offset])
        {
            return (Accessor<> {*this})[offset];
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

/**
 * The auxiliary accessor class, allowing the usage of array operator.
 * @tparam I The current accessed level in the matrix's topology.
 * @since 0.1.1
 */
template <typename T, size_t D>
template <size_t I>
class Matrix<T, D>::Accessor
{
    private:
        const Matrix& matrix;                   /// The target matrix to access.
        mutable nTuple<ptrdiff_t, D> offset;    /// The required offset.

    public:
        /**
         * Initializes the accessor externally.
         * @param matrix The target matrix to access.
         */
        template <size_t J = I, typename = typename std::enable_if<(!J)>::type>
        __host__ __device__ inline Accessor(const Matrix& matrix) noexcept
        :   matrix {matrix}
        {}

        /**
         * Gives access to a further dimension within the matrix.
         * @param val The value given to current dimension.
         * @return The following matrix's dimension accessor.
         */
        template <size_t J = I + 1, typename = typename std::enable_if<(J < D)>::type>
        __host__ __device__ inline Accessor<J> operator[](ptrdiff_t val) noexcept
        {
            tuple::set<I>(offset, val);
            return {matrix, offset};
        }

        /**
         * Gives constant access to a further dimension within the matrix.
         * @param val The value given to current dimension.
         * @return The following matrix's dimension constant accessor.
         */
        template <size_t J = I + 1, typename = typename std::enable_if<(J < D)>::type>
        __host__ __device__ inline const Accessor<J> operator[](ptrdiff_t val) const noexcept
        {
            tuple::set<I>(offset, val);
            return {matrix, offset};
        }

        /**
         * Accesses the requested element in matrix and returns it.
         * @param val The last dimension value.
         * @return The requested element from matrix.
         */
        template <size_t J = I + 1, typename = typename std::enable_if<(J == D)>::type>
        __host__ __device__ inline T& operator[](ptrdiff_t val)
        {
            tuple::set<I>(offset, val);
            // If we got here not const-qualified, than we can guarantee that
            // this matrix is non-const, and so we can safely remove it.
            return const_cast<T&>(matrix.getOffset(Cartesian<D>::fromTuple(offset)));
        }

        /**
         * Accesses the requested constant element in matrix and returns it.
         * @param val The last dimension value.
         * @return The requested constant element from matrix.
         */
        template <size_t J = I + 1, typename = typename std::enable_if<(J == D)>::type>
        __host__ __device__ inline const T& operator[](ptrdiff_t val) const
        {
            tuple::set<I>(offset, val);
            return matrix.getOffset(Cartesian<D>::fromTuple(offset));
        }

    protected:
        /**
         * Initializes a new accessor in a level further than previously.
         * @param matrix The target matrix to be accessed.
         * @param offset The previous offset value.
         */
        __host__ __device__ inline Accessor(const Matrix& matrix, const nTuple<ptrdiff_t, D>& offset) noexcept
        :   matrix {matrix}
        ,   offset {offset}
        {}

    friend class Accessor<I - 1>;
};

#endif
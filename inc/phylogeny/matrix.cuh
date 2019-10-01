/**
 * Multiple Sequence Alignment neighbor-joining distance matrices file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_MATRIX_CUH_INCLUDED
#define PG_MATRIX_CUH_INCLUDED

#include <utility>

#include "cuda.cuh"
#include "matrix.hpp"
#include "pointer.hpp"
#include "pairwise.cuh"

namespace phylogeny
{
    namespace matrix
    {
        /**
         * Maps virtual matrix line offsets to real memory offsets.
         * @since 0.1.1
         */
        struct LineOffset
        {
            Pointer<uint16_t[]> ptr;    /// The matrix's line offsets mapping.
            size_t count;               /// The number of currently active offsets.

            __host__ __device__ inline LineOffset() noexcept = default;
            __host__ __device__ inline LineOffset(const LineOffset&) noexcept = default;
            __host__ __device__ inline LineOffset(LineOffset&&) noexcept = default;

            /**
             * Initializes a brand new mapping from given line count.
             * @param count The maximum number of elements in mapping.
             */
            inline LineOffset(size_t count) noexcept
            :   LineOffset {new uint16_t[count], count}
            {
                for(size_t i = 0; i < count; ++i)
                    ptr[i] = static_cast<uint16_t>(i);
            }

            /**
             * Initializes a new map from a given pointer.
             * @param ptr The pointer to access the map from.
             * @param count The number of currently active elements.
             */
            __host__ __device__ inline LineOffset(const Pointer<uint16_t[]>& ptr, size_t count) noexcept
            :   ptr {ptr}
            ,   count {count}
            {}

            __host__ __device__ inline LineOffset& operator=(const LineOffset&) = default;
            __host__ __device__ inline LineOffset& operator=(LineOffset&&) = default;

            /**
             * Maps the requested virtual offset to the real memory offset.
             * @param offset The requested virtual offset.
             * @return The real memory offset.
             */
            __host__ __device__ inline uint16_t operator[](ptrdiff_t offset) const noexcept
            {
                return ptr[offset];
            }

            /**
             * Informs the number of currently active lines and columns in matrix.
             * @return The currenlt number of active lines and columns.
             */
            __host__ __device__ inline size_t getCount() const noexcept
            {
                return count;
            }

            __host__ __device__ void remove(ptrdiff_t target) noexcept;
        };
    };

    /**
     * Represents a shrinkable matrix that is essencial for the construction of
     * the pseudo-phylogenetic tree.
     * @since 0.1.1
     */
    template <typename T>
    class ShrinkableMatrix : public Matrix<T>
    {
        protected:
            matrix::LineOffset offset;  /// The matrix's elements lines and columns offsets.

        public:
            inline ShrinkableMatrix() noexcept = default;
            inline ShrinkableMatrix(const ShrinkableMatrix&) = default;
            inline ShrinkableMatrix(ShrinkableMatrix&&) = default;

            /**
             * Initializes a new matrix based on an already initiliazed matrix.
             * @param matrix The matrix with values to be copied.
             * @param count The number of lines and columns in matrix.
             */
            inline ShrinkableMatrix(const Matrix<T>& matrix, size_t count)
            :   Matrix<T> {matrix}
            ,   offset {count}
            {}

            /**
             * Initializes a new empty matrix with given square size.
             * @param count The number lines and columns in matrix.
             */
            inline ShrinkableMatrix(size_t count)
            :   ShrinkableMatrix {Matrix<T> {count, count}, count}
            {}

            inline ShrinkableMatrix& operator=(const ShrinkableMatrix&) = default;
            inline ShrinkableMatrix& operator=(ShrinkableMatrix&&) = default;

            /**
             * Gives access to an element in the matrix.
             * @param x The first dimension offset.
             * @param y The second dimension offset.
             * @return The requested matrix element.
             */
            __host__ __device__ inline T& operator()(ptrdiff_t x, ptrdiff_t y)
            {
                return Matrix<T>::getOffset({offset[x], offset[y]});
            }

            /**
             * Gives access to a constant element in the matrix.
             * @param x The first dimension offset.
             * @param y The second dimension offset.
             * @return The requested matrix constant element.
             */
            __host__ __device__ inline const T& operator()(ptrdiff_t x, ptrdiff_t y) const
            {
                return Matrix<T>::getOffset({offset[x], offset[y]});
            }

            /**
             * Informs the number of currently active lines and columns in matrix.
             * @return The currenly of active lines and columns.
             */
            __host__ __device__ inline size_t getCount() const noexcept
            {
                return offset.getCount();
            }

            /**
             * Gives external constant direct access to the matrix's line offset.
             * @return The matrix's lines and columns constant offset.
             */
            __host__ __device__ inline const matrix::LineOffset& getOffset() const noexcept
            {
                return offset;
            }

            /**
             * Effectively removes a line and a column from the matrix.
             * @param x The line and column offset to be removed from matrix.
             */
            __host__ __device__ inline  void removeOffset(ptrdiff_t x) noexcept
            {
                offset.remove(x);
            }

            ShrinkableMatrix toDevice() const;

        protected:
            using Matrix<T>::Matrix;

            /**
             * Creates a new instance from prepared dependencies.
             * @param matrix The matrix of values to be used.
             * @param offset The list of line offsets to use.
             */
            inline ShrinkableMatrix(const Buffer<T>& buffer, const matrix::LineOffset& offset)
            :   Matrix<T> {buffer, {offset.getCount(), offset.getCount()}}
            ,   offset {offset}
            {}
    };

    /**
     * Represents a shrinkable triangular matrix.
     * @since 0.1.1
     */
    template <typename T>
    class TriangularMatrix : public ShrinkableMatrix<T>
    {
        public:
            inline TriangularMatrix() noexcept = default;
            inline TriangularMatrix(const TriangularMatrix&) = default;
            inline TriangularMatrix(TriangularMatrix&&) = default;

            /**
             * Initializes a new empty matrix with given square size.
             * @param count The number lines and columns in matrix.
             */
            inline TriangularMatrix(size_t count)
            :   ShrinkableMatrix<T> {Matrix<T> {1, utils::combinations(count)}, count}
            {}

            inline TriangularMatrix& operator=(const TriangularMatrix&) = default;
            inline TriangularMatrix& operator=(TriangularMatrix&&) = default;

            /**
             * Gives access to an element in the matrix.
             * @param x The first dimension offset.
             * @param y The second dimension offset.
             * @return The requested matrix element.
             */
            __host__ __device__ inline T& operator()(ptrdiff_t x, ptrdiff_t y)
            {
                const auto nx = this->offset[x];
                const auto ny = this->offset[y];
                const ptrdiff_t max = utils::max(nx, ny);
                const ptrdiff_t min = utils::min(nx, ny);
                return Matrix<T>::getOffset({0, utils::combinations(max) + min});
            }

            /**
             * Gives access to a constant element in the matrix.
             * @param x The first dimension offset.
             * @param y The second dimension offset.
             * @return The requested matrix constant element.
             */
            __host__ __device__ inline const T& operator()(ptrdiff_t x, ptrdiff_t y) const
            {
                const auto nx = this->offset[x];
                const auto ny = this->offset[y];
                const ptrdiff_t max = utils::max(nx, ny);
                const ptrdiff_t min = utils::min(nx, ny);
                return Matrix<T>::getOffset({0, utils::combinations(max) + min});
            }

            /**
             * Transfers the matrix to device's memory.
             * @return The new device allocated instance.
             */
            TriangularMatrix toDevice() const
            {
                ShrinkableMatrix<T> dmat = ShrinkableMatrix<T>::toDevice();
                return *reinterpret_cast<TriangularMatrix*>(&dmat);
            }

            /**
             * Creates a new instance from a pairwise buffer.
             * @param pw The pairwise buffer to create matrix from.
             * @return The new matrix instance.
             */
            template <typename U = T>
            static auto fromPairwise(const Pairwise& pw)
            -> typename std::enable_if<std::is_same<U, Score>::value, TriangularMatrix>::type
            {
                Matrix<T> rawmat {pw, {1, utils::combinations(pw.getCount())}};
                return TriangularMatrix {rawmat, pw.getCount()};
            }

        protected:
            using ShrinkableMatrix<T>::ShrinkableMatrix;
    };

    template class ShrinkableMatrix<Score>;
    template class TriangularMatrix<Score>;

    /**
     * Creates an alias for the matrix to be generally used in phylogeny calculations.
     * @since 0.1.1
     */
    typedef TriangularMatrix<Score> PhyloMatrix;
};

#endif
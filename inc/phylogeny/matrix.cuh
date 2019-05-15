/**
 * Multiple Sequence Alignment phylogeny matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PG_MATRIX_CUH_INCLUDED
#define PG_MATRIX_CUH_INCLUDED

#include "matrix.hpp"
#include "pointer.hpp"
#include "cartesian.hpp"

#include "pairwise.cuh"

namespace phylogeny
{
    /**
     * The pairwise sequences alignments' distance matrix. This matrix is actually
     * represented as a contiguous buffer and a list of line pointers.
     * @since 0.1.1
     */
    class Matrix : public ::Matrix<Score>
    {
        protected:
            Pointer<uint16_t[]> index;      /// The indeces of active lines and columns.
            Pointer<Score*[]> lineptr;      /// The matrix's active line pointers.
            uint16_t count = 0;             /// The number of current active lines.

        public:
            inline Matrix() noexcept = default;
            inline Matrix(const Matrix&) = default;
            inline Matrix(Matrix&&) = default;

            /**
             * Initializes a new matrix with scores obtained from pairwise module.
             * @param pw The pairwise module instance.
             */
            inline Matrix(const Pairwise& pw)
            :   ::Matrix<Score> {pw.getCount(), pw.getCount()}
            ,   index {new uint16_t[pw.getCount()]}
            ,   lineptr {new Score*[pw.getCount()]}
            ,   count {static_cast<uint16_t>(pw.getCount())}
            {
                for(size_t i = 0; i < count; ++i) {
                    lineptr[i] = &this->ptr[i * count];
                    index[i] = static_cast<uint16_t>(i);
                }

                for(size_t i = 0, n = 0; i < count; ++i)
                    for(size_t j = i; j < count; ++j)
                        lineptr[i][j] = lineptr[j][i] = (i == j) ? 0 : pw[n++];
            }

            inline Matrix& operator=(const Matrix&) = default;
            inline Matrix& operator=(Matrix&&) noexcept = default;

            /**
             * Retrieves the current matrix dimensional size.
             * @return The current size of the active matrix area.
             */
            __host__ __device__ inline Cartesian<2> getSize() const noexcept
            {
                return {count, count};
            }

            /**
             * Gives access to an element in the matrix.
             * @param point The point to access in matrix.
             * @return The recovered matrix element.
             */
            __host__ __device__ inline Score& at(const Cartesian<2>& point) const
            {
#if defined(msa_compile_cython) && !defined(msa_compile_cuda)
                if(point >= getSize())
                    throw Exception("matrix offset out of range");
#endif
                return lineptr[index[point[0]]][index[point[1]]];
            }

            /**
             * Gives access to an element in the matrix.
             * @param x The requested element line index.
             * @param y The requested element column index.
             * @return The recovered matrix element.
             */
            __host__ __device__ inline Score& at(ptrdiff_t x, ptrdiff_t y) const
            {
                return at(Cartesian<2> {x, y});
            }
    };
};

#endif
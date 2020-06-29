/**
 * Multiple Sequence Alignment symmetric matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <utils.hpp>
#include <matrix.hpp>
#include <allocator.hpp>

namespace msa
{
    /**
     * Represents a general symmetric matrix. As a symmetric matrix must always
     * also be a square matrix, with an obvious symmetry axis over its main diagonal,
     * we optimize the amount of memory needed for storing the matrix in memory.
     * @tparam T The matrix's element type.
     * @since 0.1.1
     */
    template <typename T>
    class symmatrix : protected matrix<T>
    {
        protected:
            using underlying_matrix = matrix<T>;        /// The matrix's base underlying type.

        public:
            using element_type = typename underlying_matrix::element_type;
            using cartesian_type = typename underlying_matrix::cartesian_type;

        public:
            inline symmatrix() noexcept = default;
            inline symmatrix(const symmatrix&) noexcept = default;
            inline symmatrix(symmatrix&&) noexcept = default;

            inline symmatrix& operator=(const symmatrix&) = default;
            inline symmatrix& operator=(symmatrix&&) = default;

            /**
             * Gives access to an element in the matrix.
             * @param offset The offset's values.
             * @return The requested element.
             */
            __host__ __device__ inline element_type& operator[](const cartesian_type& offset)
            {
                return direct(transform(offset, reprdim()));
            }

            /**
             * Gives access to a const-qualified element in the matrix.
             * @param offset The offset's values.
             * @return The requested constant element.
             */
            __host__ __device__ inline const element_type& operator[](const cartesian_type& offset) const
            {
                return direct(transform(offset, reprdim()));
            }

            /**
             * Gives direct access to a not-transformed position in matrix.
             * @param offset The requested matrix offset.
             * @return The element at requested position.
             */
            __host__ __device__ inline element_type& direct(const cartesian_type& offset)
            {
                return underlying_matrix::operator[](offset);
            }

            /**
             * Gives direct const- access to a not-transformed position in matrix.
             * @param offset The requested matrix offset.
             * @return The const-qualified element at requested position.
             */
            __host__ __device__ inline const element_type& direct(const cartesian_type& offset) const
            {
                return underlying_matrix::operator[](offset);
            }

            /**
             * Informs the matrix's dimensional values. It is important to note
             * that the values returned by this function does correspond directly
             * to how the matrix is stored in memory.
             * @return The matrix's virtual dimensions.
             */
            __host__ __device__ inline const cartesian_type dimension() const noexcept
            {
                const auto width = reprdim()[1];
                return {width, width};
            }

            /**
             * Informs the real matrix's representation size. Although we really
             * expose the matrix's size via this method, it does not mean, in
             * any way, it is the real amount of memory used by it in memory.
             * @return The matrix's memory representation size.
             */
            __host__ __device__ inline const cartesian_type& reprdim() const noexcept
            {
                return underlying_matrix::dimension();
            }

            /**
             * Copies data from an existing matrix instance.
             * @param mat The target matrix to copy data from.
             * @return A newly created matrix instance.
             */
            static inline symmatrix copy(const symmatrix& mat) noexcept
            {
                return symmatrix {underlying_matrix::copy(mat)};
            }

            /**
             * Creates a new symmetric matrix for the given number of elements.
             * @param count The number of distinct target elements.
             * @return The newly created symmetric matrix instance.
             */
            static inline symmatrix make(size_t count) noexcept
            {
                return symmatrix {underlying_matrix::make(shape(count))};
            }

            /**
             * Creates a new symmetric matrix for elements with an allocator.
             * @param allocator The allocator to be used to new matrix.
             * @param count The number of distinct target elements.
             * @return The newly created symmetric matrix instance.
             */
            static inline symmatrix make(const msa::allocator& allocator, size_t count) noexcept
            {
                return symmatrix {underlying_matrix::make(allocator, shape(count))};
            }

        protected:
            /**
             * Initializes a new symmetric matrix from an ordinary matrix.
             * @param other The underlying matrix with correct memory layout.
             */
            inline explicit symmatrix(underlying_matrix&& other) noexcept
            :   underlying_matrix {std::forward<decltype(other)>(other)}
            {}

            /**
             * Transforms an external cartesian offset into the internal memory
             * representation used by the symmetric matrix.
             * @param offset The cartesian offset to transform.
             * @param dim The matrix's compressed dimensions.
             * @return The transformed offset.
             */
            __host__ __device__ static inline cartesian_type transform(
                    const cartesian_type& offset
                ,   const cartesian_type& dim
                ) noexcept
            {
                const auto i = utils::max(offset[0], offset[1]);
                const auto j = utils::min(offset[0], offset[1]);

                const auto x = (i < dim[0]) ? i : dim[1] - i - 1;
                const auto y = (i < dim[0]) ? j : dim[1] - j - 1;
                return {x, y};
            }

        private:
            /**
             * Transforms the base number of elements to corresponding dimension.
             * @param count The base number of matrix elements.
             */
            __host__ __device__ static inline cartesian_type shape(size_t count) noexcept
            {
                return {(count >> 1) + (count & 1), count};
            }
    };
}

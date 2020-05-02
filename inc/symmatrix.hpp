/**
 * Multiple Sequence Alignment symmetric matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

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

        protected:
            cartesian_type m_dim;                       /// The matrix's current dimensions.

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
                return direct(transform(offset));
            }

            /**
             * Gives access to a const-qualified element in the matrix.
             * @param offset The offset's values.
             * @return The requested constant element.
             */
            __host__ __device__ inline const element_type& operator[](const cartesian_type& offset) const
            {
                return direct(transform(offset));
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
                return {m_dim[1], m_dim[1]};
            }

            /**
             * Informs the real matrix's representation size. Although we really
             * expose the matrix's size via this method, it does not mean, in
             * any way, it is the real amount of memory used by it in memory.
             * @return The matrix's memory representation size.
             */
            __host__ __device__ inline const cartesian_type& reprdim() const noexcept
            {
                return m_dim;
            }

            /**
             * Creates a new distance matrix for the given number of elements.
             * @param count The number of distinct target elements.
             * @return The newly created distance matrix instance.
             */
            static inline symmatrix make(size_t count) noexcept
            {
                const cartesian_type dim = shape(count);
                return symmatrix {underlying_matrix::make(dim), dim};
            }

            /**
             * Creates a new distance matrix for elements with an allocator.
             * @param allocator The allocator to be used to new matrix.
             * @param count The number of distinct target elements.
             * @return The newly created distance matrix instance.
             */
            static inline symmatrix make(const msa::allocator& allocator, size_t count) noexcept
            {
                const cartesian_type dim = shape(count);
                return symmatrix {underlying_matrix::make(allocator, dim), dim};
            }

        protected:
            /**
             * Initializes a new distance matrix from an ordinary matrix.
             * @param other The underlying matrix with correct memory layout.
             * @param dim The new distance matrix's memory layout dimensions.
             */
            inline explicit symmatrix(underlying_matrix&& other, const cartesian_type& dim) noexcept
            :   underlying_matrix {other}
            ,   m_dim {dim}
            {}

            /**
             * Initializes a square matrix in-place to an already existing matrix
             * and reinterprets it to work with given amount of elements. Please
             * note the reinterpretation must always be to smaller dimensions.
             * @param other The original matrix to be reinterpreted to new dimension.
             * @param count The base number of elements to reinterpret the matrix with.
             */
            __host__ __device__ inline explicit symmatrix(const symmatrix& other, size_t count) noexcept
            :   underlying_matrix {other}
            ,   m_dim {shape(count)}
            {}

            /**
             * Transforms an external cartesian offset into the internal memory
             * representation used by the distance matrix.
             * @param offset The cartesian offset to transform.
             * @return The transformed offset.
             */
            __host__ __device__ inline cartesian_type transform(const cartesian_type& offset) const noexcept
            {
                const auto i = utils::max(offset[0], offset[1]);
                const auto j = utils::min(offset[0], offset[1]);

                const auto x = (i < m_dim[0]) ? i : m_dim[1] - i - 1;
                const auto y = (i < m_dim[0]) ? j : m_dim[1] - j - 1;
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

/**
 * Multiple Sequence Alignment phylogeny matrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <space.hpp>
#include <utils.hpp>
#include <matrix.hpp>
#include <pairwise.cuh>
#include <transform.hpp>

namespace msa
{
    namespace phylogeny
    {
        /**
         * While building the phylogenetic tree, we will eventually need to shrink
         * the pairwise distance matrix, as OTUs are joined together. If we choose
         * to employ a simple matrix for that job, this class will be responsible
         * for managing and performing all needed shrinking tasks.
         * @tparam D Is the matrix stored on device memory?
         * @tparam T The matrix's spacial transformation.
         * @since 0.1.1
         */
        template <bool D = false, typename T = transform::linear<2>>
        class matrix : public msa::matrix<pairwise::score>
        {
            protected:
                using underlying_matrix = msa::matrix<pairwise::score>;

            public:
                static constexpr bool on_device = D;    /// Is matrix data on device memory?

            public:
                using transform_type = T;
                using element_type = typename underlying_matrix::element_type;
                using point_type = typename underlying_matrix::point_type;

            protected:
                point_type m_virtual;                   /// The matrix's dynamic space.

            public:
                __host__ __device__ inline matrix() noexcept = delete;
                __host__ __device__ inline matrix(const matrix&) noexcept = default;
                __host__ __device__ inline matrix(matrix&&) noexcept = default;

                /**
                 * Instantiate from a pairwise module's distance matrix.
                 * @param mat The pairwise module's resulting matrix.
                 */
                inline explicit matrix(const pairwise::distance_matrix& mat) noexcept
                :   matrix {mat, mat.count()}
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
                    return underlying_matrix::operator[](transform_type::transform(m_virtual, offset));
                }

                /**
                 * Gives access to a const-qualified element in the matrix.
                 * @param offset The element's offset value.
                 * @return The requested const-qualified element.
                 */
                __host__ __device__ inline const element_type& operator[](const point_type& offset) const
                {
                    return underlying_matrix::operator[](transform_type::transform(m_virtual, offset));
                }

                /**
                 * Informs the matrix's projection dimensions.
                 * @return The matrix's projected size.
                 */
                __host__ __device__ inline point_type dimension() const noexcept
                {
                    return transform_type::projection(m_virtual);
                }

                /**
                 * Informs the matrix's internal representation dimensions.
                 * @return The matrix's shape in memory.
                 */
                __host__ __device__ inline point_type reprdim() const noexcept
                {
                    return m_virtual;
                }

                auto remove(uint32_t) -> void;
                auto swap(uint32_t, uint32_t) -> void;
                auto to_device() const -> matrix<true, T>;

            protected:
                /**
                 * Creates a new instance from an underlying matrix instance.
                 * @param mat The base matrix instance to be copied.
                 * @param side The matrix's virtual side size.
                 */
                inline explicit matrix(const underlying_matrix& mat, size_t side) noexcept
                :   underlying_matrix {mat}
                ,   m_virtual {transform_type::shape(point_type {side, side})}
                {}

                /**
                 * Inflates the pairwise module's virtual distance matrix into a
                 * full-fledged matrix for the phylogeny module.
                 * @param mat The pairwise module's distance matrix.
                 * @param count The total number of sequences represented in matrix.
                 */
                inline explicit matrix(const pairwise::distance_matrix& mat, size_t count) noexcept
                :   underlying_matrix {underlying_matrix::make({count, count})}
                ,   m_virtual {transform_type::shape(point_type {count, count})}
                {
                    for(size_t i = 0; i < count; ++i)
                        for(size_t j = 0; j <= i; ++j)
                            (*this)[{i, j}] = (*this)[{i, j}] = mat[{i, j}];
                }

                /**
                 * Creates a new matrix of given side size.
                 * @param side The new matrix's width and height.
                 * @return The newly created matrix instance.
                 */
                inline static auto make(size_t side) noexcept -> matrix
                {
                    return matrix {underlying_matrix::make({side, side}), side};
                }

                /**
                 * Creates a new matrix of given side size with an allocator.
                 * @param allocator The allocator to be used to new matrix.
                 * @param side The new matrix's width and height.
                 * @return The newly created matrix instance.
                 */
                inline static auto make(const msa::allocator& allocator, size_t side) -> matrix
                {
                    return matrix {underlying_matrix::make(allocator, {side, side}), side};
                }

            friend class matrix<false, transform_type>;
        };

        /**
         * Implements a shrinkable symmetric matrix.
         * @tparam D Is the matrix stored on device memory?
         * @since 0.1.1
         */
        template <bool D = false>
        using symmatrix = matrix<D, transform::symmetric>;
    }
}

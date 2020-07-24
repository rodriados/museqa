/**
 * Multiple Sequence Alignment phylogeny symmatrix header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <utils.hpp>
#include <pairwise.cuh>
#include <symmatrix.hpp>

namespace msa
{
    namespace phylogeny
    {
        /**
         * While building the phylogenetic tree, we will eventually need to shrink
         * the pairwise distance matrix, as OTUs are joined together. If we choose
         * to employ a symmetric matrix for that job, this class will be responsible
         * for managing and performing all needed shrinking tasks.
         * @since 0.1.1
         */
        class symmatrix : protected msa::symmatrix<score>
        {
            protected:
                using underlying_matrix = msa::symmatrix<score>;

            public:
                using element_type = typename underlying_matrix::element_type;
                using cartesian_type = typename underlying_matrix::cartesian_type;

            protected:
                bool m_device = false;          /// Is the matrix on device memory?
                size_t m_width;                 /// The matrix's dynamic width.

            public:
                __host__ __device__ inline symmatrix() noexcept = default;
                __host__ __device__ inline symmatrix(const symmatrix&) noexcept = default;
                __host__ __device__ inline symmatrix(symmatrix&&) noexcept = default;

                /**
                 * Instantiate from a pairwise module's distance matrix.
                 * @param mat The pairwise module's resulting matrix.
                 */
                inline explicit symmatrix(const pairwise::distance_matrix& mat) noexcept
                :   symmatrix {mat, mat.count()}
                {}

                inline explicit symmatrix(size_t width) noexcept
                :   symmatrix {underlying_matrix::make(width), width, false}
                {}

                __host__ __device__ inline symmatrix& operator=(const symmatrix&) = default;
                __host__ __device__ inline symmatrix& operator=(symmatrix&&) = default;

                /**
                 * Gives access to an element in the matrix.
                 * @param offset The requested element's offset.
                 * @return The requested element value.
                 */
                __host__ __device__ inline element_type& operator[](const cartesian_type& offset)
                {
                    return direct(underlying_matrix::transform(offset, reprdim()));
                }

                /**
                 * Gives access to a const-qualified element in the matrix.
                 * @param offset The requested element's offset.
                 * @return The requested const-qualified element value.
                 */
                __host__ __device__ inline const element_type& operator[](const cartesian_type& offset) const
                {
                    return direct(underlying_matrix::transform(offset, reprdim()));
                }

                /**
                 * Gives direct access to an element offset in the matrix.
                 * @param offset The requested matrix offset.
                 * @return The element at requested offset.
                 */
                __host__ __device__ inline element_type& direct(const cartesian_type& offset)
                {
                    return underlying_matrix::direct(offset);
                }

                /**
                 * Gives direct const-qualified access to an element in the matrix.
                 * @param offset The requested matrix offset.
                 * @return The const-qualified element at requested offset.
                 */
                __host__ __device__ inline const element_type& direct(const cartesian_type& offset) const
                {
                    return underlying_matrix::direct(offset);
                }

                /**
                 * Informs the matrix's dimensions. It is important to note that
                 * the values returned by this function does correspond directly
                 * to how the matrix is stored in memory.
                 * @return The matrix's virtual dimensions.
                 */
                __host__ __device__ inline const cartesian_type dimension() const noexcept
                {
                    return {m_width, m_width};
                }

                /**
                 * Informs the matrix's virtual in-memory dimensions. This allows
                 * direct access to the matrix's elements without applying any
                 * kind of offset transformations.
                 * @return The matrix's virtual in-memory dimensions.
                 */
                __host__ __device__ inline cartesian_type reprdim() const noexcept
                {
                    return underlying_matrix::shape(m_width);
                }

                auto remove(uint32_t) -> void;
                auto swap(uint32_t, uint32_t) -> void;
                auto to_device() const -> symmatrix;

            protected:
                /**
                 * Copies the contents of an already existing instance but interpret
                 * it to have a different, hopefully smaller, width. 
                 * @param other The instance to be copied.
                 * @param width The new matrix's width.
                 */
                inline explicit symmatrix(const symmatrix& other, size_t width) noexcept
                :   symmatrix {other, width, other.m_device}
                {}

                /**
                 * Creates a new instance from an underlying matrix instance.
                 * @param mat The base matrix instance to be copied.
                 * @param width The matrix's virtual width.
                 * @param device Is the matrix on device memory?
                 */
                inline explicit symmatrix(const underlying_matrix& mat, size_t width, bool device) noexcept
                :   underlying_matrix {mat}
                ,   m_device {device}
                ,   m_width {width}
                {}

                /**
                 * Inflates the pairwise module's virtual distance matrix into a
                 * full-fledged matrix for the phylogeny module.
                 * @param mat The pairwise module's distance matrix.
                 * @param count The total number of sequences represented in matrix.
                 */
                inline explicit symmatrix(const pairwise::distance_matrix& mat, size_t count) noexcept
                :   underlying_matrix {underlying_matrix::make(count)}
                ,   m_width {count}
                {
                    for(size_t i = 0; i < count; ++i)
                        for(size_t j = 0; j <= i; ++j)
                            operator[]({i, j}) = mat[{i, j}];
                }

                /**
                 * Creates a new symmetric matrix of given width and height.
                 * @param width The new matrix's width and height.
                 * @return The newly created matrix instance.
                 */
                inline static auto make(size_t width) noexcept -> symmatrix
                {
                    return symmatrix {underlying_matrix::make(width), width, false};
                }

                /**
                 * Creates a new symmetric matrix of given size with an allocator.
                 * @param allocator The allocator to be used to new matrix.
                 * @param width The new matrix's width and height.
                 * @param device Is the matrix being allocated on device memory?
                 * @return The newly created matrix instance.
                 */
                inline static auto make(const msa::allocator& allocator, size_t width, bool device) -> symmatrix
                {
                    return symmatrix {underlying_matrix::make(allocator, width), width, device};
                }
        };
    }
}

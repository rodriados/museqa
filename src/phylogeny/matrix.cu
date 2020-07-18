/**
 * Multiple Sequence Alignment phylogeny matrix file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#include <cstdint>

#include <cuda.cuh>
#include <utils.hpp>
#include <matrix.hpp>

#include <phylogeny/matrix.cuh>

using namespace msa;

/**
 * Aliases the phylogeny matrix element type, to reduce verbosity.
 * @since 0.1.1
 */
using element_type = typename phylogeny::matrix::element_type;

namespace
{    
    namespace device
    {
        /**
         * Performs a horizontal shift on the given source matrix. This operation
         * effectivelly removes a column at the given offset from the matrix.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void hshift(phylogeny::matrix dest, const phylogeny::matrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t height, width;

            height = (int32_t) src.reprdim()[0];
            width  = (int32_t) src.reprdim()[1];

            for(int32_t i = 0; i < height; i += gridDim.x) {
                for(int32_t j = offset; j < width - 1; j += blockDim.x) {
                    const auto lin = i + blockIdx.x;
                    const auto col = j + threadIdx.x;

                    __syncthreads(); if(lin < height && col < width - 1) element = src[{lin, col + 1}];
                    __syncthreads(); if(lin < height && col < width - 1) dest[{lin, col}] = element;
                    __syncthreads();
                }
            }
        }

        /**
         * Performs a vertical shift on the given source matrix. This operation
         * effectivelly removes a line at the given offset from the matrix.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void vshift(phylogeny::matrix dest, const phylogeny::matrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t height, width;

            height = (int32_t) src.reprdim()[0];
            width  = (int32_t) src.reprdim()[1];

            for(int32_t j = 0; j < width; j += gridDim.x) {
                for(int32_t i = offset; i < height - 1; i += blockDim.x) {
                    const auto lin = i + threadIdx.x;
                    const auto col = j + blockIdx.x;

                    __syncthreads(); if(lin < height - 1 && col < width) element = src[{lin + 1, col}];
                    __syncthreads(); if(lin < height - 1 && col < width) dest[{lin, col}] = element;
                    __syncthreads();
                }
            }
        }

        /**
         * Exchanges positions of two columns and lines at the given offsets.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        __global__ void xchange(phylogeny::matrix target, int32_t offset1, int32_t offset2)
        {
            __shared__ int32_t diagonal;

            diagonal  = (int32_t) utils::min(target.reprdim()[0], target.reprdim()[1]);

            for(int32_t i = threadIdx.x; i < diagonal; i += blockDim.x) {
                if(i != offset1 && i != offset2) {
                    utils::swap(target[{offset1, i}], target[{offset2, i}]);
                    utils::swap(target[{i, offset1}], target[{i, offset2}]);
                }
            }
        }

        /**
         * Shrinks the matrix sitting on device memory when using a simple matrix
         * as the base matrix type. This function simply moves all elements that
         * must be moved to their new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        void shrink(phylogeny::matrix& dest, const phylogeny::matrix& src, int32_t offset)
        {
            const auto height = (int32_t) src.reprdim()[0];
            const auto width  = (int32_t) src.reprdim()[1];

            // Let's calculate the total number of columns and lines that must be
            // shifted, so we don't have to rebuild the whole matrix every time.
            const auto left   = width - offset - 1;
            const auto bottom = height - offset - 1;

            // Due to the memory layout of our matrix in memory, we don't need to
            // do anything if we want to remove the last column.
            if(left <= 0 || bottom <= 0) return;

            const auto device_props = cuda::device::properties();
            const auto max_threads  = device_props.maxThreadsDim[0];
            const auto max_blocks   = device_props.maxGridSize[0];

            // To remove a column from the matrix, we will first shift left all columns
            // located to the right of the column being removed. And then shift
            // up all lines below the one being removed.
            hshift<<<utils::min(max_blocks, height), utils::min(max_threads, left)>>>(dest, src, offset);
            vshift<<<utils::min(max_blocks, width), utils::min(max_threads, bottom)>>>(dest, src, offset);
            cuda::barrier();
        }

        /**
         * Swaps two columns and lines on device memory.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        void swap(phylogeny::matrix& target, int32_t offset1, int32_t offset2)
        {
            const auto height   = (int32_t) target.reprdim()[0];
            const auto width    = (int32_t) target.reprdim()[1];
            const auto diagonal = utils::min(width, height);

            // Let's verify whether both offsets are valid for a swap in our matrix.
            // If at least one of them is not valid, then we bail out.
            if(utils::max(offset1, offset2) >= diagonal) return;

            const auto device_props = cuda::device::properties();
            const auto max_threads  = device_props.maxThreadsDim[0];

            // To swap two offsets in our matrix, we iterate over its diagonal so
            // we can swap both axis at the same time, without any run conditions.
            xchange<<<1, utils::min(max_threads, diagonal - 1)>>>(target, offset1, offset2);
            cuda::barrier();
        }
    }

    namespace host
    {
        /**
         * Shrinks the matrix on host when using a simple matrix as the base matrix
         * type. This function simply moves all elements that must be moved to their
         * new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        void shrink(phylogeny::matrix& dest, const phylogeny::matrix& src, int32_t offset)
        {
            const auto height = (int32_t) src.reprdim()[0];
            const auto width  = (int32_t) src.reprdim()[1];

            // For the usual matrix memory layout, we must move all elements that
            // are located below the line and to right of column being removed.
            // Thus, we move all elements one line up and one column to the left.
            for(int32_t i = 0; i < offset; ++i)
                for(int32_t j = offset; j < width - 1; ++j)
                    dest[{j, i}] = dest[{i, j}] = src[{i, j + 1}];

            // For those elements that are both below and to the left of the target
            // offset being removed, we must move them diagonally to occupy the
            // space left by the removed offset both horizontally and vertically.
            for(int32_t i = offset; i < width - 1; ++i)
                for(int32_t j = offset; j < height - 1; ++j)
                    dest[{i, j}] = src[{i + 1, j + 1}];
        }

        /**
         * Swaps two columns and lines on host memory according to their given offsets.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        void swap(phylogeny::matrix& target, int32_t offset1, int32_t offset2)
        {
            const auto height   = (int32_t) target.reprdim()[0];
            const auto width    = (int32_t) target.reprdim()[1];
            const auto diagonal = utils::min(width, height);

            // Let's check whether both offsets exist and can be swapped to one
            // another. If at least one of them do not exist, we bail out.
            if(utils::max(offset1, offset2) > diagonal - 1) return;

            // Iterating through the matrix's diagonal allow us to perform both
            // column and line swaps at once. As we iterate, we also want to preserve
            // the relation between our offsets, as it cannot be moved.
            for(int32_t i = 0; i < diagonal; ++i) {
                if(i != offset1 && i != offset2) {
                    utils::swap(target[{offset1, i}], target[{offset2, i}]);
                    utils::swap(target[{i, offset1}], target[{i, offset2}]);
                }
            }
        }
    }

    /**
     * Shrinks the source matrix into the given destination. This function is a
     * proxy for the one implementing the shrinking algorithm on the correct context.
     * @param dest A matrix instance with the new dimensions.
     * @param src The original matrix instance, with original dimensions.
     * @param offset The column and line offset to be removed.
     * @param device Is the source matrix on device memory?
     */    
    inline void shrink(phylogeny::matrix& dest, const phylogeny::matrix& src, uint32_t offset, bool device)
    {
        if(device) {
            ::device::shrink(dest, src, int32_t(offset));
        } else {
            ::host::shrink(dest, src, int32_t(offset));
        }
    }

    /**
     * Swaps two given offsets on the given matrix. This function is a proxy to
     * the one implementing the swapping algorithm on the correct context.
     * @param target The target matrix to have its columns and lines swapped.
     * @param offset1 The first column and line offset to swap.
     * @param offset2 The second column and line offset to swap.
     * @param device Is the source matrix on device memory?
     */
    inline void swap(phylogeny::matrix& target, uint32_t offset1, uint32_t offset2, bool device)
    {
        if(device) {
            ::device::swap(target, int32_t(offset1), int32_t(offset2));
        } else {
            ::host::swap(target, int32_t(offset1), int32_t(offset2));
        }
    }
}

namespace msa
{
    /**
     * Removes a column from matrix and effectively shrinks the matrix. Although
     * we are removing a column by its offset, we will shift left all other columns
     * to its right, thus occupying the removed column's position.
     * @param offset The line and column offset to be removed.
     */
    auto phylogeny::matrix::remove(uint32_t offset) -> void
    {
        phylogeny::matrix smaller {*this, m_width - 1};
        ::shrink(smaller, *this, offset, m_device);
        operator=(smaller);
    }

    /**
     * Swaps two columns and lines elements with each other.
     * @param offset1 The first column and line offset to swap.
     * @param offset2 The second column and line offset to swap.
     */
    auto phylogeny::matrix::swap(uint32_t offset1, uint32_t offset2) -> void
    {
        phylogeny::matrix target {*this};
        ::swap(*this, offset1, offset2, m_device);
    }

    /**
     * Copies the matrix to the compute-capable device memory.
     * @return The matrix allocated in device memory.
     */
    auto phylogeny::matrix::to_device() const -> phylogeny::matrix
    {
        const auto underlying_dimension = underlying_matrix::dimension();
        const auto occupied_memory = underlying_dimension.volume();

        auto instance = phylogeny::matrix::make(cuda::allocator::device, m_width, true);
        cuda::memory::copy(instance.raw(), this->raw(), occupied_memory);

        return instance;
    }
}

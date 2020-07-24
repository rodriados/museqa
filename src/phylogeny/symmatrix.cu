/**
 * Multiple Sequence Alignment phylogeny symmetric matrix file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020 Rodrigo Siqueira
 */
#include <cstdint>

#include <cuda.cuh>
#include <utils.hpp>
#include <symmatrix.hpp>

#include <phylogeny/symmatrix.cuh>

using namespace msa;

/**
 * Aliases the phylogeny matrix element type, to reduce verbosity.
 * @since 0.1.1
 */
using element_type = typename phylogeny::symmatrix::element_type;

namespace
{    
    namespace device
    {
        /**
         * Performs a horizontal shift on the given source matrix. This operation
         * effectivelly maps the columns and lines affected by the offset removal
         * to their new position in the new shrunk matrix.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void hshift(phylogeny::symmatrix dest, const phylogeny::symmatrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t total, midpoint, height;

            total    = (int32_t) dest.reprdim()[1];
            midpoint = (int32_t) dest.reprdim()[0];
            height   = min(total - offset, midpoint - (total & 1));

            for(int32_t i = total - height; i < total; i += gridDim.x) {
                for(int32_t j = offset - 1; j >= 0; j -= blockDim.x) {
                    const int32_t lin = i + blockIdx.x;
                    const int32_t col = j - threadIdx.x;

                    __syncthreads(); if(lin < total && col >= 0)
                        element = src[{lin + 1, col}];
                    __syncthreads(); if(lin < total && col >= 0)
                        dest[{lin, col}] = element;
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
        __global__ void vshift(phylogeny::symmatrix dest, const phylogeny::symmatrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t midpoint;

            midpoint = (int32_t) dest.reprdim()[0];

            for(int32_t j = 0; j < offset; j += gridDim.x) {
                for(int32_t i = offset; i < midpoint; i += blockDim.x) {
                    const int32_t lin = i + threadIdx.x;
                    const int32_t col = j + blockIdx.x;

                    __syncthreads(); if(lin < midpoint && col < offset)
                        element = src[{lin + 1, col}];
                    __syncthreads(); if(lin < midpoint && col < offset)
                        dest[{lin, col}] = element;
                }
            }
        }

        /**
         * Performs a diagonal shift for elements above the matrix's symmetry axis.
         * This operation effectivelly removes elements from the given offset line.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void dshifta(phylogeny::symmatrix dest, const phylogeny::symmatrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t midpoint, total;

            midpoint = (int32_t) dest.reprdim()[0];
            total    = (int32_t) dest.reprdim()[1];

            for(int32_t j = offset - 2; j >= 0; j -= gridDim.x) {
                for(int32_t i = offset - 1; i >= midpoint; i -= blockDim.x) {
                    const int32_t lin = i - threadIdx.x;
                    const int32_t col = j - threadIdx.x - blockIdx.x;

                    __syncthreads(); if(lin >= midpoint && col >= 0)
                        element = src[{lin, col}];
                    __syncthreads(); if(lin >= midpoint && col >= 0)
                        dest[{lin, col}] = element;
                }
            }
        }

        /**
         * Performs a diagonal shift for elements below the matrix's symmetry axis.
         * This operation effectivelly moves elements in the source matrix to their
         * new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void dshiftb(phylogeny::symmatrix dest, const phylogeny::symmatrix src, int32_t offset)
        {
            element_type element;
            __shared__ int32_t midpoint;

            midpoint = (int32_t) dest.reprdim()[0];

            for(int32_t i = offset + 1; i < midpoint; i += gridDim.x) {
                for(int32_t j = offset; j < midpoint; j += blockDim.x) {
                    const int32_t lin = i + blockIdx.x + threadIdx.x;
                    const int32_t col = j + threadIdx.x;

                    __syncthreads(); if(lin < midpoint && col < midpoint - 1)
                        element = src[{lin + 1, col + 1}];
                    __syncthreads(); if(lin < midpoint && col < midpoint - 1)
                        dest[{lin, col}] = element;
                }
            }
        }

        /**
         * Exchanges positions of two columns and lines at the given offsets.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        __global__ void xchange(phylogeny::symmatrix target, int32_t offset1, int32_t offset2)
        {
            __shared__ int32_t total;

            total  = (int32_t) utils::min(target.dimension()[0], target.dimension()[1]);

            for(int32_t i = threadIdx.x; i < total; i += blockDim.x) {
                if(i != offset1 && i != offset2) {
                    utils::swap(target[{offset1, i}], target[{offset2, i}]);
                }
            }
        }

        /**
         * Shrinks the matrix sitting on device memory when using a symmetric matrix
         * as the base matrix type. This function simply moves all elements that
         * must be moved to their new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        void shrink(phylogeny::symmatrix& dest, const phylogeny::symmatrix& src, int32_t offset)
        {
            using namespace utils;
            const auto midpoint = (int32_t) dest.reprdim()[0];
            const auto total    = (int32_t) dest.reprdim()[1];

            // Let's calculate the "distance" between the given offset and the target
            // matrix's bottom and middle coordinates. These values will help us
            // decide the number of blocks or threads to spawn on device.
            const auto dbottom = min(total - offset, midpoint);
            const auto dmiddle = midpoint - offset - 1;

            const auto device_props = cuda::device::properties();
            const auto max_threads  = device_props.maxThreadsDim[0];
            const auto max_blocks   = device_props.maxGridSize[0];

            // Let's instantiate streams so our kernels can run in parallel. We
            // explicitly create streams as we know this is the most we might use.
            cuda::stream s[3];

            // Firstly, let's horizontally move the elements which will be kept
            // on from the old matrix to their position on the new one. This step
            // will not need to be performed when removing the first or the last
            // line and column from the matrix. On these offsets, the amount of
            // elements to be moved on this step is exactly zero.
            if(dbottom > 0 && offset > 0)
                hshift<<<min(max_blocks, dbottom), min(max_threads, offset), 0, s[0]>>>(dest, src, offset);

            // Unfortunately, our horizontal shift kernel cannot move elements which
            // find themselves "above" the midpoint line. For these elements to
            // be moved, we explicitly perform an effective vertical shift on these
            // elements, so they occupy the space left by the removed line.
            if(offset > 0 && dmiddle + 1 > 0)
                vshift<<<min(max_blocks, offset), min(max_threads, dmiddle + 1), 0, s[1]>>>(dest, src, offset);

            // Lastly, we need to diagonally move any elements which may remain
            // on their original position, and thus misplaced on the new matrix.
            // Although the logical operation is quite similar, these kernels differ
            // in that they behave differently depending on whether these elements
            // find themselves "above" or "below" the midpoint line.
            if(midpoint > offset && dmiddle > 0)
                dshiftb<<<min(max_blocks, dmiddle), min(max_threads, dmiddle), 0, s[2]>>>(dest, src, offset);
            else if(offset >= midpoint)
                dshifta<<<min(max_blocks, offset - 2), min(max_threads, offset), 0, s[2]>>>(dest, src, offset);

            cuda::barrier();
        }

        /**
         * Swaps two columns and lines on device memory.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        void swap(phylogeny::symmatrix& target, int32_t offset1, int32_t offset2)
        {
            using namespace utils;
            const auto height   = (int32_t) target.dimension()[0];
            const auto width    = (int32_t) target.dimension()[1];
            const auto diagonal = min(width, height);

            // Let's verify whether both offsets are valid for a swap in our matrix.
            // If at least one of them is not valid, then we bail out.
            if(max(offset1, offset2) >= diagonal) return;

            const auto device_props = cuda::device::properties();
            const auto max_threads  = device_props.maxThreadsDim[0];

            // To swap two offsets in our matrix, we iterate over its diagonal so
            // we can swap both axis at the same time, without any run conditions.
            xchange<<<1, min(max_threads, diagonal - 1)>>>(target, offset1, offset2);
            cuda::barrier();
        }
    }

    namespace host
    {

        /**
         * Shrinks the matrix on host when using a symmetric matrix as the base matrix
         * type. This function simply moves all elements that must be moved to their
         * new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        void shrink(phylogeny::symmatrix& dest, const phylogeny::symmatrix& src, uint32_t offset)
        {
            const auto midpoint = (int32_t) src.reprdim()[0];
            const auto total    = (int32_t) src.reprdim()[1];

            // Due to the unique memory layout of our symmetric matrix implementation,
            // the element moves must follow a specific order to not lose data.
            // So, first we move the elements below the line being removed and to
            // the left of the column being removed.
            for(int32_t i = offset; i < total - 1; ++i)
                for(int32_t j = offset - 1; j >= 0; --j)
                    dest[{i, j}] = src[{i + 1, j}];

            // Then, we move every element below the line being removed up to the
            // internal representation's inflection point, in which the rows are 
            // represented vertically in memory and not horizontally as usual.
            for(int32_t i = offset + 1; i < midpoint; ++i)
                for(int32_t j = offset; j < i; ++j)
                    dest[{i, j}] = src[{i + 1, j + 1}];

            // Lastly, we move the columns after the inflection point to their new,
            // positions. These columns are represented horizontally in memory.
            for(int32_t i = offset - 1; i >= midpoint - (total & 1); --i)
                for(int32_t j = i - 1; j >= 0; --j)
                    dest[{i, j}] = src[{i, j}];
        }

        /**
         * Swaps two columns and lines on host memory according to their given offsets.
         * @param target The target matrix to have its columns and lines swapped.
         * @param offset1 The first column and line offset to swap.
         * @param offset2 The second column and line offset to swap.
         */
        void swap(phylogeny::symmatrix& target, int32_t offset1, int32_t offset2)
        {
            const auto height   = (int32_t) target.dimension()[0];
            const auto width    = (int32_t) target.dimension()[1];
            const auto diagonal = utils::min(width, height);

            // Let's check whether both offsets exist and can be swapped to one
            // another. If at least one of them do not exist, we bail out.
            if(utils::max(offset1, offset2) > diagonal - 1) return;

            // Iterating through the matrix's diagonal allow us to perform both
            // column and line swaps at once. As we iterate, we also want to preserve
            // the relation between our offsets, as it cannot be moved.
            for(int32_t i = 0; i < diagonal; ++i)
                if(i != offset1 && i != offset2)
                    utils::swap(target[{offset1, i}], target[{offset2, i}]);
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
    inline void shrink(phylogeny::symmatrix& dest, const phylogeny::symmatrix& src, uint32_t offset, bool device)
    {
        if(device) {
            ::device::shrink(dest, src, (int32_t) offset);
        } else {
            ::host::shrink(dest, src, (int32_t) offset);
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
    inline void swap(phylogeny::symmatrix& target, uint32_t offset1, uint32_t offset2, bool device)
    {
        if(device) {
            ::device::swap(target, (int32_t) offset1, (int32_t) offset2);
        } else {
            ::host::swap(target, (int32_t) offset1, (int32_t) offset2);
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
    auto phylogeny::symmatrix::remove(uint32_t offset) -> void
    {
        phylogeny::symmatrix smaller {*this, m_width - 1};
        ::shrink(smaller, *this, offset, m_device);
        operator=(smaller);
    }

    /**
     * Swaps two columns and lines elements with each other.
     * @param offset1 The first column and line offset to swap.
     * @param offset2 The second column and line offset to swap.
     */
    auto phylogeny::symmatrix::swap(uint32_t offset1, uint32_t offset2) -> void
    {
        phylogeny::symmatrix target {*this};
        ::swap(*this, offset1, offset2, m_device);
    }

    /**
     * Copies the symmetric matrix to the compute-capable device memory.
     * @return The matrix allocated in device memory.
     */
    auto phylogeny::symmatrix::to_device() const -> phylogeny::symmatrix
    {
        const auto underlying_dimension = underlying_matrix::reprdim();
        const auto occupied_memory = underlying_dimension.volume();

        auto instance = phylogeny::symmatrix::make(cuda::allocator::device, m_width, true);
        cuda::memory::copy(instance.raw(), this->raw(), occupied_memory);

        return instance;
    }
}

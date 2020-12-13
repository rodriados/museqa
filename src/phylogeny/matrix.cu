/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the phylogeny-specialized matrix.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#include <cstdint>

#include "cuda.cuh"
#include "point.hpp"
#include "utils.hpp"
#include "matrix.hpp"

#include "phylogeny/matrix.cuh"

namespace museqa
{
    namespace phylogeny
    {
        /*
         * Creating aliases for the classes we are implementing the methods of, so they
         * can be more easily identified and managed.
         */
        using matrixh = phylogeny::matrix<false>;
        using matrixd = phylogeny::matrix<true>;
        using symmatrixh = phylogeny::symmatrix<false>;
        using symmatrixd = phylogeny::symmatrix<true>;
    }
}

using namespace museqa;

/**
 * Aliases the phylogeny matrix element type, to reduce verbosity.
 * @since 0.1.1
 */
using element_type = typename phylogeny::matrix<>::element_type;

// Unfortunately, NVCC gives us false-positives warnings regarding to declared but
// unused functions on anonymous namespaces. For that reason, we brute-force suppress
// this diagnostic here, as this bug had not yet been fixed by the version we use.
#pragma push
#pragma diag_suppress 177

namespace
{    
    namespace common
    {
        /**#@+
         * Performs a single element swap on the given matrix.
         * @tparam D Is the given matrix stored on device memory?
         * @tparam T The given indeces type.
         * @param i The diagonal index at which the swap is being performed.
         * @param a The index of first column and row to be swapped.
         * @param b The index of second column and row to be swapped.
         */
        template <bool D, typename T>
        __host__ __device__ inline void swap(phylogeny::matrix<D>& target, const T& i, const T& a, const T& b)
        {
            utils::swap(target[{a, i}], target[{b, i}]);
            utils::swap(target[{i, a}], target[{i, b}]);
        }

        template <bool D, typename T>
        __host__ __device__ inline void swap(phylogeny::symmatrix<D>& target, const T& i, const T& a, const T& b)
        {
            utils::swap(target[{a, i}], target[{b, i}]);
        }
        /**#@-*/
    }

    namespace device
    {
        namespace d = cuda::device;

        /**
         * Performs a horizontal shift on the given source matrix. This operation
         * effectivelly removes a column at the given offset from the matrix.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void hmove(phylogeny::matrixd dest, const phylogeny::matrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t height, width;
            height = (int32_t) src.dimension()[0];
            width  = (int32_t) src.dimension()[1];

            for(int32_t i = 0; i < height; i += gridDim.x) {
                for(int32_t j = offset; j < width - 1; j += blockDim.x) {
                    const int32_t lin = i + blockIdx.x;
                    const int32_t col = j + threadIdx.x;
                    const bool condition = lin < height && col < width - 1;

                    __syncthreads(); if(condition) elem = src[{lin, col + 1}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
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
        __global__ void vmove(phylogeny::matrixd dest, const phylogeny::matrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t height, width;
            height = (int32_t) src.dimension()[0];
            width  = (int32_t) src.dimension()[1];

            for(int32_t j = 0; j < width; j += gridDim.x) {
                for(int32_t i = offset; i < height - 1; i += blockDim.x) {
                    const int32_t lin = i + threadIdx.x;
                    const int32_t col = j + blockIdx.x;
                    const bool condition = lin < height - 1 && col < width;

                    __syncthreads(); if(condition) elem = src[{lin + 1, col}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
                }
            }
        }

        /**
         * Removes the column and row at the given offset from the matrix. Effectively,
         * shrinks the matrix sitting on device memory when using a simple matrix
         * as the base matrix type. This function simply moves all elements that
         * must be moved to their new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        inline void remove(phylogeny::matrixd& dest, const phylogeny::matrixd& src, int32_t offset)
        {
            const auto height = (int32_t) src.dimension()[0];
            const auto width  = (int32_t) src.dimension()[1];

            // Let's calculate the total number of columns and lines that must be
            // shifted, so we don't have to rebuild the whole matrix every time.
            const auto left   = width - offset - 1;
            const auto bottom = height - offset - 1;

            // Due to the memory layout of our matrix in memory, we don't need to
            // do anything if we want to remove the last column.
            if(left <= 0 || bottom <= 0) return;

            // To remove a column from the matrix, we will first shift left all columns
            // located to the right of the column being removed. And then shift
            // up all lines below the one being removed.
            hmove<<<d::blocks(height), d::threads(left)>>>(dest, src, offset);
            vmove<<<d::blocks(width), d::threads(bottom)>>>(dest, src, offset);
        }

        /**
         * Performs a horizontal shift on the given source matrix. This operation
         * effectivelly maps the columns and lines affected by the offset removal
         * to their new position in the new shrunk matrix.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        __global__ void hmove(phylogeny::symmatrixd dest, const phylogeny::symmatrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t total, midpoint, height;
            total    = (int32_t) dest.reprdim()[1];
            midpoint = (int32_t) dest.reprdim()[0];
            height   = min(total - offset, midpoint - (total & 1));

            for(int32_t i = total - height; i < total; i += gridDim.x) {
                for(int32_t j = offset - 1; j >= 0; j -= blockDim.x) {
                    const int32_t lin = i + blockIdx.x;
                    const int32_t col = j - threadIdx.x;
                    const bool condition = lin < total && col >= 0;

                    __syncthreads(); if(condition) elem = src[{lin + 1, col}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
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
        __global__ void vmove(phylogeny::symmatrixd dest, const phylogeny::symmatrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t midpoint;
            midpoint = (int32_t) dest.reprdim()[0];

            for(int32_t j = 0; j < offset; j += gridDim.x) {
                for(int32_t i = offset; i < midpoint; i += blockDim.x) {
                    const int32_t lin = i + threadIdx.x;
                    const int32_t col = j + blockIdx.x;
                    const bool condition = lin < midpoint && col < offset;

                    __syncthreads(); if(condition) elem = src[{lin + 1, col}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
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
        __global__ void admove(phylogeny::symmatrixd dest, const phylogeny::symmatrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t midpoint;
            midpoint = (int32_t) dest.reprdim()[0];

            for(int32_t j = offset - 2; j >= 0; j -= gridDim.x) {
                for(int32_t i = offset - 1; i >= midpoint; i -= blockDim.x) {
                    const int32_t lin = i - threadIdx.x;
                    const int32_t col = j - threadIdx.x - blockIdx.x;
                    const bool condition = lin >= midpoint && col >= 0;

                    __syncthreads(); if(condition) elem = src[{lin, col}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
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
        __global__ void bdmove(phylogeny::symmatrixd dest, const phylogeny::symmatrixd src, int32_t offset)
        {
            element_type elem;

            __shared__ int32_t midpoint;
            midpoint = (int32_t) dest.reprdim()[0];

            for(int32_t i = offset + 1; i < midpoint; i += gridDim.x) {
                for(int32_t j = offset; j < midpoint; j += blockDim.x) {
                    const int32_t lin = i + blockIdx.x + threadIdx.x;
                    const int32_t col = j + threadIdx.x;
                    const bool condition = lin < midpoint && col < midpoint - 1;

                    __syncthreads(); if(condition) elem = src[{lin + 1, col + 1}];
                    __syncthreads(); if(condition) dest[{lin, col}] = elem;
                }
            }
        }

        /**
         * Removes the column and row at the given offset from the matrix. Effectively,
         * shrinks the matrix sitting on device memory when using a symmetric matrix
         * as the base matrix type. This function simply moves all elements that
         * must be moved to their new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param offset The column and line offset to be removed.
         */
        inline void remove(phylogeny::symmatrixd& dest, const phylogeny::symmatrixd& src, int32_t offset)
        {
            const auto midpoint = (int32_t) dest.reprdim()[0];
            const auto total    = (int32_t) dest.reprdim()[1];

            // Let's calculate the "distance" between the given offset and the target
            // matrix's bottom and middle coordinates. These values will help us
            // decide the number of blocks or threads to spawn on device.
            const auto dbottom = utils::min(total - offset, midpoint);
            const auto dmiddle = midpoint - offset - 1;

            // Let's instantiate streams so our kernels can run in parallel. We
            // explicitly create streams as we know this is the most we might use.
            cuda::stream s[3];

            // Firstly, let's horizontally move the elements which will be kept
            // on from the old matrix to their position on the new one. This step
            // will not need to be performed when removing the first or the last
            // line and column from the matrix. On these offsets, the amount of
            // elements to be moved on this step is exactly zero.
            if(dbottom > 0 && offset > 0)
                hmove<<<d::blocks(dbottom), d::threads(offset), 0, s[0]>>>(dest, src, offset);

            // Unfortunately, our horizontal shift kernel cannot move elements which
            // find themselves "above" the midpoint line. For these elements to
            // be moved, we explicitly perform an effective vertical shift on these
            // elements, so they occupy the space left by the removed line.
            if(offset > 0 && dmiddle + 1 > 0)
                vmove<<<d::blocks(offset), d::threads(dmiddle + 1), 0, s[1]>>>(dest, src, offset);

            // Lastly, we need to diagonally move any elements which may remain
            // on their original position, and thus misplaced on the new matrix.
            // Although the logical operation is quite similar, these kernels differ
            // in that they behave differently depending on whether these elements
            // find themselves "above" or "below" the midpoint line.
            if(midpoint > offset && dmiddle > 0)
                bdmove<<<d::blocks(dmiddle), d::threads(dmiddle), 0, s[2]>>>(dest, src, offset);
            else if(offset >= midpoint)
                admove<<<d::blocks(offset - 2), d::threads(offset), 0, s[2]>>>(dest, src, offset);
        }
    }

    namespace host
    {
        /**
         * Removes the column and row at the given offset from the matrix. Effectively,
         * shrinks the matrix on host when using a simple matrix as the base matrix
         * type. This function simply moves all elements that must be moved to their
         * new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param x The column and line offset to be removed.
         */
        inline void remove(phylogeny::matrixh& dest, const phylogeny::matrixh& src, int32_t x)
        {
            const auto height = (int32_t) src.dimension()[0];
            const auto width  = (int32_t) src.dimension()[1];

            // For the usual matrix memory layout, we must move all elements that
            // are located below the line and to right of column being removed.
            // Thus, we move all elements one line up and one column to the left.
            for(int32_t i = 0; i < x; ++i)
                for(int32_t j = x; j < width - 1; ++j)
                    dest[{i, j}] = dest[{j, i}] = src[{i, j + 1}];

            // For those elements that are both below and to the left of the target
            // offset being removed, we must move them diagonally to occupy the
            // space left by the removed offset both horizontally and vertically.
            for(int32_t i = x; i < width - 1; ++i)
                for(int32_t j = x; j < height - 1; ++j)
                    dest[{i, j}] = src[{i + 1, j + 1}];
        }

        /**
         * Removes the column and row at the given offset from the matrix. Effectively,
         * shrinks the matrix on host when using a symmetric matrix as the base matrix
         * type. This function simply moves all elements that must be moved to their
         * new position when the matrix is shrunk.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param x The column and line offset to be removed.
         */
        inline void remove(phylogeny::symmatrixh& dest, const phylogeny::symmatrixh& src, uint32_t x)
        {
            const auto midpoint = (int32_t) src.reprdim()[0];
            const auto total    = (int32_t) src.reprdim()[1];

            // Due to the unique memory layout of our symmetric matrix implementation,
            // the element moves must follow a specific order to not lose data.
            // So, first we move the elements below the line being removed and to
            // the left of the column being removed.
            for(int32_t i = x; i < total - 1; ++i)
                for(int32_t j = x - 1; j >= 0; --j)
                    dest[{i, j}] = src[{i + 1, j}];

            // Then, we move every element below the line being removed up to the
            // internal representation's inflection point, in which the rows are 
            // represented vertically in memory and not horizontally as usual.
            for(int32_t i = x + 1; i < midpoint; ++i)
                for(int32_t j = x; j < i; ++j)
                    dest[{i, j}] = src[{i + 1, j + 1}];

            // Lastly, we move the columns after the inflection point to their new,
            // positions. These columns are represented horizontally in memory.
            for(int32_t i = x - 1; i >= midpoint - (total & 1); --i)
                for(int32_t j = i - 1; j >= 0; --j)
                    dest[{i, j}] = src[{i, j}];
        }
    }

    namespace device
    {
        namespace d = cuda::device;

        /**
         * Swaps positions of two columns and lines at the given offsets.
         * @tparam T The spatial transformation applied to the given matrix.
         * @param target The target matrix to have its columns and lines swapped.
         * @param a The first column and line offset to swap.
         * @param b The second column and line offset to swap.
         */
        template <typename T>
        __global__ void dswap(phylogeny::matrix<true, T> target, int32_t a, int32_t b)
        {
            __shared__ int32_t diagonal;
            diagonal = min(target.dimension()[0], target.dimension()[1]);

            for(int32_t i = threadIdx.x; i < diagonal; i += blockDim.x)
                if(i != a && i != b)
                    ::common::swap(target, i, a, b);
        }

        /**
         * Swaps two columns and lines on device memory.
         * @tparam T The spatial transformation applied to the given matrix.
         * @param target The target matrix to have its columns and lines swapped.
         * @param a The first column and line offset to swap.
         * @param b The second column and line offset to swap.
         */
        template <typename T>
        inline void swap(phylogeny::matrix<true, T>& target, int32_t a, int32_t b)
        {
            const auto height   = (int32_t) target.dimension()[0];
            const auto width    = (int32_t) target.dimension()[1];
            const auto diagonal = utils::min(width, height);

            // Let's verify whether both offsets are valid for a swap in our matrix.
            // If at least one of them is not valid, then we bail out.
            if(utils::max(a, b) >= diagonal) return;

            // To swap two offsets in our matrix, we iterate over its diagonal so
            // we can swap both axis at the same time, without any run conditions.
            dswap<<<1, d::threads(diagonal - 1)>>>(target, a, b);
        }
    }

    namespace host
    {
        /**
         * Swaps the given columns and lines offsets of a linear matrix on host
         * memory. The matrix is assumed to have the same number of lines and columns.
         * @tparam T The spatial transformation applied to the given matrix.
         * @param target The target matrix to have its columns and lines swapped.
         * @param a The first column and line offset to swap.
         * @param b The second column and line offset to swap.
         */
        template <typename T>
        inline void swap(phylogeny::matrix<false, T>& target, int32_t a, int32_t b)
        {
            const auto height   = (int32_t) target.dimension()[0];
            const auto width    = (int32_t) target.dimension()[1];
            const auto diagonal = utils::min(width, height);

            // Let's check whether both offsets exist and can be swapped to one
            // another. If at least one of them do not exist, we bail out.
            if(utils::max(a, b) > diagonal - 1) return;

            // Iterating through the matrix's diagonal allow us to perform both
            // column and line swaps at once. As we iterate, we also want to preserve 
            // the relation between our offsets, as it cannot be moved.
            for(int32_t i = 0; i < diagonal; ++i)
                if(i != a && i != b)
                    ::common::swap(target, i, a, b);
        }
    }

    namespace proxy
    {
        /**#@+
         * Shrinks the source matrix into the given destination. This function is a
         * proxy for the one implementing the removal operation on the correct context.
         * @tparam T The matrix's spacial transformation.
         * @param dest A matrix instance with the new dimensions.
         * @param src The original matrix instance, with original dimensions.
         * @param x The column and line offset to be removed.
         */    
        template <typename T>
        inline void remove(phylogeny::matrix<true, T>& dest, const phylogeny::matrix<true, T>& src, uint32_t x)
        {
            ::device::remove(dest, src, x);
        }

        template <typename T>
        inline void remove(phylogeny::matrix<false, T>& dest, const phylogeny::matrix<false, T>& src, uint32_t x)
        {
            ::host::remove(dest, src, x);
        }
        /**#@-*/

        /**#@+
         * Swaps two given offsets on the given matrix. This function is a proxy to
         * the one implementing the swapping operation on the correct context.
         * @tparam T The matrix's spacial transformation.
         * @param target The target matrix to have its columns and lines swapped.
         * @param a The first column and line offset to swap.
         * @param b The second column and line offset to swap.
         */
        template <typename T>
        inline void swap(phylogeny::matrix<true, T>& target, uint32_t a, uint32_t b)
        {
            ::device::swap(target, a, b);
        }

        template <typename T>
        inline void swap(phylogeny::matrix<false, T>& target, uint32_t a, uint32_t b)
        {
            ::host::swap(target, a, b);
        }
        /**#@-*/
    }
}

#pragma pop

namespace museqa
{
    /**
     * Removes a column from matrix and effectively shrinks the matrix. Although
     * we are removing a column by its offset, we will shift left all other columns
     * to its right, thus occupying the removed column's position.
     * @tparam D Is the matrix stored on device memory?
     * @tparam T The matrix's spacial transformation.
     * @param x The line and column offset to be removed.
     */
    template <bool D, typename T>
    auto phylogeny::matrix<D, T>::remove(uint32_t x) -> void
    {
        const auto side = this->dimension()[1];
        phylogeny::matrix<D, T> smaller {*this, side - 1};
        ::proxy::remove(smaller, *this, x);
        this->operator=(smaller);
    }

    /**
     * Swaps two columns and lines elements with each other.
     * @tparam D Is the matrix stored on device memory?
     * @tparam T The matrix's spacial transformation.
     * @param a The first column and line offset to swap.
     * @param b The second column and line offset to swap.
     */
    template <bool D, typename T>
    auto phylogeny::matrix<D, T>::swap(uint32_t a, uint32_t b) -> void
    {
        ::proxy::swap(*this, a, b);
    }

    /**
     * Copies the matrix to the compute-capable device memory.
     * @tparam D Is the matrix stored on device memory?
     * @tparam T The matrix's spacial transformation.
     * @return The matrix allocated in device memory.
     */
    template <bool D, typename T>
    auto phylogeny::matrix<D, T>::to_device() const -> phylogeny::matrix<true, T>
    {
        const auto side = this->dimension()[1];
        auto instance = phylogeny::matrix<true, T>::make(cuda::allocator::device, side);
        cuda::memory::copy(instance.raw(), this->raw(), this->m_space.volume());
        return instance;
    }

    /*
     * Explicit template instations for the objects and methods defined above. These
     * declarations allow the linker to find our concrete implementations here.
     */
    template class phylogeny::matrix<true>;
    template class phylogeny::matrix<false>;
    template class phylogeny::matrix<true, transform::symmetric>;
    template class phylogeny::matrix<false, transform::symmetric>;
}

/**
 * Multiple Sequence Alignment neighbor-joining distance matrices file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019 Rodrigo Siqueira
 */
#include "cuda.cuh"
#include "buffer.hpp"

#include "phylogeny/matrix.cuh"

using namespace phylogeny;

/**
 * Removes the selected element of a device offset array.
 * @param target The target element to be removed from list.
 */
__host__ __device__ void matrix::LineOffset::remove(ptrdiff_t target) noexcept
{
    const uint16_t *endptr = &ptr[--count];

#ifdef msa_gpu_code
    const uint32_t thread = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t blockv = blockDim.y * blockDim.x;

    for(uint16_t aux, *it = &ptr[target]; it < endptr; it += blockv) {
        if(it + thread < endptr)
            aux = *(it + thread + 1);

        __syncthreads();

        if(it + thread < endptr)
            *(it + thread) = aux;

        __syncthreads();
    }
#else
    for(uint16_t *it = &ptr[target]; it < endptr; ++it)
        *(it) = *(it + 1);
#endif
}

/**
 * Transfers the matrix to device's memory.
 * @return The new device allocated instance.
 */
template <typename T>
ShrinkableMatrix<T> ShrinkableMatrix<T>::toDevice() const
{
    const size_t count = this->getCount();
    const size_t volume = this->getDimension().getVolume();
    
    matrix::LineOffset dOffset {cuda::allocate<uint16_t>(count), count};
    Buffer<T> dBuffer = Buffer<T> {cuda::allocate<T>(volume), volume};

    cuda::copy<T>(dBuffer.getBuffer(), this->getBuffer(), volume);
    cuda::copy<uint16_t>(&dOffset.ptr, &this->offset.ptr, count);

    return {dBuffer, dOffset};
}

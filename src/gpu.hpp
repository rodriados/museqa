/** @file gpu.hpp
 * @brief Parallel Multiple Sequence Alignment GPU header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _GPU_HPP
#define _GPU_HPP

#include <cuda.h>

#include "msa.h"

#define NV_ALIGN_BYTES 16

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#  error A device of compute capability 2.0 or higher is required.
#endif

#define __cudacheck(call)                                                       \
    if((call) != cudaSuccess) {                                                 \
        cudaError_t err = cudaGetLastError();                                   \
        __debugd("%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        finish(CUDAERROR);                                                      \
    }

#define __cudaerror()                                                           \
    cudaError_t err = cudaGetLastError();                                       \
    if(err != cudaSuccess) {                                                    \
        __debugd("%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        finish(CUDAERROR);                                                      \
    }

namespace gpu
{
    extern int count();
    extern bool check();
    extern bool multi();
    extern int assign();
    extern unsigned align(unsigned);
}

#endif
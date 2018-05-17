/** 
 * Multiple Sequence Alignment GPU tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _GPU_HPP_
#define _GPU_HPP_

#include <cuda.h>

#include "msa.hpp"

/*
 * Checks whether a compatible device is available. If not, compilation
 * fails and informs the error.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#  error A device of compute capability 2.0 or higher is required.
#endif

/*
 * CUDA error handling macros. At least one of these macros should be used whenever
 * a CUDA function is called. This verifies for any errors and tries to inform what
 * was the error.
 */
#define __cudacall(call)                                                        \
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

namespace gpu {

/*
 * Declaring functions to be available to external usage.
 */
extern int count();
extern bool check();
extern bool multi();
extern int assign();

}

#endif
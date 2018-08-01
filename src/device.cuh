/** 
 * Multiple Sequence Alignment device tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _DEVICE_CUH_
#define _DEVICE_CUH_

#include "msa.hpp"

/*
 * Checks whether a compatible device is available. If not, compilation
 * fails and informs the error.
 */
#if defined(__CUDACC__)

#include <cuda.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#  error A device of compute capability 2.0 or higher is required.
#endif

/**
 * Aliases the device error enumerator to a more meaningful name.
 * @since 0.1.alpha
 */
typedef cudaError_t DeviceError;

/**
 * Exposes some CUDA devices's properties.
 * @since 0.1.alpha
 */
typedef cudaDeviceProp DeviceProperties;

/*
 * CUDA error handling macros. At least one of these macros should be used whenever
 * a CUDA function is called. This verifies for any errors and tries to inform what
 * was the error.
 */
#define __cudacall(call)                                                        \
    if((call) != cudaSuccess) {                                                 \
        DeviceError err = cudaGetLastError();                                   \
        __debugd("%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        finalize(ErrorCode::CudaError);                                         \
    }

#define __cudacheck()                                                           \
    DeviceError err = cudaGetLastError();                                       \
    if(err != cudaSuccess) {                                                    \
        __debugd("%s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        finalize(ErrorCode::CudaError);                                         \
    }

#endif

/*
 * Macros that allows seamless declaration of device functions within host code.
 */
#ifndef __CUDACC__
#  define __host__
#  define __device__
#endif

/**
 * Offers a set of tools to easily gather information about the device(s) connected.
 * @since 0.1.alpha
 */
namespace device
{
    extern int count();
    extern bool check();
#ifdef __CUDACC__
    extern const DeviceProperties& properties();
#endif
};

#endif
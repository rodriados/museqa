/** 
 * Multiple Sequence Alignment device tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _DEVICE_CUH_
#define _DEVICE_CUH_

#include <cuda.h>

#include "msa.hpp"

/**
 * Aliases the device error enumerator to a more meaningful name.
 * @since 0.1.alpha
 */
typedef cudaError_t DeviceError;

/**
 * Aliases the device property struct to a more meaningful name.
 * @since 0.1.alpha
 */
typedef cudaDeviceProp DeviceProperty;

/*
 * Checks whether a compatible device is available. If not, compilation
 * fails and informs the error.
 */
#if defined(__CUDACC__)

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

/**
 * Offers a set of tools to gather easily information about the device connected.
 * @since 0.1.alpha
 */
namespace Device
{
    extern int count();
    extern bool check();

    extern int get();
    extern bool set(int);

    extern bool reset(int = ~0);
    extern DeviceProperty properties(int = ~0);
};

#endif
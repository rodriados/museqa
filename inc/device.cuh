/** 
 * Multiple Sequence Alignment device tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef DEVICE_CUH_INCLUDED
#define DEVICE_CUH_INCLUDED

#pragma once

#include "msa.hpp"

/**
 * Allows the creation of error instances related to device execution.
 * @since 0.1.alpha
 */
struct DeviceError : public Error
{
    using Error::Error;
    static const DeviceError noGPU();
    static const DeviceError execution(const char *);
};

/*
 * Checks whether a compatible device is available. If not, compilation
 * fails and informs the error.
 */
#ifdef __CUDACC__

#  include <cuda.h>

#  if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#    error A device of compute capability 2.0 or higher is required.
#  endif

/**
 * Aliases the device error enumerator to a more meaningful name.
 * @since 0.1.alpha
 */
typedef cudaError_t DeviceStatus;

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
#  define cudacall(call)                                                        \
    if((call) != cudaSuccess) {                                                 \
        DeviceStatus err = cudaGetLastError();                                  \
        debug("error in %s:%d", __FILE__, __LINE__);                            \
        finalize(DeviceError::execution(cudaGetErrorString(err)));              \
    }

#  define cudacheck()                                                           \
    DeviceStatus err = cudaGetLastError();                                      \
    if(err != cudaSuccess) {                                                    \
        debug("error in %s:%d", __FILE__, __LINE__);                            \
        finalize(DeviceError::execution(cudaGetErrorString(err)));              \
    }

#endif

/*
 * Creation of conditional macros that allow CUDA declarations to be used seamlessly
 * throughout the code without any problems.
 */
#ifdef __CUDACC__
#  define cudadecl __host__ __device__
#else
#  define cudadecl
#  define __device__
#  define __host__
#endif

/**
 * Offers a set of tools to easily gather information about the device(s) connected.
 * @since 0.1.alpha
 */
namespace device
{
    extern int count();
    extern bool exists();
    extern int select();

#ifdef __CUDACC__
    extern const DeviceProperties& properties();
#endif
    
#ifdef __CUDACC__
    /**
     * The function for freeing device pointers.
     * @tparam T The pointer type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline void deleter(T *ptr)
    {
        cudacall(cudaFree(ptr));
    }
#endif
};

#endif
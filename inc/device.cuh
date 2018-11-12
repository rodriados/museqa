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

#  define cudacheck() {                                                         \
    DeviceStatus err = cudaGetLastError();                                      \
    if(err != cudaSuccess) {                                                    \
        debug("error in %s:%d", __FILE__, __LINE__);                            \
        finalize(DeviceError::execution(cudaGetErrorString(err)));              \
    }}

#endif

/*
 * Creation of conditional macros that allow CUDA declarations to be used
 * seamlessly throughout the code without any problems.
 */
#ifdef __CUDACC__
#  define cudadecl __host__ __device__
#else
#  define cudadecl
#endif

/**
 * Offers a set of tools to easily gather information about the device(s) connected.
 * @since 0.1.alpha
 */
namespace device
{
#ifdef __CUDACC__
    /**
     * Aliases the memory copy kind indicator.
     * @since 0.1.alpha
     */
    typedef cudaMemcpyKind CopyKind;

    /*
     * Enumerates all kinds of memory copies to and from device.
     */
    constexpr const CopyKind HtoD = cudaMemcpyHostToDevice;
    constexpr const CopyKind DtoH = cudaMemcpyDeviceToHost;
    constexpr const CopyKind DtoD = cudaMemcpyDeviceToDevice;
#endif

    /*
     * Declaring global functions.
     */
    extern int get();
    extern int count();
    extern bool exists();
    extern int select();
#ifdef __CUDACC__
    extern const DeviceProperties& properties();
#endif

#ifdef __CUDACC__
    /**
     * Allocates a block of memory in device.
     * @tparam T The pointer type.
     * @param ptr The pointer to which address will be stored.
     * @param byter The size of memory block to allocate.
     */
    template <typename T>
    inline void malloc(T *& ptr, size_t bytes)
    {
        cudacall(cudaMalloc(&ptr, bytes));
    }

    /**
     * Copies a memory block into a different destination.
     * @param dest The destination to which data will be copied.
     * @param source The source of data to be copied.
     * @param bytes The number of bytes to copy.
     * @param kind The kind of memory transfer.
     */
    inline void memcpy(void *dest, const void *source, size_t bytes, CopyKind kind = HtoD)
    {
        cudacall(cudaMemcpy(dest, source, bytes, kind));
    }

    /**
     * The function for freeing device pointers.
     * @tparam T The pointer type.
     * @param ptr The pointer to be deleted.
     */
    template <typename T>
    inline void free(T *ptr)
    {
        cudacall(cudaFree(ptr));
    }

    /**
     * Synchronizes the device and host execution. This blocks the
     * host execution until the device's execution is over.
     */
    inline void sync()
    {
        cudacall(cudaThreadSynchronize());
    }
#endif
};

/*
 * Creation of helper macros so the kernel code becomes more readable.
 */
#ifdef __CUDACC__
#  define _calcBlckId_ (blockIdx.y * gridDim.x + blockIdx.x)
#  define _calcThrdId_ (threadIdx.y * blockDim.x + threadIdx.x)

#  define threaddecl(blk, thd)                                                  \
    const uint32_t blk = _calcBlckId_, thd = _calcThrdId_;                      \
    const uint32_t& _intBlkIdRef_ = blk, & _intThdIdRef_ = thd;
    
#  define onlyblock(i) if(_intBlkIdRef_ == (i))
#  define onlythread(i) if(_intThdIdRef_ == (i))
#endif

#endif
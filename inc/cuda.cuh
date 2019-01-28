/** 
 * Multiple Sequence Alignment CUDA tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CUDA_CUH_INCLUDED
#define CUDA_CUH_INCLUDED

#ifdef __CUDACC__
  /*
   * Checks whether a compatible device is available. If not, compilation
   * fails and informs the error.
   */
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error A device of compute capability 2.0 or higher is required.
  #endif

  #include <cuda.h>
#endif

/*
 * Creation of conditional macros that allow CUDA declarations to be used
 * seamlessly throughout the code without any problems.
 */
#ifdef __CUDA_ARCH__
  #define cudadecl __host__ __device__
#else
  #define cudadecl 
#endif

#include <string>
#include <utility>

#include "pointer.hpp"
#include "exception.hpp"

namespace cuda
{
    /**
     * The native CUDA word type. This might be changed in future architectures, but
     * this is good for now.
     * @since 0.1.1
     */
    using NativeWord = unsigned;

    /**
     * Indicates either the result (success or error index) of a CUDA Runtime API call,
     * or the overall status of the Runtime API (which is typically the last triggered
     * error).
     * @since 0.1.1
     */
    using Status = NativeWord;

#ifdef __CUDACC__
    namespace status
    {
        /**
         * Aliases for CUDA error types and status codes enumeration.
         * @since 0.1.1
         */
        enum : std::underlying_type<cudaError_t>::type
        {
            success                     = cudaSuccess
        ,   missingConfiguration        = cudaErrorMissingConfiguration
        ,   memoryAllocation            = cudaErrorMemoryAllocation
        ,   initializationError         = cudaErrorInitializationError
        ,   launchFailure               = cudaErrorLaunchFailure
        ,   priorLaunchFailure          = cudaErrorPriorLaunchFailure
        ,   launchTimeout               = cudaErrorLaunchTimeout
        ,   launchOutOfResources        = cudaErrorLaunchOutOfResources
        ,   invalidDeviceFunction       = cudaErrorInvalidDeviceFunction
        ,   invalidConfiguration        = cudaErrorInvalidConfiguration
        ,   invalidDevice               = cudaErrorInvalidDevice
        ,   invalidValue                = cudaErrorInvalidValue
        ,   invalidPitchValue           = cudaErrorInvalidPitchValue
        ,   invalidSymbol               = cudaErrorInvalidSymbol
        ,   mapBufferObjectFailed       = cudaErrorMapBufferObjectFailed
        ,   unmapBufferObjectFailed     = cudaErrorUnmapBufferObjectFailed
        ,   invalidHostPointer          = cudaErrorInvalidHostPointer
        ,   invalidDevicePointer        = cudaErrorInvalidDevicePointer
        ,   invalidTexture              = cudaErrorInvalidTexture
        ,   invalidTextureBinding       = cudaErrorInvalidTextureBinding
        ,   invalidChannelDescriptor    = cudaErrorInvalidChannelDescriptor
        ,   invalidMemcpyDirection      = cudaErrorInvalidMemcpyDirection
        ,   addressOfConstant           = cudaErrorAddressOfConstant
        ,   textureFetchFailed          = cudaErrorTextureFetchFailed
        ,   textureNotBound             = cudaErrorTextureNotBound
        ,   synchronizationError        = cudaErrorSynchronizationError
        ,   invalidFilterSetting        = cudaErrorInvalidFilterSetting
        ,   invalidNormSetting          = cudaErrorInvalidNormSetting
        ,   mixedDeviceExecution        = cudaErrorMixedDeviceExecution
        ,   cudaRuntimeUnloading        = cudaErrorCudartUnloading
        ,   unknown                     = cudaErrorUnknown
        ,   notYetImplemented           = cudaErrorNotYetImplemented
        ,   memoryValueTooLarge         = cudaErrorMemoryValueTooLarge
        ,   invalidResourceHandle       = cudaErrorInvalidResourceHandle
        ,   notReady                    = cudaErrorNotReady
        ,   insufficientDriver          = cudaErrorInsufficientDriver
        ,   setOnActiveProcess          = cudaErrorSetOnActiveProcess
        ,   invalidSurface              = cudaErrorInvalidSurface
        ,   noDevice                    = cudaErrorNoDevice
        ,   eccUncorrectable            = cudaErrorECCUncorrectable
        ,   sharedObjectSymbolNotFound  = cudaErrorSharedObjectSymbolNotFound
        ,   sharedObjectInitFailed      = cudaErrorSharedObjectInitFailed
        ,   unsupportedLimit            = cudaErrorUnsupportedLimit
        ,   duplicateVariableName       = cudaErrorDuplicateVariableName
        ,   duplicateTextureName        = cudaErrorDuplicateTextureName
        ,   duplicateSurfaceName        = cudaErrorDuplicateSurfaceName
        ,   devicesUnavailable          = cudaErrorDevicesUnavailable
        ,   invalidKernelImage          = cudaErrorInvalidKernelImage
        ,   noKernelImageForDevice      = cudaErrorNoKernelImageForDevice
        ,   incompatibleDriverContext   = cudaErrorIncompatibleDriverContext
        ,   peerAccessAlreadyEnabled    = cudaErrorPeerAccessAlreadyEnabled
        ,   peerAccessNotEnabled        = cudaErrorPeerAccessNotEnabled
        ,   deviceAlreadyInUse          = cudaErrorDeviceAlreadyInUse
        ,   profilerDisabled            = cudaErrorProfilerDisabled
        ,   profilerNotInitialized      = cudaErrorProfilerNotInitialized
        ,   profilerAlreadyStarted      = cudaErrorProfilerAlreadyStarted
        ,   profilerAlreadyStopped      = cudaErrorProfilerAlreadyStopped
        ,   assert                      = cudaErrorAssert
        ,   tooManyPeers                = cudaErrorTooManyPeers
        ,   hostMemoryAlreadyRegistered = cudaErrorHostMemoryAlreadyRegistered
        ,   hostMemoryNotRegistered     = cudaErrorHostMemoryNotRegistered
        ,   operatingSystem             = cudaErrorOperatingSystem
        ,   peerAccessUnsupported       = cudaErrorPeerAccessUnsupported
        ,   launchMaxDepthExceeded      = cudaErrorLaunchMaxDepthExceeded
        ,   launchFileScopedTex         = cudaErrorLaunchFileScopedTex
        ,   launchFileScopedSurf        = cudaErrorLaunchFileScopedSurf
        ,   syncDepthExceeded           = cudaErrorSyncDepthExceeded
        ,   launchPendingCountExceeded  = cudaErrorLaunchPendingCountExceeded
        ,   notPermitted                = cudaErrorNotPermitted
        ,   notSupported                = cudaErrorNotSupported
        ,   hardwareStackError          = cudaErrorHardwareStackError
        ,   illegalInstruction          = cudaErrorIllegalInstruction
        ,   misalignedAddress           = cudaErrorMisalignedAddress
        ,   invalidAddressSpace         = cudaErrorInvalidAddressSpace
        ,   invalidPc                   = cudaErrorInvalidPc
        ,   illegalAddress              = cudaErrorIllegalAddress
        ,   invalidPtx                  = cudaErrorInvalidPtx
        ,   invalidGraphicsContext      = cudaErrorInvalidGraphicsContext
        ,   startupFailure              = cudaErrorStartupFailure
        ,   apiFailureBase              = cudaErrorApiFailureBase
        };

        /**
         * Clears the last error and resets it to success.
         * @return The last error registered by a runtime call.
         */
        inline Status clear() noexcept {
            return cudaGetLastError();
        }

        /**
         * Gets the last error from a runtime call in the same host thread.
         * @return The last error registered by a runtime call.
         */
        inline Status last() noexcept {
            return cudaPeekAtLastError();
        }

        /**
         * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
         * or error code.
         * @param status The error status obtained.
         * @return The error description.
         */
        inline std::string describe(Status status) noexcept {
            return cudaGetErrorString(static_cast<cudaError_t>(status));
        }
    };
#endif

    /**
     * Holds an error message so it can be propagated through the code.
     * @since 0.1.1
     */
    struct Exception : public ::Exception
    {
        Status status;              /// The status code.

        /**
         * Builds a new exception instance from status code.
         * @param status The status code.
         */
        inline Exception(Status status)
        :   ::Exception {"CUDA Excpetion: " + status::describe(status)}
        ,   status {status}
        {}

        /**
         * Builds a new exception instance from status code.
         * @tparam P The format parameters' types.
         * @param status The status code.
         * @param fmt The additional message's format.
         * @param args The format's parameters.
         */
        template <typename ...P>
        inline Exception(Status status, const std::string& fmt, P... args)
        :   ::Exception {"CUDA Exception: " + status::describe(status) + ": " + fmt, args...}
        ,   status {status}
        {}

        /**
         * Informs the status code received from an operation.
         * @return The error status code.
         */
        inline Status getCode() const { return status; }
    };

#ifdef __CUDACC__
    /**
     * Checks whether a CUDA has been successful and throws error if not.
     * @tparam P The format string parameter types.
     * @param status The status code obtained from a function.
     * @param fmt The format string to use as error message.
     * @param args The format string values.
     * @throw The error status code obtained raised to exception.
     */
    template <typename ...P>
    inline void call(Status status, const std::string& fmt = {}, P... args) throw (Exception)
    {
        if(status != cuda::status::success)
            throw Exception {status, fmt, args...};
    }
#endif

    /**
     * The ID type for devices. Here we simply define it as a numeric identifier,
     * which is useful for breaking dependencies and for interaction with code using the
     * original CUDA APIs.
     * @since 0.1.1
     */
    using Device = int;

    namespace device
    {
#ifdef __CUDACC__
        /**
         * Besides attributes, every CUDA device also has properties. This is the type
         * for device properties, aliasing {@ref cudaDeviceProp}.
         * @since 0.1.1
         */
        using Properties = cudaDeviceProp;
#endif

        /**
         * If the CUDA runtime has not been set to a specific device, this
         * is the ID of the device it defaults to.
         * @see cuda::Device
         */
        static constexpr const Device original = 0;

        extern int getCount();
        extern Device getCurrent();
        extern void setCurrent(const Device& = original);
#ifdef __CUDACC__
        extern Properties getProperties(const Device& = original);
#endif
    };

    /**
     * Type aliasing for CUDA kernel types.
     * @tparam P The kernel parameter types.
     * @since 0.1.1
     */
    template <typename ...P>
    using Kernel = void (*)(P...);

#ifdef __CUDACC__
    namespace cache
    {
        /**
         * In some GPU micro-architectures, it's possible to have the multiprocessors
         * change the balance in the allocation of L1-cache-like resources between
         * actual L1 cache and shared memory; these are the possible choices.
         * @since 0.1.1
         */
        enum Preference : std::underlying_type<cudaFuncCache>::type
        {
            none    = cudaFuncCachePreferNone
        ,   equal   = cudaFuncCachePreferEqual
        ,   shared  = cudaFuncCachePreferShared
        ,   l1      = cudaFuncCachePreferL1
        };
    };

    namespace kernel
    {
        /**
         * Sets the cache preference to a kernel.
         * @tparam P The kernel parameter types.
         * @param kernel The kernel to set the cache preference.
         * @param preference The cache preference to set.
         * @return Any detected error or nothing.
         */
        template <typename ...P>
        inline void setCachePreference(const Kernel<P...> kernel, cache::Preference preference) 
        {
            call(cudaFuncSetCacheConfig(
                reinterpret_cast<const void *>(kernel)
            ,   static_cast<cudaFuncCache>(preference))
            );
        }
    };

    template <typename T = void>
    inline RawPointer<T> allocate(size_t = 1);

    template <typename T = void>
    inline void free(Pure<T> *);

    /**
     * Allocates device-side memory on the current device.
     * @tparam T Type of pointer to create.
     * @param elems The number of elements of type T to allocate.
     * @return The allocated memory pointer.
     */
    template <typename T>
    inline RawPointer<T> allocate(size_t elems)
    {
        Pure<T> *rawptr = nullptr;
        using S = typename std::conditional<std::is_void<T>::value, char, Pure<T>>::type;
        call(cudaMalloc(&rawptr, sizeof(S) * elems));

        return {rawptr, free<T>};
    }

    /**
     * Frees an allocated memory region pointer.
     * @tparam T The pointer type.
     * @param ptr The pointer to be freed.
     */
    template <typename T>
    inline void free(Pure<T> *ptr)
    {
        call(cudaFree(ptr));
    }

    /**#@+
     * Synchronously copies data between memory spaces or within a memory space.
     * @note Since we assume Compute Capability >= 2.0, all devices support the
     * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
     * where the data is located, and one does not have to specify this.
     * @tparam T The pointer type.
     * @param destination A pointer to a memory region either in host memory or on a device's global memory.
     * @param source A pointer to a a memory region either in host memory or on a device's global memory.
     * @param elems The number of elements to copy from source to destination.
     */
    template <typename T = void>
    inline void copy(T *destination, const T *source, size_t elems)
    {
        using S = typename std::conditional<std::is_void<T>::value, char, T>::type;
        call(cudaMemcpy(destination, source, sizeof(S) * elems, cudaMemcpyDefault));
    }

    template <typename T>
    inline void copy(T& destination, const T& source)
    {
        copy(&destination, &source, 1);
    }
    /**#@-*/

    /**
     * Synchronously sets all bytes in a region of memory to a fixed value.
     * @param ptr The position from where region starts.
     * @param value The value to set the memory region to.
     * @param bytes The number of bytes to set.
     */
    inline void set(void *ptr, unsigned char value, size_t bytes)
    {
        call(cudaMemset(ptr, value, bytes));
    }

    /**
     * Synchronously sets all bytes in a region of memory to 0 (zero).
     * @param ptr Position from where to start.
     * @param bytes Size of the memory region in bytes.
     */
    inline void zero(void *ptr, size_t bytes)
    {
        set(ptr, 0, bytes);
    }

    namespace pinned
    {
        template <typename T = void>
        inline RawPointer<T> allocate(size_t = 1);

        template <typename T = void>
        inline void free(Pure<T> *);

        /**
         * Allocates pinned host-side memory so copies to and from device are faster.
         * @tparam T Type of pointer to create.
         * @param elems The number of elements of type T to allocate.
         * @return The allocated memory pointer.
         */
        template <typename T>
        inline RawPointer<T> allocate(size_t elems)
        {
            Pure<T> *ptr = nullptr;
            using S = typename std::conditional<std::is_void<T>::value, char, Pure<T>>::type;
            call(cudaMallocHost(&ptr, sizeof(S) * elems));

            return {ptr, free<T>};
        }

        /**
         * Frees an allocated pinned host memory region pointer.
         * @tparam T The pointer type.
         * @param ptr The pointer to be freed.
         */
        template <typename T>
        inline void free(Pure<T> *ptr)
        {
            call(cudaFreeHost(ptr));
        }
    };
#endif

    /**
     * CUDA's NVCC allows use the use of the warpSize identifier, without having
     * to define it. Un(?)fortunately, warpSize is not a compile-time constant; it
     * is replaced at some point with the appropriate immediate value which goes into,
     * the SASS instruction as a literal. This is apparently due to the theoretical
     * possibility of different warp sizes in the future. However, it is useful -
     * both for host-side and more importantly for device-side code - to have the
     * warp size available at compile time. This allows all sorts of useful
     * optimizations, as well as its use in constexpr code.
     * @since 0.1.1
     */
    static constexpr NativeWord warpSize = 32;
};

#endif
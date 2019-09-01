/** 
 * Multiple Sequence Alignment CUDA tools header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef CUDA_CUH_INCLUDED
#define CUDA_CUH_INCLUDED

#ifdef __CUDACC__
  #define msa_compile_cuda 1
#endif

#ifdef msa_compile_cuda
  /*
   * Checks whether a compatible device is available. If not, compilation
   * fails and informs the error.
   */
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error A device of compute capability 2.0 or higher is required.
  #endif

  #include <cuda.h>
#endif

#include <string>
#include <utility>

#include "utils.hpp"
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

    namespace status
    {
#ifdef msa_compile_cuda
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
        inline Status clear() noexcept
        {
            return cudaGetLastError();
        }

        /**
         * Gets the last error from a runtime call in the same host thread.
         * @return The last error registered by a runtime call.
         */
        inline Status last() noexcept
        {
            return cudaPeekAtLastError();
        }
#endif

        extern std::string describe(Status) noexcept;
    };

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
        :   ::Exception {"CUDA Exception: %s", status::describe(status).c_str()}
        ,   status {status}
        {}

        /**
         * Builds a new exception instance from status code.
         * @tparam P The format parameters' types.
         * @param status The status code.
         * @param fmtstr The additional message's format.
         * @param args The format's parameters.
         */
        template <typename ...P>
        inline Exception(Status status, const char *fmtstr, P&&... args)
        :   ::Exception {fmtstr, args...}
        ,   status {status}
        {}

        /**
         * Informs the status code received from an operation.
         * @return The error status code.
         */
        inline Status getCode() const noexcept
        {
            return status;
        }
    };

#ifdef msa_compile_cuda
    /**
     * Checks whether a CUDA has been successful and throws error if not.
     * @tparam P The format string parameter types.
     * @param status The status code obtained from a function.
     * @param fmtstr The format string to use as error message.
     * @param args The format string values.
     * @throw The error status code obtained raised to exception.
     */
    template <typename ...P>
    inline void call(Status status, const std::string& fmtstr = {}, P&&... args)
    {
        if(status != cuda::status::success)
            throw Exception {status, fmtstr.c_str(), args...};
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
#ifdef msa_compile_cuda
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
        enum : Device { original = 0 };

#ifdef msa_compile_cuda
        /**
         * Returns the free amount of memory available for allocation by the device.
         * @return The amount of free memory in bytes.
         */
        inline size_t freeMemory()
        {
            size_t free, _;
            call(cudaMemGetInfo(&free, &_));
            return free;
        }
#endif

        extern int getCount();
        extern Device getCurrent();
        extern void setCurrent(const Device& = original);
#ifdef msa_compile_cuda
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

#ifdef msa_compile_cuda
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
         * @param pref The cache preference to set.
         * @return Any detected error or nothing.
         */
        template <typename ...P>
        inline void preference(const Kernel<P...> kernel, cache::Preference pref) 
        {
            call(cudaFuncSetCacheConfig(
                reinterpret_cast<const void *>(kernel)
            ,   static_cast<cudaFuncCache>(pref))
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
        using S = typename std::conditional<std::is_void<T>::value, char, Pure<T>>::type;

        Pure<T> *ptr = nullptr;
        call(cudaMalloc(&ptr, sizeof(S) * elems));

        return {ptr, free<T>};
    }

    /**
     * Synchronously copies data between memory spaces or within a memory space.
     * @note Since we assume Compute Capability >= 2.0, all devices support the
     * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
     * where the data is located, and one does not have to specify this.
     * @tparam T The pointer type.
     * @param destination A pointer to a memory region either in host memory or on a device's global memory.
     * @param source A pointer to a a memory region either in host memory or on a device's global memory.
     * @param elems The number of elements to copy from source to destination.
     */
    template <typename T>
    inline void copy(T *destination, const T *source, size_t elems = 1)
    {
        call(cudaMemcpy(destination, source, sizeof(T) * elems, cudaMemcpyDefault));
    }

    /**
     * Synchronously sets all bytes in a region of memory to a fixed value.
     * @tparam T The pointer type.
     * @param ptr The position from where region starts.
     * @param value The value to set the memory region to.
     * @param bytes The number of bytes to set.
     */
    template <typename T>
    inline void set(T *ptr, unsigned value, size_t elems = 1)
    {
        call(cudaMemset(ptr, value, sizeof(T) * elems));
    }

    /**
     * Synchronously sets all bytes in a region of memory to 0 (zero).
     * @tparam T The pointer type.
     * @param ptr Position from where to start.
     * @param bytes Size of the memory region in bytes.
     */
    template <typename T>
    inline void zero(T *ptr, size_t elems = 1)
    {
        set(ptr, 0, elems);
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
            using S = typename std::conditional<std::is_void<T>::value, char, Pure<T>>::type;

            Pure<T> *ptr = nullptr;
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

    /**
     * Halts host CPU thread execution until the device has finished processing all
     * previously requested tasks, such as kernel launches, data copies and etc.
     */
    inline void barrier()
    {
        call(cudaDeviceSynchronize());
    }
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
    enum : NativeWord { warpSize = 32 };
};

#endif
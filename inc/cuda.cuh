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
   * Checks whether the compilation is targeting a compatible device. If not,
   * compilation fails and we inform about the error.
   */
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error a device of compute capability 2.0 or higher is required
  #endif

  #include <cuda.h>

  #define onlynvcc 1
#endif

#include <string>
#include <utility>

#include <utils.hpp>
#include <pointer.hpp>
#include <allocatr.hpp>
#include <exception.hpp>

namespace cuda
{
    /**
     * The native CUDA word type. This might be changed in future architectures,
     * but this is good enough for now.
     * @since 0.1.1
     */
    using word = unsigned;

    /**
     * Indicates either the result (success or error index) of a CUDA Runtime API
     * call, or the overall status of the Runtime API (which is typically the last
     * triggered error).
     * @since 0.1.1
     */
    using status_code = word;

    namespace status
    {
        #if defined(onlynvcc)
            /**
             * Aliases for CUDA error types and status codes enumeration.
             * @since 0.1.1
             */
            enum : status_code
            {
                success                         = cudaSuccess
            ,   assert                          = cudaErrorAssert
            ,   api_failure_base                = cudaErrorApiFailureBase
            ,   address_of_constant             = cudaErrorAddressOfConstant
            ,   cudaruntime_unloading           = cudaErrorCudartUnloading
            ,   device_already_in_use           = cudaErrorDeviceAlreadyInUse
            ,   devices_unavailable             = cudaErrorDevicesUnavailable
            ,   duplicate_surface_name          = cudaErrorDuplicateSurfaceName
            ,   duplicate_texture_name          = cudaErrorDuplicateTextureName
            ,   duplicate_variable_name         = cudaErrorDuplicateVariableName
            ,   ecc_uncorrectable               = cudaErrorECCUncorrectable
            ,   hardware_stack_error            = cudaErrorHardwareStackError
            ,   host_memory_already_registered  = cudaErrorHostMemoryAlreadyRegistered
            ,   host_memory_not_registered      = cudaErrorHostMemoryNotRegistered
            ,   illegal_address                 = cudaErrorIllegalAddress
            ,   illegal_instruction             = cudaErrorIllegalInstruction
            ,   incompatible_driver_context     = cudaErrorIncompatibleDriverContext
            ,   initialization_error            = cudaErrorInitializationError
            ,   insufficient_driver             = cudaErrorInsufficientDriver
            ,   invalid_address_space           = cudaErrorInvalidAddressSpace
            ,   invalid_channel_descriptor      = cudaErrorInvalidChannelDescriptor
            ,   invalid_configuration           = cudaErrorInvalidConfiguration
            ,   invalid_device                  = cudaErrorInvalidDevice
            ,   invalid_device_function         = cudaErrorInvalidDeviceFunction
            ,   invalid_device_pointer          = cudaErrorInvalidDevicePointer
            ,   invalid_filter_setting          = cudaErrorInvalidFilterSetting
            ,   invalid_graphics_context        = cudaErrorInvalidGraphicsContext
            ,   invalid_host_pointer            = cudaErrorInvalidHostPointer
            ,   invalid_kernel_image            = cudaErrorInvalidKernelImage
            ,   invalid_memcpy_direction        = cudaErrorInvalidMemcpyDirection
            ,   invalid_norm_setting            = cudaErrorInvalidNormSetting
            ,   invalid_pc                      = cudaErrorInvalidPc
            ,   invalid_pitch_value             = cudaErrorInvalidPitchValue
            ,   invalid_ptx                     = cudaErrorInvalidPtx
            ,   invalid_resource_handle         = cudaErrorInvalidResourceHandle
            ,   invalid_surface                 = cudaErrorInvalidSurface
            ,   invalid_symbol                  = cudaErrorInvalidSymbol
            ,   invalid_texture                 = cudaErrorInvalidTexture
            ,   invalid_texture_binding         = cudaErrorInvalidTextureBinding
            ,   invalid_value                   = cudaErrorInvalidValue
            ,   launch_failure                  = cudaErrorLaunchFailure
            ,   launch_file_scoped_tex          = cudaErrorLaunchFileScopedTex
            ,   launch_file_scoped_surf         = cudaErrorLaunchFileScopedSurf
            ,   launch_max_depth_exceeded       = cudaErrorLaunchMaxDepthExceeded
            ,   launch_pending_count_exceeded   = cudaErrorLaunchPendingCountExceeded
            ,   launch_out_of_resources         = cudaErrorLaunchOutOfResources
            ,   launch_timeout                  = cudaErrorLaunchTimeout
            ,   map_buffer_object_failed        = cudaErrorMapBufferObjectFailed
            ,   memory_allocation               = cudaErrorMemoryAllocation
            ,   memory_value_too_large          = cudaErrorMemoryValueTooLarge
            ,   misaligned_address              = cudaErrorMisalignedAddress
            ,   missing_configuration           = cudaErrorMissingConfiguration
            ,   mixed_device_execution          = cudaErrorMixedDeviceExecution
            ,   no_device                       = cudaErrorNoDevice
            ,   no_kernel_image_for_device      = cudaErrorNoKernelImageForDevice
            ,   not_permitted                   = cudaErrorNotPermitted
            ,   not_ready                       = cudaErrorNotReady
            ,   not_supported                   = cudaErrorNotSupported
            ,   not_yet_implemented             = cudaErrorNotYetImplemented
            ,   operating_system                = cudaErrorOperatingSystem
            ,   peer_access_already_enabled     = cudaErrorPeerAccessAlreadyEnabled
            ,   peer_access_not_enabled         = cudaErrorPeerAccessNotEnabled
            ,   peer_access_unsupported         = cudaErrorPeerAccessUnsupported
            ,   prior_launch_failure            = cudaErrorPriorLaunchFailure
            ,   profiler_already_started        = cudaErrorProfilerAlreadyStarted
            ,   profiler_already_stopped        = cudaErrorProfilerAlreadyStopped
            ,   profiler_disabled               = cudaErrorProfilerDisabled
            ,   profiler_not_initialized        = cudaErrorProfilerNotInitialized
            ,   set_on_active_process           = cudaErrorSetOnActiveProcess
            ,   shared_object_init_failed       = cudaErrorSharedObjectInitFailed
            ,   shared_object_symbol_not_found  = cudaErrorSharedObjectSymbolNotFound
            ,   startup_failure                 = cudaErrorStartupFailure
            ,   sync_depth_exceeded             = cudaErrorSyncDepthExceeded
            ,   synchronization_error           = cudaErrorSynchronizationError
            ,   texture_fetch_failed            = cudaErrorTextureFetchFailed
            ,   texture_not_bound               = cudaErrorTextureNotBound
            ,   too_many_peers                  = cudaErrorTooManyPeers
            ,   unmap_buffer_object_failed      = cudaErrorUnmapBufferObjectFailed
            ,   unsupported_limit               = cudaErrorUnsupportedLimit
            ,   unknown                         = cudaErrorUnknown
            };

            /**
             * Clears the last error and resets it to success.
             * @return The last error registered by a runtime call.
             */
            inline status_code clear() noexcept
            {
                return cudaGetLastError();
            }

            /**
             * Gets the last error from a runtime call in the same host thread.
             * @return The last error registered by a runtime call.
             */
            inline status_code last() noexcept
            {
                return cudaPeekAtLastError();
            }
        #endif

        extern std::string describe(status_code) noexcept;
    }

    /**
     * Holds an error message so it can be propagated through the code.
     * @since 0.1.1
     */
    class exception : public ::exception
    {
        protected:
            using underlying_type = ::exception;    /// The underlying exception.

        protected:
            status_code mcode;                       /// The status code.

        public:     
            /**
             * Builds a new exception instance from status code.
             * @param code The status code.
             */
            inline exception(status_code code) noexcept
            :   underlying_type {"cuda exception: %s", status::describe(code)}
            ,   mcode {code}
            {}

            /**
             * Builds a new exception instance from status code.
             * @tparam T The format parameters' types.
             * @param code The status code.
             * @param fmtstr The additional message's format.
             * @param args The format's parameters.
             */
            template <typename ...T>
            inline exception(status_code code, const std::string& fmtstr, T&&... args) noexcept
            :   underlying_type {fmtstr, args...}
            ,   mcode {code}
            {}

            using underlying_type::exception;

            /**
             * Informs the status code received from an operation.
             * @return The error status code.
             */
            inline status_code code() const noexcept
            {
                return mcode;
            }
    };

    #if defined(onlynvcc)
        /**
         * Checks whether a CUDA has been successful and throws error if not.
         * @param code The status code obtained from a function.
         * @throw The error status code obtained raised to exception.
         */
        inline void check(status_code code)
        {
            enforce<exception>(code == cuda::status::success, "cuda exception: %s", status::describe(code));
        }
    #endif

    namespace device
    {
        /**
         * The id type for devices. Here we simply define it as a numeric identifier,
         * which is useful for breaking dependencies and for interaction with code
         * using the original CUDA APIs.
         * @since 0.1.1
         */
        using id = word;

        /**
         * If the CUDA runtime has not been set to a specific device, this is the
         * id of the device it defaults to.
         * @see cuda::device::id
         */
        enum : id { init = 0 };

        #if defined(onlynvcc)
            /**
             * Every CUDA device can provide information about its fisical properties.
             * @see cudaDeviceProp
             * @since 0.1.1
             */
            using props = cudaDeviceProp;

            /**
             * Returns the amount of free global memory in the current device.
             * @return The amount of free global memory in bytes.
             */
            inline size_t free_memory()
            {
                size_t free, total;
                check(cudaMemGetInfo(&free, &total));
                return free;
            }
        #endif

        extern auto count() -> size_t;
        extern auto current() -> id;
        extern auto select(const id& = init) -> void;

        #if defined(onlynvcc)
            extern auto properties(const id& = init) -> props;
        #endif
    }

    #if defined(onlynvcc)
        /**
         * In some GPU micro-architectures, it's possible to have the multiprocessors
         * change the balance in the allocation of L1-cache-like resources between
         * actual L1 cache and shared memory; these are the possible choices.
         * @since 0.1.1
         */
        enum cache : std::underlying_type<cudaFuncCache>::type
        {
            none        = cudaFuncCachePreferNone
        ,   equal       = cudaFuncCachePreferEqual
        ,   shared      = cudaFuncCachePreferShared
        ,   l1          = cudaFuncCachePreferL1
        };

        /**
         * On devices with configurable shared memory banks, it's possible to force
         * all launches of a device function to have one of the following shared
         * memory bank size configuration.
         * @since 0.1.1
         */
        enum shared_mem : std::underlying_type<cudaSharedMemConfig>::type
        {
            init        = cudaSharedMemBankSizeDefault
        ,   four_byte   = cudaSharedMemBankSizeFourByte
        ,   eight_byte  = cudaSharedMemBankSizeEightByte
        };

        namespace kernel
        {
            /**
             * Describes a kernel pointer type.
             * @tparam T The kernel's argument types.
             * @since 0.1.1
             */
            template <typename ...T>
            using pointer = void (*)(T...);

            /**#@+
             * Sets preferences to a kernel.
             * @tparam T The kernel parameter types.
             * @param kernel The kernel to set the preference to.
             * @param pref The preference to set to kernel.
             */
            template <typename ...T>
            inline void preference(const pointer<T...> kernel, cache pref)
            {
                check(cudaFuncSetCacheConfig(
                    reinterpret_cast<const void *>(kernel)
                ,   static_cast<cudaFuncCache>(pref)
                ));
            }

            template <typename ...T>
            inline void preference(const pointer<T...> kernel, shared_mem pref)
            {
                check(cudaFuncSetSharedMemConfig(
                    reinterpret_cast<const void *>(kernel)
                ,   static_cast<cudaSharedMemConfig>(pref)
                ));
            }
            /**#@-*/
        }

        namespace memory
        {
            /**
             * Creates an allocator for reserving and managing device-side global
             * memory on the active device.
             * @tparam T Type of pointer to create.
             * @return The newly created allocator.
             */
            template <typename T>
            inline auto global() noexcept -> allocatr<T>
            {
                return {
                    typename allocatr<T>::up {
                        [](size_t count) -> pure<T> * {
                            pure<T> *ptr;
                            check(cudaMalloc(&ptr, sizeof(pure<T>) * count));
                            return ptr;
                        }
                    }
                ,   typename allocatr<T>::down {
                        [](pure<T> *ptr) -> void {
                            check(cudaFree(ptr));
                        }
                    }
                };
            }

            /**
             * Creates an allocator for reserving and managing pinned host-side
             * memory. Pinned memory is unpaged and thus can be accessed faster.
             * @tparam T Type of pointer to create.
             * @return The newly created allocator.
             */
            template <typename T>
            inline auto pinned() noexcept -> allocatr<T>
            {
                return {
                    typename allocatr<T>::up {
                        [](size_t count) -> pure<T> * {
                            pure<T> *ptr;
                            check(cudaMallocHost(&ptr, sizeof(pure<T>) * count));
                            return ptr;
                        }
                    }
                ,   typename allocatr<T>::down {
                        [](pure<T> *ptr) -> void {
                            check(cudaFreeHost(ptr));
                        }
                    }
                };
            }

            /**
             * Synchronously copies data between memory spaces or pointers.
             * @note Since we assume Compute Capability >= 2.0, all devices support
             * the Unified Virtual Address Space, so the CUDA driver can determine,
             * for each pointer, where the data is located, and one does not have
             * to specify this.
             * @tparam T The pointer type.
             * @param dest A pointer on host memory or on a device's global memory.
             * @param src A pointer on host memory or on a device's global memory.
             * @param count The number of elements to copy from source to destination.
             */
            template <typename T>
            inline void copy(T *dest, const T *src, size_t count = 1)
            {
                check(cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDefault));
            }

            /**
             * Synchronously sets all bytes in a region of memory to a fixed value.
             * @param ptr The position from where region starts.
             * @param value The value to set the memory region to.
             * @param bytes The number of bytes to set.
             */
            inline void set(void *ptr, uint8_t value, size_t bytes = 1)
            {
                check(cudaMemset(ptr, value, bytes));
            }

            /**
             * Synchronously sets all bytes in a region of memory to 0 (zero).
             * @param ptr Position from where to start.
             * @param bytes Size of the memory region in bytes.
             */
            inline void zero(void *ptr, size_t bytes = 1)
            {
                set(ptr, 0, bytes);
            }
        }

        /**
         * Halts host CPU thread execution until the device has finished processing
         * all previously requested tasks, such as kernel launches, data copies...
         */
        inline void barrier()
        {
            check(cudaDeviceSynchronize());
        }
    #endif

    /**
     * CUDA's NVCC allows use the use of the warpSize identifier, without having
     * to define it. Un(?)fortunately, warpSize is not a compile-time constant;
     * it is replaced at some point with the appropriate immediate value which
     * goes into, the SASS instruction as a literal. This is apparently due to
     * the theoretical possibility of different warp sizes in the future. However,
     * it is useful, both for host-side and more importantly for device-side code,
     * to have the warp size available at compile time. This allows all sorts of
     * useful optimizations, as well as its use in constexpr code.
     * @since 0.1.1
     */
    enum : word { warp_size = 32 };
}

#endif
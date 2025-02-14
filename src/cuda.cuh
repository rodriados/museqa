/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a wrapper around CUDA functions and structures.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <limits>
#include <string>
#include <cstdint>
#include <utility>

#include "utils.hpp"
#include "pointer.hpp"
#include "allocator.hpp"
#include "exception.hpp"
#include "environment.h"

#if defined(__museqa_compiler_nvcc)
  /*
   * Checks whether the compilation is targeting a compatible device. If not,
   * compilation fails and we inform about the error.
   */
  #if defined(__museqa_runtime_device) && (__CUDA_ARCH__ < 200)
    #error a device of compute capability 2.0 or higher is required
  #endif

  #include <cuda.h>
#endif

namespace museqa
{
    namespace cuda
    {
        /**
         * The native CUDA word type. This might be changed in future architectures,
         * but this is good enough for now.
         * @since 0.1.1
         */
        using word = unsigned;

        namespace status
        {
            /**
             * Indicates either the result (success or error index) of a CUDA Runtime
             * API call, or the overall status of the Runtime API (which is typically
             * the last triggered error).
             * @since 0.1.1
             */
            using code = word;

            #if defined(__museqa_compiler_nvcc)
                /**
                 * Aliases for CUDA error types and status codes enumeration.
                 * @since 0.1.1
                 */
                enum : status::code
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
                inline status::code clear() noexcept
                {
                    return cudaGetLastError();
                }

                /**
                 * Gets the last error from a runtime call in the same host thread.
                 * @return The last error registered by a runtime call.
                 */
                inline status::code last() noexcept
                {
                    return cudaPeekAtLastError();
                }
            #endif

            extern std::string describe(status::code) noexcept;
        }

        /**
         * Holds an error message so it can be propagated through the code.
         * @since 0.1.1
         */
        class exception : public museqa::exception
        {
            protected:
                using underlying_type = museqa::exception;  /// The underlying exception type.

            protected:
                status::code m_code;                        /// The exception's status code.

            public:
                /**
                 * Builds a new exception instance from status code.
                 * @param code The status code.
                 */
                inline exception(status::code code) noexcept
                :   underlying_type {"cuda exception: %s", status::describe(code)}
                ,   m_code {code}
                {}

                /**
                 * Builds a new exception instance from status code.
                 * @tparam T The format parameters' types.
                 * @param code The status code.
                 * @param fmtstr The additional message's format.
                 * @param args The format's parameters.
                 */
                template <typename ...T>
                inline exception(status::code code, const std::string& fmtstr, T&&... args) noexcept
                :   underlying_type {fmtstr, args...}
                ,   m_code {code}
                {}

                using underlying_type::exception;

                /**
                 * Informs the status code received from an operation.
                 * @return The error status code.
                 */
                inline status::code code() const noexcept
                {
                    return m_code;
                }
        };

        #if defined(__museqa_compiler_nvcc)
            /**
             * Checks whether a CUDA has been successful and throws error if not.
             * @param code The status code obtained from a function.
             * @throw The error status code obtained raised to exception.
             */
            inline void check(status::code code)
            {
                using E = museqa::cuda::exception;
                enforce<E>(code == status::success, "cuda exception: %s", status::describe(code));
            }
        #endif

        namespace device
        {
            /**
             * The id type for devices. Here we simply define it as a numeric identifier,
             * which is useful for breaking dependencies and for interaction with
             * code using the original CUDA APIs.
             * @since 0.1.1
             */
            using id = word;

            /**
             * If the CUDA runtime has not been set to a specific device, this
             * is the id of the device it defaults to.
             * @see museqa::cuda::device::id
             */
            enum : id { init = 0 };

            #if defined(__museqa_compiler_nvcc)
                /**
                 * Every CUDA device can provide information about its fisical properties.
                 * @see cudaDeviceProp
                 * @since 0.1.1
                 */
                using property = cudaDeviceProp;

                /**
                 * Returns the amount of free global memory in the current device.
                 * @return The amount of free global memory in bytes.
                 */
                inline size_t free_memory()
                {
                    size_t free, total;
                    cuda::check(cudaMemGetInfo(&free, &total));
                    return free;
                }
            #endif

            extern auto count() -> size_t;
            extern auto current() -> device::id;
            extern void select(device::id = init);

            #if defined(__museqa_compiler_nvcc)
                extern auto properties() -> property;
                extern auto properties(device::id) -> property;
                extern auto blocks(size_t = std::numeric_limits<size_t>::max()) -> size_t;
                extern auto threads(size_t = std::numeric_limits<size_t>::max()) -> size_t;
            #endif
        }

        #if defined(__museqa_compiler_nvcc)
            /**
             * Creates an asynchronous stream, which allows multiple device operations
             * to be parallelized.
             * @since 0.1.1
             */
            class stream
            {
                protected:
                    using underlying_type = cudaStream_t;   /// The underlying stream type.

                public:
                    using id = underlying_type;             /// The stream id type.

                protected:
                    underlying_type m_stream;               /// The internal device stream reference.

                public:
                    /**
                     * Creates a new asynchronous stream. This is the stream reference
                     * which will be wrapped and abstracted by this object.
                     */
                    inline stream()
                    {
                        cuda::check(cudaStreamCreate(&m_stream));
                    }

                    /**
                     * Destroys and cleans up the wrapped stream. In case the device
                     * is still doing work in the stream, the resources will be
                     * released once the device has completed has completed all work.
                     * @see museqa::cuda::stream::stream
                     */
                    inline ~stream()
                    {
                        cuda::check(cudaStreamDestroy(m_stream));
                    }

                    /**
                     * Casts the stream wrapper as its underlying type, so it can
                     * be seamlessly used in all existing CUDA interface by default.
                     * @return The internal stream representation.
                     */
                    inline operator underlying_type() const noexcept
                    {
                        return m_stream;
                    }

                    /**
                     * Blocks host CPU execution until all tasks on the wrapped
                     * stream have been completed.
                     */
                    inline void barrier() const
                    {
                        cuda::check(cudaStreamSynchronize(m_stream));
                    }
            };

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
                inline void preference(const kernel::pointer<T...> kernel, cache pref)
                {
                    const auto fptr = reinterpret_cast<const void *>(kernel);
                    cuda::check(cudaFuncSetCacheConfig(fptr, static_cast<cudaFuncCache>(pref)));
                }

                template <typename ...T>
                inline void preference(const kernel::pointer<T...> kernel, shared_mem pref)
                {
                    const auto fptr = reinterpret_cast<const void *>(kernel);
                    cuda::check(cudaFuncSetSharedMemConfig(fptr, static_cast<cudaSharedMemConfig>(pref)));
                }
                /**#@-*/
            }

            namespace allocator
            {
                extern museqa::allocator device;
                extern museqa::allocator pinned;
            }

            namespace memory
            {
                /**
                 * Synchronously copies data between memory spaces or pointers.
                 * @note Since we assume compute capability bigger or equal to 2.0,
                 * all devices support the Unified Virtual Address Space, so the
                 * CUDA driver can determine, for each pointer, where the data is
                 * located, and one does not have to explicitly specify this.
                 * @tparam T The pointer type.
                 * @param dest A pointer on host memory or on a device's global memory.
                 * @param src A pointer on host memory or on a device's global memory.
                 * @param count The number of elements to copy from source to destination.
                 */
                template <typename T>
                inline void copy(T *dest, const T *src, size_t count = 1)
                {
                    cuda::check(cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDefault));
                }

                /**
                 * Synchronously sets all bytes in a region of memory to a value.
                 * @param ptr The position from where region starts.
                 * @param value The value to set the memory region to.
                 * @param bytes The number of bytes to set.
                 */
                inline void set(void *ptr, uint8_t value, size_t bytes = 1)
                {
                    cuda::check(cudaMemset(ptr, value, bytes));
                }

                /**
                 * Synchronously sets all bytes in a region of memory to zero.
                 * @param ptr The position from where to start.
                 * @param bytes The size of the memory region in bytes.
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
                cuda::check(cudaDeviceSynchronize());
            }
        #endif

        /**
         * CUDA's NVCC allows use the use of the warpSize identifier, without having
         * to define it. Un(?)fortunately, warpSize is not a compile-time constant;
         * it is replaced at some point with the appropriate immediate value which
         * goes into, the SASS instruction as a literal. This is apparently due
         * to the theoretical possibility of different warp sizes in the future.
         * However, it is useful, both for host-side and more importantly for device-side
         * code, to have the warp size available at compile time. This allows all
         * sorts of useful optimizations, as well as its use in constexpr code.
         * @since 0.1.1
         */
        enum : word { warp_size = 32 };
    }
}

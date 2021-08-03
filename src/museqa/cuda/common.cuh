/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Miscellaneous utilities and function wrappers for CUDA.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <string>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/exception.hpp>
#include <museqa/environment.h>

#if defined(MUSEQA_COMPILER_NVCC)
  #include <cuda.h>

  /*
   * Checks whether the compilation is targeting a minimally compatible device.
   * If not, compilation must fail and an error must be shown.
   */
  #if defined(MUSEQA_RUNTIME_DEVICE) && (__CUDA_ARCH__ < 200)
    #error CUDA compilation must target devices of compute capability 2.0 or higher
  #endif
#endif

namespace museqa
{
    namespace cuda
    {
        /**
         * The native CUDA word type. Although this might type be changed in future
         * architectures, it is good enough for now.
         * @since 1.0
         */
        using word = uint32_t;

        namespace error
        {
            /**
             * Indicates either the result of a successful or failed CUDA-runtime
             * API call, or the overall status of the runtime API, which is typically
             * the last triggered error.
             * @since 1.0
             */
          #if defined(MUSEQA_COMPILER_NVCC)
            using code = decltype(cudaSuccess);
          #else
            using code = word;
          #endif

          #if defined(MUSEQA_COMPILER_NVCC)
            /**
             * Enumerates all CUDA error types and codes that can be returned by
             * the CUDA API.
             * @since 1.0
             */
            enum : std::underlying_type<code>::type
            {
                success                        = cudaSuccess
              , assert                         = cudaErrorAssert
              , api_failure_base               = cudaErrorApiFailureBase
              , address_of_constant            = cudaErrorAddressOfConstant
              , cudaruntime_unloading          = cudaErrorCudartUnloading
              , device_already_in_use          = cudaErrorDeviceAlreadyInUse
              , devices_unavailable            = cudaErrorDevicesUnavailable
              , duplicate_surface_name         = cudaErrorDuplicateSurfaceName
              , duplicate_texture_name         = cudaErrorDuplicateTextureName
              , duplicate_variable_name        = cudaErrorDuplicateVariableName
              , ecc_uncorrectable              = cudaErrorECCUncorrectable
              , hardware_stack_error           = cudaErrorHardwareStackError
              , host_memory_already_registered = cudaErrorHostMemoryAlreadyRegistered
              , host_memory_not_registered     = cudaErrorHostMemoryNotRegistered
              , illegal_address                = cudaErrorIllegalAddress
              , illegal_instruction            = cudaErrorIllegalInstruction
              , incompatible_driver_context    = cudaErrorIncompatibleDriverContext
              , initialization_error           = cudaErrorInitializationError
              , insufficient_driver            = cudaErrorInsufficientDriver
              , invalid_address_space          = cudaErrorInvalidAddressSpace
              , invalid_channel_descriptor     = cudaErrorInvalidChannelDescriptor
              , invalid_configuration          = cudaErrorInvalidConfiguration
              , invalid_device                 = cudaErrorInvalidDevice
              , invalid_device_function        = cudaErrorInvalidDeviceFunction
              , invalid_device_pointer         = cudaErrorInvalidDevicePointer
              , invalid_filter_setting         = cudaErrorInvalidFilterSetting
              , invalid_graphics_context       = cudaErrorInvalidGraphicsContext
              , invalid_host_pointer           = cudaErrorInvalidHostPointer
              , invalid_kernel_image           = cudaErrorInvalidKernelImage
              , invalid_memcpy_direction       = cudaErrorInvalidMemcpyDirection
              , invalid_norm_setting           = cudaErrorInvalidNormSetting
              , invalid_pc                     = cudaErrorInvalidPc
              , invalid_pitch_value            = cudaErrorInvalidPitchValue
              , invalid_ptx                    = cudaErrorInvalidPtx
              , invalid_resource_handle        = cudaErrorInvalidResourceHandle
              , invalid_surface                = cudaErrorInvalidSurface
              , invalid_symbol                 = cudaErrorInvalidSymbol
              , invalid_texture                = cudaErrorInvalidTexture
              , invalid_texture_binding        = cudaErrorInvalidTextureBinding
              , invalid_value                  = cudaErrorInvalidValue
              , launch_failure                 = cudaErrorLaunchFailure
              , launch_file_scoped_tex         = cudaErrorLaunchFileScopedTex
              , launch_file_scoped_surf        = cudaErrorLaunchFileScopedSurf
              , launch_max_depth_exceeded      = cudaErrorLaunchMaxDepthExceeded
              , launch_pending_count_exceeded  = cudaErrorLaunchPendingCountExceeded
              , launch_out_of_resources        = cudaErrorLaunchOutOfResources
              , launch_timeout                 = cudaErrorLaunchTimeout
              , map_buffer_object_failed       = cudaErrorMapBufferObjectFailed
              , memory_allocation              = cudaErrorMemoryAllocation
              , memory_value_too_large         = cudaErrorMemoryValueTooLarge
              , misaligned_address             = cudaErrorMisalignedAddress
              , missing_configuration          = cudaErrorMissingConfiguration
              , mixed_device_execution         = cudaErrorMixedDeviceExecution
              , no_device                      = cudaErrorNoDevice
              , no_kernel_image_for_device     = cudaErrorNoKernelImageForDevice
              , not_permitted                  = cudaErrorNotPermitted
              , not_ready                      = cudaErrorNotReady
              , not_supported                  = cudaErrorNotSupported
              , not_yet_implemented            = cudaErrorNotYetImplemented
              , operating_system               = cudaErrorOperatingSystem
              , peer_access_already_enabled    = cudaErrorPeerAccessAlreadyEnabled
              , peer_access_not_enabled        = cudaErrorPeerAccessNotEnabled
              , peer_access_unsupported        = cudaErrorPeerAccessUnsupported
              , prior_launch_failure           = cudaErrorPriorLaunchFailure
              , profiler_already_started       = cudaErrorProfilerAlreadyStarted
              , profiler_already_stopped       = cudaErrorProfilerAlreadyStopped
              , profiler_disabled              = cudaErrorProfilerDisabled
              , profiler_not_initialized       = cudaErrorProfilerNotInitialized
              , set_on_active_process          = cudaErrorSetOnActiveProcess
              , shared_object_init_failed      = cudaErrorSharedObjectInitFailed
              , shared_object_symbol_not_found = cudaErrorSharedObjectSymbolNotFound
              , startup_failure                = cudaErrorStartupFailure
              , sync_depth_exceeded            = cudaErrorSyncDepthExceeded
              , synchronization_error          = cudaErrorSynchronizationError
              , texture_fetch_failed           = cudaErrorTextureFetchFailed
              , texture_not_bound              = cudaErrorTextureNotBound
              , too_many_peers                 = cudaErrorTooManyPeers
              , unmap_buffer_object_failed     = cudaErrorUnmapBufferObjectFailed
              , unsupported_limit              = cudaErrorUnsupportedLimit
              , unknown                        = cudaErrorUnknown
            };
          #endif

          #if defined(MUSEQA_COMPILER_NVCC)
            /**
             * Clears the last error and resets it to success.
             * @return The last error triggered by a runtime call.
             */
            inline auto clear() noexcept -> error::code
            {
                return cudaGetLastError();
            }

            /**
             * Retrieves the last error code resulting from a CUDA runtime call
             * in the current host thread.
             * @return The last error registered by a runtime call.
             */
            inline auto last() noexcept -> error::code
            {
                return cudaPeekAtLastError();
            }
          #endif

            extern auto describe(error::code) noexcept -> std::string;
        }

        /**
         * Represents a CUDA-error detected during execution that be thrown and
         * propagated through the code carrying an error message.
         * @since 1.0
         */
        class exception : public museqa::exception
        {
          protected:
            typedef museqa::exception underlying_type;      /// The underlying exception type.

          protected:
            error::code m_err;                              /// The error code detected during execution.

          public:
            /**
             * Builds a new exception instance.
             * @param err The error code reported by the CUDA runtime.
             */
            inline exception(error::code err) noexcept
              : underlying_type {"cuda exception: {}", error::describe(err)}
              , m_err {err}
            {}

            /**
             * Builds a new exception instance from an error code and message.
             * @tparam T The format parameters' types.
             * @param err The error code.
             * @param fmtstr The exception message's format.
             * @param args The format string's parameters.
             */
            template <typename ...T>
            inline exception(error::code err, const std::string& fmtstr, T&&... args) noexcept
              : underlying_type {fmtstr, std::forward<decltype(args)>(args)...}
              , m_err {err}
            {}

            using underlying_type::exception;

            /**
             * Retrieves the CUDA error code wrapped by the exception.
             * @return The error code.
             */
            inline error::code code() const noexcept
            {
                return m_err;
            }
        };

      #if defined(MUSEQA_COMPILER_NVCC)
        /**
         * Asserts whether the given condition is met and throws exception otherwise.
         * @tparam E The exception type to be raised in case of error.
         * @tparam T The exception's parameters' types.
         * @param condition The condition that must be evaluated as true.
         * @param params The assertion exception's parameters.
         */
        template <typename E = cuda::exception, typename ...T>
        inline void ensure(bool condition, T&&... params) noexcept(!safe)
        {
            museqa::ensure<E>(condition, std::forward<decltype(params)>(params)...);
        }

        /**
         * Checks whether a CUDA has been successful and throws error otherwise.
         * @param code The status code obtained from a function.
         * @throw The error status code obtained raised to exception.
         */
        inline void check(error::code err) noexcept(!safe)
        {
            cuda::ensure((error::code) error::success == err, err);
        }
      #endif
    }
}

#endif

/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA device utilities and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <cuda.h>

#include <museqa/utility.hpp>
#include <museqa/cuda/common.cuh>

namespace museqa
{
    namespace cuda
    {
        namespace device
        {
            /**
             * The CUDA device identification type. Here, we simply define a device
             * as its numeric identifier, which is usefull for breaking dependencies
             * between distinct objects and easier interation with code using the
             * original CUDA APIs.
             * @since 1.0
             */
            using id = cuda::word;

            /**
             * If the CUDA runtime has not yet been set to a specific device, this
             * is the device id it defaults to.
             * @since 1.0
             */
            enum : id { default_device = 0 };

            namespace memory
            {
                extern auto available() noexcept(!safe) -> size_t;
                extern auto available(device::id) noexcept(!safe) -> size_t;

                extern auto total() noexcept(!safe) -> size_t;
                extern auto total(device::id) noexcept(!safe) -> size_t;
            }

          #if defined(MUSEQA_COMPILER_NVCC)
            /**
             * Hardware, firmware and physical properties about and attributes of
             * a CUDA compute-capable device. Although extremely useful, the request
             * for the properties of a device is expensive due to the overwhelming
             * amount of information gathered about it.
             * @since 1.0
             */
            struct properties : public cudaDeviceProp
            {
                __host__ inline properties(device::id device) noexcept(!safe)
                {
                    cuda::check(cudaGetDeviceProperties(this, device));
                }
            };
          #endif

            namespace current
            {
                /**
                 * A RAII-based mechanism for setting the CUDA runtime API's current
                 * device for what remains of the scope, changing it back to the
                 * previous device when exiting the scope.
                 * @since 1.0
                 */
                class scope
                {
                  private:
                    const device::id m_previous = device::default_device;

                  public:
                    /**
                     * Sets the given device as the currently active one and remembers
                     * the one previously active for restoring after scope end.
                     * @param target The target device to be currently active.
                     */
                    inline scope(device::id target) noexcept(!safe)
                      : m_previous {replace(target)}
                    {}

                    /**
                     * Restores the previously active device. Although we cannot
                     * guarantee the device has been manually changed after the
                     * constructor, we set it back to what it was.
                     */
                    inline ~scope() noexcept(!safe)
                    {
                        replace(m_previous);
                    }

                  protected:
                    static device::id replace(device::id) noexcept(!safe);
                };

                extern __host__ __device__ auto get() noexcept(!safe) -> device::id;
                extern void set(device::id = device::default_device) noexcept(!safe);

              #if defined(MUSEQA_COMPILER_NVCC)
                /**
                 * Retrieves hardware, firmware and physical properties about the
                 * CUDA compute-capable device that is currently active.
                 * @note This function is expensive and should be called with care.
                 * @return The retrieved device properties.
                 */
                inline auto properties() noexcept(!safe) -> properties
                {
                    return device::properties {device::current::get()};
                }
              #endif
            }

            extern __host__ __device__ auto count() noexcept(!safe) -> size_t;
        }
    }
}

#endif

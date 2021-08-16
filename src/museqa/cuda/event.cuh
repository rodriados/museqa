/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA event utilities and object implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#if !defined(MUSEQA_AVOID_CUDA) && defined(MUSEQA_COMPILER_NVCC)

#include <cuda.h>

#include <cstdint>

#include <museqa/utility.hpp>
#include <museqa/memory/pointer/shared.hpp>

#include <museqa/cuda/common.cuh>
#include <museqa/cuda/device.cuh>
#include <museqa/cuda/stream.cuh>

namespace museqa
{
    namespace cuda
    {
        /**
         * Represents a CUDA event, which are synchronization markers that can be
         * used to time asynchronous tasks in streams, fine grained synchronization
         * within a stream or inter-stream synchronization.
         * @since 1.0
         */
        class event : private museqa::memory::pointer::shared<void>
        {
          protected:
            typedef cudaEvent_t event_type;
            typedef cuda::stream::id stream_type;
            typedef museqa::memory::pointer::shared<void> underlying_type;

          public:
            using id = event_type;

          protected:
            const cuda::device::id m_device;    /// The device the event is associated with.

          public:
            inline event(const event&) noexcept = default;
            inline event(event&&) noexcept = default;

            /**
             * The list of all flags available for event creation.
             * @since 1.0
             */
            enum flag : uint32_t {
                blocking_sync   = cudaEventBlockingSync
              , disable_timing  = cudaEventDisableTiming
              , interprocess    = cudaEventInterprocess
            };

            /**
             * Creates a new CUDA event associated to the currently active device.
             * @param flags The flags to create a new event with.
             */
            inline event(uint32_t flags = 0) noexcept(!safe)
              : underlying_type {create(flags)}
              , m_device {cuda::device::current::get()}
            {}

            /**
             * Creates a new CUDA event associated to the given device.
             * @param device The device to associate the new event with.
             * @param flags The flags to create a new event with.
             */
            inline event(cuda::device::id device, uint32_t flags = 0) noexcept(!safe)
              : underlying_type {(cuda::device::current::scope{device}, create(flags))}
              , m_device {device}
            {}

            inline event& operator=(const event&) = default;
            inline event& operator=(event&&) = default;

            /**
             * Converts this object into a raw CUDA event reference, thus allowing
             * the user to seamlessly use the event with native CUDA functions.
             * @return The raw CUDA event reference.
             */
            inline operator event_type() const noexcept
            {
                return (event_type) this->m_ptr;
            }

            static bool ready(const event&) noexcept(!safe);
            static void record(event&, stream_type = cuda::stream::default_stream) noexcept(!safe);
            static void synchronize(const event&) noexcept(!safe);

          private:
            /**
             * Creates a new raw event object with the given flags.
             * @param flags The flags for the new event creation.
             * @return The internal event pointer instance.
             */
            inline auto create(uint32_t flags) noexcept -> underlying_type
            {
                cuda::check(cudaEventCreateWithFlags((event_type*) &this->m_ptr, flags));
                auto destructor = [](void *ptr) { cuda::check(cudaEventDestroy((event_type) ptr)); };
                return underlying_type {this->m_ptr, destructor};
            }
        };
    }
}

#endif

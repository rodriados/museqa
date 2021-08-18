/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA stream utilities and object implementation.
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

namespace museqa
{
    namespace cuda
    {
        /**
         * Represents an asynchronous CUDA stream, which allows simutaneous parallel
         * execution of non-blocking tasks in one or multiple devices.
         * @since 1.0
         */
        class stream : private museqa::memory::pointer::shared<void>
        {
          protected:
            typedef cudaEvent_t event_type;
            typedef cudaStream_t stream_type;
            typedef museqa::memory::pointer::shared<void> underlying_type;

          public:
            using id = stream_type;
            static constexpr stream_type default_stream = nullptr;

          protected:
            cuda::device::id m_device = cuda::device::default_device;

          public:
            inline stream(const stream&) noexcept = default;
            inline stream(stream&&) noexcept = default;

            /**
             * The list of all flags available for stream creation.
             * @since 1.0
             */
            enum flag : uint32_t {
                non_blocking    = cudaStreamNonBlocking
            };

            /**
             * Creates a new CUDA stream associated with the currently active device.
             * @param priority The new stream's priority value.
             * @param flags The flags to create a new stream with.
             * @note Lower priority values means higher stream priority.
             */
            inline stream(int priority = 0, uint32_t flags = non_blocking) noexcept(!safe)
              : underlying_type {create(priority, flags)}
              , m_device {cuda::device::current::get()}
            {}

            /**
             * Creates a new CUDA stream associated with the given device.
             * @param device The device to associate the new stream with.
             * @param priority The new stream's priority value.
             * @param flags The flags to create a new stream with.
             * @note Lower priority values means higher stream priority.
             */
            inline stream(cuda::device::id device, int priority = 0, uint32_t flags = non_blocking) noexcept(!safe)
              : underlying_type {(cuda::device::current::scope{device}, create(priority, flags))}
              , m_device {device}
            {}

            inline stream& operator=(const stream&) = default;
            inline stream& operator=(stream&&) = default;

            /**
             * Converts this object into a raw CUDA stream reference, thus allowing
             * the user to seamlessly use the stream with native CUDA functions.
             * @return The raw CUDA stream identifier.
             */
            inline operator stream_type() const noexcept
            {
                return (stream_type) this->m_ptr;
            }

            static bool ready(const stream&) noexcept(!safe);
            static void synchronize(const stream&) noexcept(!safe);
            static void wait(const stream&, event_type) noexcept(!safe);

          private:
            /**
             * Creates a new stream within CUDA's internal state.
             * @param priority The new stream's priority value.
             * @param flags The flags to create a new stream with.
             * @return The internal stream pointer instance.
             */
            inline auto create(int priority, uint32_t flags) noexcept(!safe) -> underlying_type
            {
                cuda::check(cudaStreamCreateWithPriority((stream_type*) &this->m_ptr, flags, priority));
                auto destructor = [](void *stream) { cuda::check(cudaStreamDestroy((stream_type) stream)); };
                return underlying_type {this->m_ptr, destructor};
            }
        };
    }
}

#endif

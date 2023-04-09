/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a pointer metadata storage and instance counter.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/wrapper.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer::detail
{
    /**
     * The metadata and reference counter for a smart pointer implementation to
     * a given memory pointer. Whenever the reference count reaches zero, the pointer's
     * deallocator is automatically called and the memory is released.
     * @since 1.0
     */
    class metadata_t
    {
        private:
            typedef memory::allocator_t allocator_t;
            typedef memory::pointer::wrapper_t<void> pointer_t;

        private:
            pointer_t m_ptr = nullptr;
            const allocator_t m_allocator {};
            uint64_t m_counter = 0;

        public:
            /**
             * Creates a new pointer context metadata from given arguments.
             * @param ptr The pointer to be tracked into the context.
             * @param allocator The pointer's allocator.
             * @return The new pointer's metadata instance.
             */
            inline static metadata_t *acquire(pointer_t ptr, const allocator_t& allocator) noexcept
            {
                return new metadata_t {ptr, allocator};
            }

            /**
             * Acquires ownership of an already existing pointer context.
             * @param meta The metadata instance of pointer to be acquired.
             * @return The acquired metadata pointer.
             */
            __host__ __device__ inline static metadata_t *acquire(metadata_t *meta) noexcept
            {
              #if MUSEQA_RUNTIME_HOST
                if (meta != nullptr)
                    ++meta->m_counter;
                return meta;
              #else
                return nullptr;
              #endif
            }

            /**
             * Releases ownership of a pointer context, and frees all allocated
             * memory resources when the reference counter reaches zero.
             * @param meta The metadata of a pointer to be released.
             */
            __host__ __device__ inline static void release(metadata_t *meta) __devicesafe__
            {
              #if MUSEQA_RUNTIME_HOST
                if (meta != nullptr && --meta->m_counter <= 0)
                    delete meta;
              #endif
            }

        private:
            /**
             * Creates a new pointer context metadata from a raw pointer instance.
             * @param ptr The raw pointer to acquire ownership of.
             * @param allocator The pointer's allocator instance.
             */
            inline explicit metadata_t(pointer_t ptr, const allocator_t& allocator) noexcept
              : m_ptr (ptr)
              , m_allocator (allocator)
              , m_counter (1)
            {}

            /**
             * Releases ownership and frees the resources allocated by the pointer.
             * @see museqa::memory::pointer::detail::metadata::release
             */
            inline ~metadata_t()
            {
                if (m_ptr != nullptr)
                    m_allocator.deallocate(static_cast<void*>(m_ptr));
            }
    };
}

MUSEQA_END_NAMESPACE

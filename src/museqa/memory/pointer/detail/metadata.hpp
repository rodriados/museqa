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
#include <museqa/memory/pointer/container.hpp>

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
            typedef memory::deleter_t deleter_t;
            typedef memory::pointer::container_t<void> container_t;

        private:
            int64_t m_counter = 0;
            container_t m_ptr = nullptr;
            const deleter_t m_deleter {};

        public:
            /**
             * Creates a new pointer context metadata from given arguments.
             * @param ptr The pointer container to be tracked in the context.
             * @param deleter The pointer's deleter.
             * @return The new pointer's metadata instance.
             */
            MUSEQA_INLINE static metadata_t *acquire(
                container_t ptr
              , const deleter_t& deleter
            ) noexcept {
                return !ptr.empty()
                    ? new metadata_t(ptr, deleter)
                    : nullptr;
            }

            /**
             * Acquires ownership of an already existing pointer context.
             * @param meta The metadata instance of pointer to be acquired.
             * @return The acquired metadata pointer.
             */
            MUSEQA_CUDA_INLINE static metadata_t *acquire(metadata_t *meta) noexcept
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
            MUSEQA_CUDA_INLINE static void release(metadata_t *meta) MUSEQA_SAFE_EXCEPT
            {
              #if MUSEQA_RUNTIME_HOST
                if (meta != nullptr && --meta->m_counter <= 0)
                    delete meta;
              #endif
            }

        private:
            /**
             * Creates a new pointer context metadata from a raw pointer instance.
             * @param ptr The raw pointer container to acquire ownership of.
             * @param deleter The pointer's deleter instance.
             */
            MUSEQA_INLINE explicit metadata_t(container_t ptr, const deleter_t& deleter) noexcept
              : m_counter (1)
              , m_ptr (ptr)
              , m_deleter (deleter)
            {}

            /**
             * Releases ownership and frees the resources allocated by the pointer.
             * @see museqa::memory::pointer::detail::metadata_t::release
             */
            MUSEQA_INLINE ~metadata_t()
            {
                if (!m_ptr.empty() && !m_deleter.empty()) {
                    m_deleter.deallocate(m_ptr.unwrap());
                }
            }
    };
}

MUSEQA_END_NAMESPACE

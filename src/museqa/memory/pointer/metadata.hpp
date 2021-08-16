/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements a pointer metadata storage and instance counter.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>

#include <museqa/utility.hpp>
#include <museqa/memory/allocator.hpp>

namespace museqa
{
    namespace memory
    {
        namespace pointer
        {
            /**
             * Keeps track of the metadata and the number of references to a given
             * memory pointer. Whenever the number of references to a pointer reaches
             * zero, it is automatically deallocated.
             * @since 1.0
             */
            class metadata
            {
              private:
                typedef memory::allocator allocator_type;

              private:
                void *m_ptr = nullptr;                          /// The raw wrapped pointer.
                size_t m_counter = 0;                           /// The number of pointer instances.
                const allocator_type m_allocator;               /// The pointer's allocator.

              public:
                /**
                 * Creates a new pointer context from given arguments.
                 * @param ptr The pointer to be wrapped into the context.
                 * @param allocator The pointer's allocator.
                 * @return The new pointer's metadata instance.
                 */
                inline static auto acquire(void *ptr, const allocator_type& allocator) noexcept -> metadata*
                {
                    return new metadata {ptr, allocator};
                }

                /**
                 * Acquires ownership of an already existing pointer.
                 * @param meta The metadata of pointer to be acquired.
                 * @return The acquired metadata pointer.
                 */
                __host__ __device__ inline static auto acquire(metadata *meta) noexcept -> metadata*
                {
                  #if defined(MUSEQA_RUNTIME_HOST)
                    if (nullptr != meta) { ++meta->m_counter; }
                  #endif
                    return meta;
                }

                /**
                 * Releases ownership of a pointer, and deletes it if needed.
                 * @param meta The metadata of pointer to be released.
                 */
                __host__ __device__ inline static void release(metadata *meta)
                {
                  #if defined(MUSEQA_RUNTIME_HOST)
                    if (nullptr != meta && --meta->m_counter <= 0) { delete meta; }
                  #endif
                }

              private:
                /**
                 * Initializes a new pointer metadata.
                 * @param ptr The raw pointer to be handled.
                 * @param allocator The pointer's allocator.
                 */
                inline explicit metadata(void *ptr, const allocator_type& allocator) noexcept
                  : m_ptr {ptr}
                  , m_counter {1}
                  , m_allocator {allocator}
                {}

                /**
                 * Deletes and frees the memory occupied by the raw pointer.
                 * @see museqa::memory::pointer::impl::metadata::release
                 */
                inline ~metadata()
                {
                    if (nullptr != m_ptr) { m_allocator.deallocate(m_ptr); }
                }
            };
        }
    }
}

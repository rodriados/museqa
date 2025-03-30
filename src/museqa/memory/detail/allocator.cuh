/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The type-erased abstract base allocator declaration.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace memory::detail
{
    /**
     * The abstract base for a generic allocator.
     * Every dynamic memory allocation shall be performed by an instance of a specific
     * implementation of this generic allocator.
     * @since 1.0
     */
    struct allocator_t
    {
        MUSEQA_INLINE allocator_t() noexcept = default;
        MUSEQA_INLINE allocator_t(const allocator_t&) noexcept = default;
        MUSEQA_INLINE allocator_t(allocator_t&&) noexcept = default;

        MUSEQA_INLINE allocator_t& operator=(const allocator_t&) noexcept = default;
        MUSEQA_INLINE allocator_t& operator=(allocator_t&&) noexcept = default;

        MUSEQA_INLINE virtual ~allocator_t() = default;

        /**
         * Allocates the requested amount of memory into a pointer.
         * @param size The size of each element to be allocated.
         * @param count The number of elements to allocate memory to.
         * @return The pointer to the newly allocated memory region.
         */
        virtual void* allocate(size_t size, size_t count) const = 0;
    };

    /**
     * The abstract base for a generic deallocator.
     * Every dynamic memory allocation shall be deallocated by an instance of a specific
     * implementation of this generic deallocator.
     * @since 1.0
     */
    struct deleter_t
    {
        MUSEQA_INLINE deleter_t() noexcept = default;
        MUSEQA_INLINE deleter_t(const deleter_t&) noexcept = default;
        MUSEQA_INLINE deleter_t(deleter_t&&) noexcept = default;

        MUSEQA_INLINE deleter_t& operator=(const deleter_t&) noexcept = default;
        MUSEQA_INLINE deleter_t& operator=(deleter_t&&) noexcept = default;

        MUSEQA_INLINE virtual ~deleter_t() = default;

        /**
         * Deallocates a memory region previously allocated.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        virtual void deallocate(void* ptr) const = 0;
    };
}

MUSEQA_END_NAMESPACE

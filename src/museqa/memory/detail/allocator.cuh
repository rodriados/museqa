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
     * The abstract base for a generic memory allocator.
     * Every dynamic memory allocation shall be performed by an instance of a specific
     * implementation of this generic allocator.
     * @since 1.0
     */
    struct allocator_t
    {
        MUSEQA_CONSTEXPR allocator_t() noexcept = default;
        MUSEQA_CONSTEXPR allocator_t(const allocator_t&) noexcept = default;
        MUSEQA_CONSTEXPR allocator_t(allocator_t&&) noexcept = default;

        MUSEQA_INLINE allocator_t& operator=(const allocator_t&) noexcept = default;
        MUSEQA_INLINE allocator_t& operator=(allocator_t&&) noexcept = default;

        MUSEQA_INLINE virtual ~allocator_t() = default;

        /**
         * Allocates a memory region for a number of generic elements.
         * @param count The number of elements to allocate memory to.
         * @param size The size of each element to be allocated.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_CUDA_ENABLED virtual void* allocate(size_t count, size_t size) const = 0;
    };

    /**
     * The abstract base for a generic memory deallocator.
     * Every dynamic memory allocation shall be deallocated by an instance of a specific
     * implementation of this generic deallocator.
     * @since 1.0
     */
    struct deleter_t
    {
        MUSEQA_CONSTEXPR deleter_t() noexcept = default;
        MUSEQA_CONSTEXPR deleter_t(const deleter_t&) noexcept = default;
        MUSEQA_CONSTEXPR deleter_t(deleter_t&&) noexcept = default;

        MUSEQA_INLINE deleter_t& operator=(const deleter_t&) noexcept = default;
        MUSEQA_INLINE deleter_t& operator=(deleter_t&&) noexcept = default;

        MUSEQA_INLINE virtual ~deleter_t() = default;

        /**
         * Deallocates a generic memory region.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        MUSEQA_CUDA_ENABLED virtual void deallocate(void* ptr, size_t) const = 0;
    };
}

MUSEQA_END_NAMESPACE

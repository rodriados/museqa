/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A CUDA-enabled generic memory allocator implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.cuh>

#include <museqa/memory/detail/allocator.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Generic dynamic memory allocator and deleter for the type T.
     * @tparam T The type to dynamically allocate and deallocate memory for.
     * @since 1.0
     */
    template <typename T = void>
    struct allocator_t : detail::allocator_t, detail::deleter_t
    {
        using element_t = pure_t<T>;

        using detail::allocator_t::allocate;
        using detail::deleter_t::deallocate;

        /**
         * Allocates memory for type T by invoking the concrete allocator implementation.
         * @param count The number of elements to allocate memory for.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_CUDA_INLINE auto allocate(size_t count) const -> element_t*
        {
            using proxy_t = std::conditional_t<std::is_void_v<T>, uint8_t, element_t>;
            return static_cast<element_t*>(allocate(count, sizeof(proxy_t)));
        }

        /**
         * Deallocates memory of type T by invoking the concrete deleter implementation.
         * @param ptr The pointer to the memory to be deallocated.
         */
        MUSEQA_CUDA_INLINE void deallocate(element_t* ptr) const
        {
            deallocate(static_cast<void*>(ptr), 0);
        }
    };

    /**
     * Indicates whether the given type is an allocator for the type T.
     * @tparam T The allocator element type.
     * @tparam A The type to check if it is an allocator.
     * @since 1.0
     */
    template <typename T, typename A>
    MUSEQA_CONSTEXPR bool is_allocator = std::is_base_of_v<allocator_t<T>, A>;

    /**
     * The default memory allocator implementation for a generic type T.
     * @tparam T The type to dynamically allocate and deallocate memory for.
     * @see museqa::memory::allocator_t
     * @since 1.0
     */
    template <typename T>
    struct default_allocator_t final : allocator_t<T>
    {
        using allocator_t<T>::allocate;
        using allocator_t<T>::deallocate;

        using typename allocator_t<T>::element_t;

        /**
         * Allocates a memory region for elements of the type T.
         * @param count The number of elements to allocate memory for.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_CUDA_INLINE void* allocate(size_t count, size_t) const override
        {
            return static_cast<void*>(new element_t[count]);
        }

        /**
         * Deallocates a memory region of elements of the type T.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        MUSEQA_CUDA_INLINE void deallocate(void* ptr, size_t) const override
        {
            delete[] static_cast<element_t*>(ptr);
        }
    };

    /**
     * The default memory allocator implementation for an unknown type.
     * @see museqa::memory::allocator_t
     * @since 1.0
     */
    template <>
    struct default_allocator_t<void> final : allocator_t<void>
    {
        using allocator_t<void>::allocate;
        using allocator_t<void>::deallocate;

        /**
         * Allocates a memory region of an unknown type.
         * @param count The number of elements to allocate memory for.
         * @param size The size, in bytes, of each element to be allocated.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_CUDA_INLINE void* allocate(size_t count, size_t size) const override
        {
            return operator new(count * size);
        }

        /**
         * Deallocates a memory region of an unknown type.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        MUSEQA_CUDA_INLINE void deallocate(void* ptr, size_t) const override
        {
            operator delete(ptr);
        }
    };

    /**
     * Creates a default allocator for the specified type, which must be well-formed,
     * concrete and publicly default-constructible.
     * @tparam T The type to make an allocator for.
     * @return An allocator for the given type.
     */
    template <typename T = void>
    MUSEQA_CUDA_CONSTEXPR auto make_allocator() noexcept
    {
        return default_allocator_t<T>();
    }

    /**
     * Creates a custom allocator for the specified type with the given functors.
     * @tparam T The type to make an allocator for.
     * @tparam C The functor-type of the custom memory allocator.
     * @tparam D The functor-type of the custom memory deallocator.
     * @param allocate The functor to use for memory allocation.
     * @param deallocate The functor to use for memory deallocation.
     * @return A custom allocator for the given type.
     */
    template <typename T = void, typename C, typename D>
    MUSEQA_CUDA_CONSTEXPR auto make_allocator(const C& allocate, const D& deallocate) noexcept
    {
        /**
         * The custom memory allocator implementation, with generic functor injection
         * specialized for the type T. This allocator type allows any functor to be
         * injected and used as allocator routine for a memory region of type T.
         * @since 1.0
         */
        struct custom_allocator_t final : allocator_t<T> {
            const C m_func_allocate;
            const D m_func_deallocate;

            using typename allocator_t<T>::element_t;

            /**
             * Instantiates the custom allocator with the given functors.
             * @param allocate The functor to use for memory allocation.
             * @param deallocate The functor to use for memory deallocation.
             */
            MUSEQA_CUDA_CONSTEXPR custom_allocator_t(const C& allocate, const D& deallocate)
              : m_func_allocate (allocate)
              , m_func_deallocate (deallocate)
            {}

            using allocator_t<T>::allocate;
            using allocator_t<T>::deallocate;

            /**
             * Invokes the functor for allocating memory for elements of type T.
             * @param count The number of elements to allocate memory to.
             * @param size The size of each element to be allocated.
             * @return The pointer to the newly allocated memory region.
             */
            MUSEQA_CUDA_INLINE void* allocate(size_t count, size_t size) const override {
                return static_cast<void*>(utility::invoke(m_func_allocate, count, size));
            }

            /**
             * Invokes the functor for deallocating memory of elements of type T.
             * @param ptr The pointer to the memory region to be deallocated.
             */
            MUSEQA_CUDA_INLINE void deallocate(void* ptr, size_t) const override {
                utility::invoke(m_func_deallocate, static_cast<element_t*>(ptr));
            }
        };

        return custom_allocator_t(allocate, deallocate);
    }
}

MUSEQA_END_NAMESPACE

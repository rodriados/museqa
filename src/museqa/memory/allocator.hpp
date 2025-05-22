/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A generic memory allocator implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Abstract base for a generic memory allocator.
     * Every dynamic memory allocation and/or deallocation to a smart pointer handled
     * by the library shall be performed by an instance of a specific implementation
     * of this generic allocator.
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
         * Allocate a memory region for a number of generic elements.
         * @param count The number of elements to allocate memory for.
         * @param size The size of each element to be allocated.
         * @param align The alignment requirement for the element type.
         * @return The pointer to the newly allocated memory region.
         */
        virtual void* allocate(size_t count, size_t size, size_t align) const = 0;

        /**
         * Deallocate a generic memory region.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        virtual void deallocate(void* ptr, size_t) const = 0;
    };

    /**
     * Check if the type is an allocator.
     * @tparam T The type to be check if it is an allocator.
     * @since 1.0
     */
    template <typename T>
    MUSEQA_CONSTEXPR bool is_allocator = std::is_base_of_v<allocator_t, T>;

    /**
     * Abstract allocator implementation for a generic type T.
     * @tparam T The type to allocate and deallocate memory for.
     * @see museqa::memory::allocator_t
     * @since 1.0
     */
    template <typename T = void>
    struct typed_allocator_t : public allocator_t
    {
        typedef pure_t<T> element_t;

        using allocator_t::allocate;
        using allocator_t::deallocate;

        static_assert(!std::is_function_v<T>, "unable to create allocator for function types");
        static_assert(!std::is_reference_v<T>, "unable to create allocator for reference types");

        /**
         * Allocate memory for a number of elements of type T.
         * @param count The number of elements to allocate memory for.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_INLINE element_t* allocate(size_t count) const
        {
            using E = std::conditional_t<std::is_void_v<T>, uint8_t, element_t>;
            return (element_t*) allocate(count, sizeof(E), alignof(E));
        }

        /**
         * Deallocate memory region of elements of type T.
         * @param ptr The pointer to the memory to be deallocated.
         */
        MUSEQA_INLINE void deallocate(element_t* ptr) const
        {
            deallocate((void*) ptr, 0);
        }
    };

    /**
     * The default memory allocator implementation for a generic type T.
     * @tparam T The type to allocate and deallocate memory for.
     * @see museqa::memory::allocator_t
     * @since 1.0
     */
    template <typename T = void>
    struct default_allocator_t : public typed_allocator_t<T>
    {
        using typed_allocator_t<T>::allocate;
        using typed_allocator_t<T>::deallocate;
        using typename typed_allocator_t<T>::element_t;

        /**
         * Allocates a memory region for elements of the type T.
         * @param count The number of elements to allocate memory for.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_INLINE void* allocate(size_t count, size_t, size_t) const override
        {
            return (void*) new element_t[count];
        }

        /**
         * Deallocates a memory region of an unknown type.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        MUSEQA_INLINE void deallocate(void* ptr, size_t) const override
        {
            delete[] (element_t*) ptr;
        }
    };

    /**
     * The default memory allocator implementation for an unknown type.
     * @see museqa::memory::allocator_t
     * @since 1.0
     */
    template <>
    struct default_allocator_t<void> : public typed_allocator_t<void>
    {
        using typed_allocator_t<void>::allocate;
        using typed_allocator_t<void>::deallocate;

        /**
         * Allocates a memory region of an unknown type.
         * @param count The number of elements to allocate memory for.
         * @param size The size, in bytes, of each element to be allocated.
         * @param align The alignment requirement for the element type.
         * @return The pointer to the newly allocated memory region.
         */
        MUSEQA_INLINE void* allocate(size_t count, size_t size, size_t align) const override
        {
            return operator new(count * size, std::align_val_t(align));
        }

        /**
         * Deallocates a memory region of an unknown type.
         * @param ptr The pointer to the memory region to be deallocated.
         */
        MUSEQA_INLINE void deallocate(void* ptr, size_t) const override
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
    MUSEQA_CONSTEXPR auto make_allocator() noexcept
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
    MUSEQA_CONSTEXPR auto make_allocator(const C& allocate, const D& deallocate) noexcept
    {
        /**
         * The custom memory allocator implementation, with generic functor injection
         * specialized for the type T. This allocator type allows any functor to be
         * injected and used as allocator routine for a memory region of type T.
         * @since 1.0
         */
        struct custom_allocator_t final : public typed_allocator_t<T> {
            const C m_func_allocate;
            const D m_func_deallocate;

            using typename typed_allocator_t<T>::element_t;

            /**
             * Instantiates the custom allocator with the given functors.
             * @param allocate The functor to use for memory allocation.
             * @param deallocate The functor to use for memory deallocation.
             */
            MUSEQA_CONSTEXPR custom_allocator_t(const C& allocate, const D& deallocate)
              : m_func_allocate (allocate)
              , m_func_deallocate (deallocate)
            {}

            using typed_allocator_t<T>::allocate;
            using typed_allocator_t<T>::deallocate;

            /**
             * Invokes the functor for allocating memory for elements of type T.
             * @param count The number of elements to allocate memory to.
             * @param size The size of each element to be allocated.
             * @param align The alignment requirement for the element type.
             * @return The pointer to the newly allocated memory region.
             */
            MUSEQA_INLINE void* allocate(size_t count, size_t size, size_t align) const override {
                return (void*) utility::invoke(m_func_allocate, count, size, align);
            }

            /**
             * Invokes the functor for deallocating memory of elements of type T.
             * @param ptr The pointer to the memory region to be deallocated.
             */
            MUSEQA_INLINE void deallocate(void* ptr, size_t) const override {
                utility::invoke(m_func_deallocate, (element_t*) ptr);
            }
        };

        return custom_allocator_t(allocate, deallocate);
    }
}

MUSEQA_END_NAMESPACE

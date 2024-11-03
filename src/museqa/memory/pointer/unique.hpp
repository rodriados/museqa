/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A managed unique pointer container implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/container.hpp>
#include <museqa/memory/pointer/detail/metadata.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Implements a generic pointer with automatically managed lifetime duration
     * that cannot be owned by more than one unique instance at a time.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class unique_t : public memory::pointer::container_t<T>
    {
        template <typename> friend class unique_t;
        template <typename> friend class shared_t;

        public:
            typedef T element_t;
            typedef T *pointer_t;

        private:
            typedef memory::pointer::container_t<T> underlying_t;
            typedef memory::deleter_t deleter_t;

        private:
            deleter_t m_deleter {};

        public:
            MUSEQA_CONSTEXPR unique_t() noexcept = default;
            MUSEQA_CONSTEXPR unique_t(const unique_t&) = delete;

            MUSEQA_INLINE unique_t& operator=(const unique_t&) = delete;

            /**
             * Builds a new unique pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            MUSEQA_INLINE explicit unique_t(T *ptr, const allocator_t& allocator) noexcept
              : underlying_t (ptr)
              , m_deleter (allocator)
            {}

            /**
             * Acquire ownership of another container's pointer.
             * @param other The container to acquire ownership from.
             */
            MUSEQA_CUDA_INLINE unique_t(unique_t&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other));
            }

            /**
             * Acquire ownership of a foreign-typed container's pointer.
             * @tparam U The foreign container's element type.
             * @param other The foreign-typed container to acquire ownership from.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unique_t(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other));
            }

            /**
             * Releases ownership of the acquired pointer.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            MUSEQA_CUDA_INLINE ~unique_t() MUSEQA_SAFE_EXCEPT
            {
                if (!this->empty() && !m_deleter.empty())
                    m_deleter.deallocate(this->unwrap());
            }

            /**
             * Releases the currently owned pointer and acquires exclusive ownership
             * of another container's instance pointer.
             * @param other The container acquire ownership from.
             * @return The current unique pointer container.
             */
            MUSEQA_CUDA_INLINE unique_t& operator=(unique_t&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases the currently owned pointer and acquires exclusive ownership
             * of a foreign-typed container's instance pointer.
             * @tparam U The foreign pointer type to acquire.
             * @param other The foreign pointer to acquire ownership from.
             * @return The current unique pointer container.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unique_t& operator=(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases the pointer ownership and returns to an empty state.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            MUSEQA_CUDA_INLINE void reset() MUSEQA_SAFE_EXCEPT
            {
                auto ephemeral = unique_t();
                swap(ephemeral);
            }

            /**
             * Swaps ownership with another pointer instance.
             * @param other The instance to swap with.
             */
            MUSEQA_CUDA_INLINE void swap(unique_t& other) noexcept
            {
                underlying_t::swap(other);
                utility::swap(m_deleter, other.m_deleter);
            }

        private:
            template <typename U> MUSEQA_CUDA_INLINE void acquire(unique_t<U>&&) MUSEQA_SAFE_EXCEPT;
    };

    /**
     * Captures ownership from a pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be moved.
     * @param other The foreign pointer instance to be moved.
     */
    template <typename T> template <typename U>
    MUSEQA_CUDA_INLINE void unique_t<T>::acquire(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
    {
        static_assert(std::is_convertible_v<U*, T*>, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            utility::swap(this->m_ptr, other.m_ptr);
            utility::swap(this->m_deleter, other.m_deleter);
            other.reset();
        }
    }

    /*
     * Deduction guides for a generic unique pointer.
     * @since 1.0
     */
    template <typename T> unique_t(T*) -> unique_t<T>;
    template <typename T> unique_t(T*, const memory::allocator_t&) -> unique_t<T>;
}

namespace factory::memory::pointer
{
    /**
     * Allocates memory for the given element type into an unique pointer.
     * @tparam T The type to be allocated into a new pointer.
     * @param count The total number of elements to be allocated.
     * @param allocator The allocator to create the elements with.
     * @return The allocated unique memory pointer.
     */
    template <
        typename T = void
      , typename = std::enable_if_t<!std::is_reference_v<T>>>
    MUSEQA_INLINE museqa::memory::pointer::unique_t<T> unique(
        size_t count = 1
      , const museqa::memory::allocator_t& allocator = factory::memory::allocator<T>()
    ) {
        T *ptr = allocator.allocate<T>(count);
        return museqa::memory::pointer::unique_t(ptr, allocator);
    }
}

MUSEQA_END_NAMESPACE

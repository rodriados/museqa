/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A managed unique pointer container implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.cuh>

#include <museqa/memory/allocator.cuh>
#include <museqa/memory/detail/refcounter.cuh>
#include <museqa/memory/detail/refmetadata.cuh>
#include <museqa/memory/pointer/wrapper.cuh>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * A generic pointer wrapper with automatic lifetime duration management that
     * guarantees unique ownership at all times.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class unique_t : public wrapper_t<T>
    {
        public:
            using typename wrapper_t<T>::element_t;

        private:
            typedef wrapper_t<T> super_t;
            detail::refcounter_t *m_ref = nullptr;

        template <typename> friend class unique_t;
        template <typename> friend class shared_t;

        public:
            MUSEQA_CONSTEXPR unique_t() noexcept = default;
            MUSEQA_CONSTEXPR unique_t(const unique_t&) = delete;

            MUSEQA_INLINE unique_t& operator=(const unique_t&) = delete;

            /**
             * Instantiates a new pointer wrapper from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer allocator.
             */
            template <typename A, typename = std::enable_if_t<is_allocator<T, A>>>
            MUSEQA_CUDA_INLINE explicit unique_t(T *ptr, const A& allocator)
              : unique_t (ptr, detail::acquire_ownership<detail::refmetadata_t<A>>(ptr, allocator))
            {}

            /**
             * Captures ownership from a foreign pointer wrapper.
             * @param other The foreign pointer to capture ownership from.
             */
            MUSEQA_CUDA_INLINE unique_t(unique_t&& other) MUSEQA_SAFE_EXCEPT
            {
                capture(std::forward<decltype(other)>(other));
            }

            /**
             * Captures ownership from a foreign-typed pointer wrapper.
             * @tparam U The foreign pointer wrapper element type.
             * @param other The foreign-typed pointer to capture ownership from.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unique_t(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                capture(std::forward<decltype(other)>(other));
            }

            /**
             * Releases ownership of the currently owned pointer.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            MUSEQA_CUDA_INLINE ~unique_t() MUSEQA_SAFE_EXCEPT
            {
                memory::detail::release_ownership(m_ref);
            }

            /**
             * Releases ownership of the currently owned pointer and then captures
             * ownership from a foreign pointer wrapper.
             * @param other The foreign pointer to capture ownership from.
             * @return The current pointer wrapper instance.
             */
            MUSEQA_CUDA_INLINE unique_t& operator=(unique_t&& other) MUSEQA_SAFE_EXCEPT
            {
                capture(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases ownership of the currently owner pointer and then captures
             * ownership from a foreign-typed pointer wrapper.
             * @tparam U The foreign pointer wrapper element type.
             * @param other The foreign-typed pointer to capture ownership from.
             * @return The current unique pointer instance.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unique_t& operator=(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                capture(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases ownership of the currently owned pointer and returns wrapper
             * to the initial empty instantiation state.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            MUSEQA_CUDA_INLINE void reset() MUSEQA_SAFE_EXCEPT
            {
                auto empty = unique_t();
                swap(empty);
            }

            /**
             * Swaps ownership with another pointer instance.
             * @param other The instance to swap with.
             */
            MUSEQA_CUDA_INLINE void swap(unique_t& other) noexcept
            {
                super_t::swap(other);
                utility::swap(m_ref, other.m_ref);
            }

        private:
            /**
             * Instantiates a pointer wrapper from an raw pointer and reference counter.
             * @param ptr The raw pointer to be wrapped.
             * @param ref The reference counter to control the pointer's ownership.
             */
            MUSEQA_CUDA_INLINE explicit unique_t(T *ptr, detail::refcounter_t *ref)
              : super_t (ptr)
              , m_ref (ref)
            {}

            /**
             * Captures ownership from a foreign-typed pointer wrapper.
             * @tparam U The element type of the foreign pointer wrapper.
             * @param other The foreign pointer to capture ownership from.
             */
            template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
            MUSEQA_CUDA_INLINE void capture(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                if (this->m_ptr != other.m_ptr) {
                    this->reset();
                    this->swap(other);
                }
            }
    };

    /**
     * Allocates memory for the given element type into an unique pointer.
     * @tparam T The type to be allocated into a new unique pointer wrapper.
     * @tparam A The type of the memory allocator to use.
     * @param count The total number of elements to be allocated.
     * @param allocator The allocator to create the elements with.
     * @return The new unique pointer wrapper instance.
     */
    template <
        typename T = void
      , typename A = decltype(make_allocator<T>())
      , typename = std::enable_if_t<is_allocator<T, A>>>
    MUSEQA_CUDA_INLINE unique_t<T> make_unique(size_t count = 1, const A& allocator = make_allocator<T>())
    {
        auto ptr = (T*) allocator.allocate(count, sizeof(nonvoid_t<T>));
        return unique_t(ptr, allocator);
    }
}

MUSEQA_END_NAMESPACE

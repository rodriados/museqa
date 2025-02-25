/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A non-owning unmanaged pointer container implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/container.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Implements a generic non-owning pointer container into the common structure
     * of an automatically managed shared pointer. This object does not perform
     * any validation whether its raw pointer is valid or still allocated whatsoever.
     * Such precautions are up to be done by its user.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class unmanaged_t : public memory::pointer::shared_t<T>
    {
        public:
            typedef T element_t;
            typedef T *pointer_t;

        private:
            typedef memory::pointer::shared_t<T> underlying_t;

        public:
            MUSEQA_CONSTEXPR unmanaged_t() noexcept = default;
            MUSEQA_CONSTEXPR unmanaged_t(const unmanaged_t&) noexcept = default;
            MUSEQA_CONSTEXPR unmanaged_t(unmanaged_t&&) noexcept = default;

            /**
             * Builds a new unmanaged shared pointer instance from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            MUSEQA_CUDA_INLINE explicit unmanaged_t(T *ptr) noexcept
              : underlying_t (ptr, nullptr)
            {}

            /**
             * Acquires reference to a foreign-typed container's pointer.
             * @tparam U The foreign container's element type.
             * @param other The foreign container to reference to.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unmanaged_t(container_t<U>& other) noexcept
              : unmanaged_t (static_cast<T*>(other))
            {}

            MUSEQA_INLINE unmanaged_t& operator=(const unmanaged_t&) noexcept = default;
            MUSEQA_INLINE unmanaged_t& operator=(unmanaged_t&&) noexcept = default;

            /**
             * Discards the currently referenced pointer and acquires reference
             * to a foreign-typed container intance pointer.
             * @tparam U The foreign pointer type to be referenced.
             * @param other The foreign pointer instance to reference to.
             * @return This current unmanaged pointer container.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE unmanaged_t& operator=(container_t<U>& other) noexcept
            {
                return *new (this) unmanaged_t (static_cast<T*>(other));
            }

            /**
             * Creates an instance to an offset of the referenced pointer.
             * @param offset The offset to create a reference to.
             * @return The new offset reference pointer instance.
             */
            MUSEQA_CUDA_INLINE unmanaged_t offset(ptrdiff_t offset) noexcept
            {
                return unmanaged_t (this->m_ptr + offset);
            }

            /**
             * Swaps two unmanaged pointer container instances.
             * @param other The instance to swap with.
             */
            MUSEQA_CUDA_INLINE void swap(unmanaged_t& other) noexcept
            {
                utility::swap(this->m_ptr, other.m_ptr);
            }

        private:
            using underlying_t::swap;
    };

    /*
     * Deduction guides for a generic unmanaged pointer.
     * @since 1.0
     */
    template <typename T> unmanaged_t(T*) -> unmanaged_t<T>;
    template <typename T> unmanaged_t(const container_t<T>&) -> unmanaged_t<T>;
}

MUSEQA_END_NAMESPACE

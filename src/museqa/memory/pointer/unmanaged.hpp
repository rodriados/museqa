/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A non-owning unmanaged pointer wrapper implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2022-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/wrapper.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Implements a generic non-owning pointer wrapper into the common structure
     * of an automatically managed shared pointer. This object does not perform
     * any validation whether its raw pointer is valid or still allocated whatsoever.
     * Such precautions are up to be done by its user.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class unmanaged_t : public memory::pointer::shared_t<T>
    {
        private:
            typedef memory::pointer::shared_t<T> underlying_t;

        public:
            using typename underlying_t::element_t;
            using typename underlying_t::pointer_t;

        public:
            __host__ __device__ inline constexpr unmanaged_t() noexcept = default;
            __host__ __device__ inline constexpr unmanaged_t(const unmanaged_t&) noexcept = default;
            __host__ __device__ inline constexpr unmanaged_t(unmanaged_t&&) noexcept = default;

            /**
             * Builds a new unmanaged shared pointer instance from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            __host__ __device__ inline explicit unmanaged_t(pointer_t ptr) noexcept
              : underlying_t (ptr, nullptr)
            {}

            /**
             * The copy constructor from a foreign pointer wrapper type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             */
            template <typename U>
            __host__ __device__ inline unmanaged_t(const wrapper_t<U>& other) noexcept
              : unmanaged_t (static_cast<pointer_t>(other))
            {}

            __host__ __device__ inline unmanaged_t& operator=(const unmanaged_t&) noexcept = default;
            __host__ __device__ inline unmanaged_t& operator=(unmanaged_t&&) noexcept = default;

            /**
             * The copy-assignment operator from a foreign pointer wrapper type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             * @return This pointer instance.
             */
            template <typename U>
            __host__ __device__ inline unmanaged_t& operator=(const wrapper_t<U>& other) noexcept
            {
                return *new (this) unmanaged_t (static_cast<pointer_t>(other));
            }

            /**
             * Creates an instance to an offset of the wrapped pointer.
             * @param offset The requested offset.
             * @return The new offset pointer instance.
             */
            __host__ __device__ inline unmanaged_t offset(ptrdiff_t offset) noexcept
            {
                return unmanaged_t (this->m_ptr + offset);
            }

            /**
             * Swaps two unmanaged pointer wrapper instances.
             * @param other The instance to swap with.
             */
            __host__ __device__ inline void swap(unmanaged_t& other) noexcept
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
    template <typename T> unmanaged_t(const wrapper_t<T>&) -> unmanaged_t<T>;
}

MUSEQA_END_NAMESPACE

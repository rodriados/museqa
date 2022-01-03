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
    class unmanaged : public memory::pointer::shared<T>
    {
        private:
            typedef memory::pointer::shared<T> underlying_type;

        public:
            using typename underlying_type::element_type;
            using typename underlying_type::pointer_type;

        public:
            __host__ __device__ inline constexpr unmanaged() noexcept = default;
            __host__ __device__ inline constexpr unmanaged(const unmanaged&) noexcept = default;
            __host__ __device__ inline constexpr unmanaged(unmanaged&&) noexcept = default;

            /**
             * Builds a new unmanaged shared pointer instance from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            __host__ __device__ inline explicit unmanaged(pointer_type ptr) noexcept
              : underlying_type {ptr, nullptr}
            {}

            /**
             * The copy constructor from a foreign pointer wrapper type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             */
            template <typename U>
            __host__ __device__ inline unmanaged(const wrapper<U>& other) noexcept
              : unmanaged {static_cast<pointer_type>(other)}
            {}

            __host__ __device__ inline unmanaged& operator=(const unmanaged&) noexcept = default;
            __host__ __device__ inline unmanaged& operator=(unmanaged&&) noexcept = default;

            /**
             * The copy-assignment operator from a foreign pointer wrapper type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             * @return This pointer instance.
             */
            template <typename U>
            __host__ __device__ inline unmanaged& operator=(const wrapper<U>& other) noexcept
            {
                return *new (this) unmanaged {static_cast<pointer_type>(other)};
            }

            /**
             * Creates an instance to an offset of the wrapped pointer.
             * @param offset The requested offset.
             * @return The new offset pointer instance.
             */
            __host__ __device__ inline unmanaged offset(ptrdiff_t offset) noexcept
            {
                return unmanaged {this->m_ptr + offset};
            }

            /**
             * Swaps two unmanaged pointer wrapper instances.
             * @param other The instance to swap with.
             */
            __host__ __device__ inline void swap(unmanaged& other) noexcept
            {
                utility::swap(this->m_ptr, other.m_ptr);
            }

        private:
            using underlying_type::swap;
    };

    /*
     * Deduction guides for a generic unmanaged pointer.
     * @since 1.0
     */
    template <typename T> unmanaged(T*) -> unmanaged<T>;
    template <typename T> unmanaged(const wrapper<T>&) -> unmanaged<T>;
}

MUSEQA_END_NAMESPACE

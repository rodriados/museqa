/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A simple wrapper for pointer of a generic type implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/guard.hpp>

#include <museqa/memory/pointer/exception.hpp>

MUSEQA_DISABLE_NVCC_WARNING_BEGIN(815)

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Wraps a generic-typed pointer into an ownership context container.
     * @tparam T The target pointer's object type.
     * @since 1.0
     */
    template <typename T>
    class wrapper
    {
        static_assert(std::is_object<T>::value, "pointers can only point to object types");

        public:
            typedef T element_type;
            typedef element_type *pointer_type;

        private:
            typedef memory::pointer::exception exception_type;

        protected:
            pointer_type m_ptr = nullptr;

        public:
            __host__ __device__ inline constexpr wrapper() noexcept = default;
            __host__ __device__ inline constexpr wrapper(const wrapper&) noexcept = default;
            __host__ __device__ inline constexpr wrapper(wrapper&&) noexcept = default;

            /**
             * Instantiates a new wrapper from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            __host__ __device__ inline constexpr wrapper(pointer_type ptr) noexcept
              : m_ptr {ptr}
            {}

            __host__ __device__ inline constexpr wrapper& operator=(const wrapper&) noexcept = default;
            __host__ __device__ inline constexpr wrapper& operator=(wrapper&&) noexcept = default;

            /**
             * Dereferences the pointer and exposes the target object via a reference.
             * @return A reference to the target object.
             */
            __host__ __device__ inline constexpr element_type& operator*() __museqasafe__
            {
                return *dereference(0);
            }

            /**
             * Dereferences the pointer and exposes the target object via a const-reference.
             * @return A const-qualified reference to the target object.
             */
            __host__ __device__ inline constexpr const element_type& operator*() const __museqasafe__
            {
                return *dereference(0);
            }

            /**
             * Unwraps the internal pointer by a structure derefenrence operator.
             * @return The wrapped pointer.
             */
            __host__ __device__ inline constexpr pointer_type operator->() __museqasafe__
            {
                return dereference(0);
            }

            /**
             * Unwraps the internal pointer by a const-qualified dereference operator.
             * @return The const-qualified wrapped pointer.
             */
            __host__ __device__ inline constexpr const pointer_type operator->() const __museqasafe__
            {
                return dereference(0);
            }

            /**
             * Dereferences a pointer offset via the array-access operator.
             * @param offset The pointer offset to be dereferenced.
             * @return A reference to an offset target object.
             */
            __host__ __device__ inline constexpr element_type& operator[](ptrdiff_t offset) __museqasafe__
            {
                return *dereference(offset);
            }

            /**
             * Dereferences a pointer offset via the const array-access operator.
             * @param offset The pointer offset to be dereferenced.
             * @return A const-qualified reference to an offset target object.
             */
            __host__ __device__ inline constexpr const element_type& operator[](ptrdiff_t offset) const __museqasafe__
            {
                return *dereference(offset);
            }

            /**
             * Unwraps the internal pointer by an implicit conversion operator.
             * @return The wrapped pointer.
             */
            __host__ __device__ inline constexpr operator pointer_type() noexcept
            {
                return m_ptr;
            }

            /**
             * Unwraps the internal pointer by a const-qualified conversion operator.
             * @return The const-qualified wrapped pointer.
             */
            __host__ __device__ inline constexpr operator const pointer_type() const noexcept
            {
                return m_ptr;
            }

            /**
             * Converts the internal pointer into a foreign-typed pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The foreign-typed wrapped pointer.
             */
            template <typename U>
            __host__ __device__ inline constexpr explicit operator U*() noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Converts the internal pointer into a foreign const-qualified pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The const-qualified foreign-typed wrapped pointer.
             */
            template <typename U>
            __host__ __device__ inline constexpr explicit operator const U*() const noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Checks whether the internal pointer is dereferentiable or not.
             * @return Is the wrapped pointer dereferentiable?
             */
            __host__ __device__ inline constexpr operator bool() const noexcept
            {
                return (nullptr != m_ptr);
            }

            /**
             * Swaps the wrapped pointer with another pointer wrapper instance.
             * @param other The wrapper to swap pointers with.
             */
            __host__ __device__ inline constexpr void swap(wrapper& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Unwraps and exposes the internal pointer.
             * @return The wrapped pointer.
             */
            __host__ __device__ inline constexpr pointer_type unwrap() noexcept
            {
                return m_ptr;
            }

            /**
             * Unwraps and exposes the const-qualified internal pointer.
             * @return The const-qualified wrapped pointer.
             */
            __host__ __device__ inline constexpr const pointer_type unwrap() const noexcept
            {
                return m_ptr;
            }

        protected:
            /**
             * Retrieves a dereferentiable offset of the wrapped pointer.
             * @param offset The offset to be dereferenced by the pointer.
             * @return The deferentiable wrapped pointer offset.
             */
            __host__ __device__ inline constexpr pointer_type dereference(ptrdiff_t offset) const __museqasafe__
            {
                museqa::guard<exception_type>(m_ptr != nullptr, "wrapped pointer is not dereferentiable");
                return m_ptr + offset;
            }
    };

    /**
     * Wraps a type-erased pointer into an ownership context containter.
     * @since 1.0
     */
    template <>
    class wrapper<void>
    {
        public:
            typedef void element_type;
            typedef element_type *pointer_type;

        protected:
            pointer_type m_ptr = nullptr;

        public:
            __host__ __device__ inline constexpr wrapper() noexcept = default;
            __host__ __device__ inline constexpr wrapper(const wrapper&) noexcept = default;
            __host__ __device__ inline constexpr wrapper(wrapper&&) noexcept = default;

            /**
             * Instantiates a new wrapper from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            __host__ __device__ inline constexpr wrapper(pointer_type ptr) noexcept
              : m_ptr {ptr}
            {}

            __host__ __device__ inline constexpr wrapper& operator=(const wrapper&) noexcept = default;
            __host__ __device__ inline constexpr wrapper& operator=(wrapper&&) noexcept = default;

            /**
             * Unwraps the internal pointer by an implicit conversion operator.
             * @return The wrapped pointer.
             */
            __host__ __device__ inline constexpr operator pointer_type() noexcept
            {
                return m_ptr;
            }

            /**
             * Unwraps the internal pointer by a const-qualified conversion operator.
             * @return The const-qualified wrapped pointer.
             */
            __host__ __device__ inline constexpr operator const pointer_type() const noexcept
            {
                return m_ptr;
            }

            /**
             * Converts the internal pointer into a foreign-typed pointer.
             * @tparam T The foreign type to convert the pointer to.
             * @return The foreign-typed wrapped pointer.
             */
            template <typename T>
            __host__ __device__ inline constexpr explicit operator T*() noexcept
            {
                return static_cast<T*>(m_ptr);
            }

            /**
             * Converts the internal pointer into a foreign const-qualified pointer.
             * @tparam T The foreign type to convert the pointer to.
             * @return The const-qualified foreign-typed wrapped pointer.
             */
            template <typename T>
            __host__ __device__ inline constexpr explicit operator const T*() const noexcept
            {
                return static_cast<T*>(m_ptr);
            }

            /**
             * Checks whether the internal pointer is dereferentiable or not.
             * @return Is the wrapped pointer dereferentiable?
             */
            __host__ __device__ inline constexpr operator bool() const noexcept
            {
                return (nullptr != m_ptr);
            }

            /**
             * Swaps the wrapped pointer with another pointer wrapper instance.
             * @param other The wrapper to swap pointers with.
             */
            __host__ __device__ inline constexpr void swap(wrapper& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Unwraps and exposes the internal pointer.
             * @return The wrapped pointer.
             */
            __host__ __device__ inline constexpr pointer_type unwrap() noexcept
            {
                return m_ptr;
            }

            /**
             * Unwraps and exposes the const-qualified internal pointer.
             * @return The const-qualified wrapped pointer.
             */
            __host__ __device__ inline constexpr const pointer_type unwrap() const noexcept
            {
                return m_ptr;
            }
    };

    /**
     * Compares the memory addresses pointed by two pointer instances.
     * @tparam T The first pointer contents type.
     * @tparam U The second pointer contents type.
     * @param a The first pointer to be compared.
     * @param b The second pointer to be compared.
     * @return Do both pointers point to the same memory address?
     */
    template <typename T, typename U>
    __host__ __device__ inline constexpr bool operator==(
        const wrapper<T>& a
      , const wrapper<U>& b
    ) noexcept {
        return static_cast<void*>(a) == static_cast<void*>(b);
    }

    /**
     * Compares the memory addresses pointed by two pointer instances.
     * @tparam T The first pointer contents type.
     * @tparam U The second pointer contents type.
     * @param a The first pointer to be compared.
     * @param b The second pointer to be compared.
     * @return Do both pointers point to different memory addresses?
     */
    template <typename T, typename U>
    __host__ __device__ inline constexpr bool operator!=(
        const wrapper<T>& a
      , const wrapper<U>& b
    ) noexcept {
        return static_cast<void*>(a) != static_cast<void*>(b);
    }
}

MUSEQA_END_NAMESPACE

MUSEQA_DISABLE_NVCC_WARNING_END(815)

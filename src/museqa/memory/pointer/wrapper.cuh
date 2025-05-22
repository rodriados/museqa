/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A simple wrapper for pointers generic types implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.cuh>
#include <museqa/guard.cuh>

#include <museqa/memory/pointer/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * A transparent wrapper for generic-typed pointers with dereference checks.
     * @tparam T The wrapped pointer's object type.
     * @since 1.0
     */
    template <typename T>
    class wrapper_t
    {
        public:
            typedef T element_t;

        protected:
            T *m_ptr = nullptr;

        static_assert(std::is_object_v<T>, "pointers can only point to object types");

        public:
            MUSEQA_CONSTEXPR wrapper_t() noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t(const wrapper_t&) noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t(wrapper_t&&) noexcept = default;

            /**
             * Instantiates a new wrapper from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            MUSEQA_CUDA_CONSTEXPR wrapper_t(T *ptr) noexcept
              : m_ptr (ptr)
            {}

            MUSEQA_CONSTEXPR wrapper_t& operator=(const wrapper_t&) noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t& operator=(wrapper_t&&) noexcept = default;

            /**#@+
             * Dereferences the pointer with null-dereference checks.
             * @return A reference to the wrapped pointer object.
             */
            MUSEQA_CUDA_CONSTEXPR       T& operator*() MUSEQA_SAFE_EXCEPT       { return *get(); }
            MUSEQA_CUDA_CONSTEXPR const T& operator*() const MUSEQA_SAFE_EXCEPT { return *get(); }
            /**#@-*/

            /**#@+
             * Dereferences the pointer for method call with null-dereference checks.
             * @return The wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       T *operator->() MUSEQA_SAFE_EXCEPT       { return get(); }
            MUSEQA_CUDA_CONSTEXPR const T *operator->() const MUSEQA_SAFE_EXCEPT { return get(); }
            /**#@-*/

            /**#@+
             * Dereferences the pointer via an array-access index with checks.
             * @param i The wrapped pointer index to be dereferenced.
             * @return A reference to the wrapped pointer object at the given index.
             */
            MUSEQA_CUDA_CONSTEXPR       T& operator[](ptrdiff_t i)       { return *get(i); }
            MUSEQA_CUDA_CONSTEXPR const T& operator[](ptrdiff_t i) const { return *get(i); }
            /**#@-*/

            /**#@+
             * Unwraps the pointer via an implicit conversion operator.
             * @return The wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR operator       T*() noexcept       { return unwrap(); }
            MUSEQA_CUDA_CONSTEXPR operator const T*() const noexcept { return unwrap(); }
            /**#@-*/

            /**#@+
             * Checks if the wrapped pointer is not null.
             * @return Is the wrapped pointer not null?
             */
            MUSEQA_CUDA_CONSTEXPR explicit operator bool() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR explicit operator bool() const noexcept { return m_ptr; }
            /**#@-*/

            /**#@+
             * Explicitly unwraps the wrapped pointer.
             * @return The wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       T *unwrap() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR const T *unwrap() const noexcept { return m_ptr; }
            /**#@-*/

            /**
             * Implicitly converts the wrapped pointer into a foreign type.
             * @tparam U The foreign type to convert the wrapped pointer to.
             * @return The wrapped pointer cast into a foreign type.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator U*() noexcept
            {
                return static_cast<U*>(unwrap());
            }

            /**
             * Implicitly converts the pointer into a const-qualified foreign type.
             * @tparam U The foreign type to convert the wrapped pointer to.
             * @return The wrapped pointer cast into a const foreign type.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator const U*() const noexcept
            {
                return static_cast<const U*>(unwrap());
            }

            /**
             * Resets the wrapper by forgetting the previously wrapped pointer.
             * @see museqa::memory::pointer::wrapper_t::wrapper_t
             */
            MUSEQA_CUDA_CONSTEXPR void reset() noexcept
            {
                utility::exchange(m_ptr, nullptr);
            }

            /**
             * Swaps the wrapped pointer with one of another wrapper instance.
             * @param other The wrapper to swap wrapped pointers with.
             */
            MUSEQA_CUDA_CONSTEXPR void swap(wrapper_t& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Checks if the wrapper is empty and therefore non-dereferentiable.
             * @return Is the wrapper currently empty?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_ptr == nullptr;
            }

        protected:
            /**
             * Gets an offset of the wrapped pointer with null-dereference checks.
             * @param offset The offset to apply to the wrapped pointer.
             * @return The pointer to an offset of the wrapped pointer.
             */
            template <typename E = memory::pointer::exception_t>
            MUSEQA_CUDA_CONSTEXPR T *get(ptrdiff_t offset = 0) const MUSEQA_SAFE_EXCEPT
            {
                guard<E>(!empty(), "pointer is null and not dereferentiable");
                return m_ptr + offset;
            }
    };

    /**
     * A transparent wrapper for void, non-dereferentiable, pointers.
     * @since 1.0
     */
    template <>
    class wrapper_t<void>
    {
        public:
            typedef void element_t;

        protected:
            void *m_ptr = nullptr;

        public:
            MUSEQA_CONSTEXPR wrapper_t() noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t(const wrapper_t&) noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t(wrapper_t&&) noexcept = default;

            /**
             * Instantiates a new wrapper from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            MUSEQA_CUDA_CONSTEXPR wrapper_t(void *ptr) noexcept
              : m_ptr (ptr)
            {}

            MUSEQA_CONSTEXPR wrapper_t& operator=(const wrapper_t&) noexcept = default;
            MUSEQA_CONSTEXPR wrapper_t& operator=(wrapper_t&&) noexcept = default;

            /**#@+
             * Unwraps the pointer via an implicit conversion operator.
             * @return The wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR operator       void*() noexcept       { return unwrap(); }
            MUSEQA_CUDA_CONSTEXPR operator const void*() const noexcept { return unwrap(); }
            /**#@-*/

            /**#@+
             * Checks if the wrapped pointer is not null
             * @return Is the wrapped pointer not null?
             */
            MUSEQA_CUDA_CONSTEXPR explicit operator bool() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR explicit operator bool() const noexcept { return m_ptr; }
            /**#@-*/

            /**#@+
             * Explicitly unwraps the wrapped pointer.
             * @return The wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       void *unwrap() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR const void *unwrap() const noexcept { return m_ptr; }
            /**#@-*/

            /**
             * Implicitly converts the wrapped pointer into a foreign type.
             * @tparam U The foreign type to convert the wrapped pointer to.
             * @return The wrapped pointer cast into a foreign type.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator U*() noexcept
            {
                return static_cast<U*>(unwrap());
            }

            /**
             * Implicitly converts the pointer into a const-qualified foreign type.
             * @tparam U The foreign type to convert the wrapped pointer to.
             * @return The wrapped pointer cast into a const foreign type.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator const U*() const noexcept
            {
                return static_cast<const U*>(unwrap());
            }

            /**
             * Resets the wrapper by forgetting the previously wrapped pointer.
             * @see museqa::memory::pointer::wrapper_t::wrapper_t
             */
            MUSEQA_CUDA_CONSTEXPR void reset() noexcept
            {
                utility::exchange(m_ptr, nullptr);
            }

            /**
             * Swaps the wrapped pointer with one of another wrapper instance.
             * @param other The wrapper to swap wrapped pointers with.
             */
            MUSEQA_CUDA_CONSTEXPR void swap(wrapper_t& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Checks if the wrapper is empty and therefore non-dereferentiable.
             * @return Is the wrapper currently empty?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_ptr == nullptr;
            }
    };

    /**
     * Compares the memory addresses of two wrapped pointers.
     * @tparam T The first wrapper element type.
     * @tparam U The second wrapper element type.
     * @param a The first wrapper to be compared.
     * @param b The second wrapper to be compared.
     * @return Do both wrappers point to the same memory address?
     */
    template <typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator==(
        const wrapper_t<T>& a
      , const wrapper_t<U>& b
    ) noexcept {
        return static_cast<const void*>(a)
            == static_cast<const void*>(b);
    }

    /**
     * Compares the memory addresses of two wrapped pointers.
     * @tparam T The first wrapper element type.
     * @tparam U The second wrapper element type.
     * @param a The first wrapper to be compared.
     * @param b The second wrapper to be compared.
     * @return Do both wrappers point to different memory addresses?
     */
    template <typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator!=(
        const wrapper_t<T>& a
      , const wrapper_t<U>& b
    ) noexcept {
        return static_cast<const void*>(a)
            != static_cast<const void*>(b);
    }
}

MUSEQA_END_NAMESPACE

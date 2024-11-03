/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A simple container for pointer of a generic type implementation.
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

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Wraps a generic-typed pointer into an ownership context container.
     * @tparam T The target pointer's object type.
     * @since 1.0
     */
    template <typename T>
    class container_t
    {
        static_assert(std::is_object_v<T>, "pointers can only point to object types");

        public:
            typedef T element_t;

        private:
            typedef memory::pointer::exception_t exception_t;

        protected:
            element_t *m_ptr = nullptr;

        public:
            MUSEQA_CONSTEXPR container_t() noexcept = default;
            MUSEQA_CONSTEXPR container_t(const container_t&) noexcept = default;
            MUSEQA_CONSTEXPR container_t(container_t&&) noexcept = default;

            /**
             * Instantiates a new container from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            MUSEQA_CUDA_CONSTEXPR container_t(T *ptr) noexcept
              : m_ptr (ptr)
            {}

            MUSEQA_CONSTEXPR container_t& operator=(const container_t&) noexcept = default;
            MUSEQA_CONSTEXPR container_t& operator=(container_t&&) noexcept = default;

            /**#@+
             * Dereferences the pointer with null-dereference checks.
             * @return A reference to the pointed object.
             */
            MUSEQA_CUDA_CONSTEXPR       T& operator*() MUSEQA_SAFE_EXCEPT       { return *deref(0); }
            MUSEQA_CUDA_CONSTEXPR const T& operator*() const MUSEQA_SAFE_EXCEPT { return *deref(0); }
            /**#@-*/

            /**#@+
             * Dereferences the pointer for method call with null-dereference checks.
             * @return The underlying wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       T *operator->() MUSEQA_SAFE_EXCEPT       { return deref(0); }
            MUSEQA_CUDA_CONSTEXPR const T *operator->() const MUSEQA_SAFE_EXCEPT { return deref(0); }
            /**#@-*/

            /**#@+
             * Dereferences a pointer via an array-access index with checks.
             * @param i The pointer index to be dereferenced.
             * @return A reference to the object in the given index.
             */
            MUSEQA_CUDA_CONSTEXPR       T& operator[](ptrdiff_t i)       { return *deref(i); }
            MUSEQA_CUDA_CONSTEXPR const T& operator[](ptrdiff_t i) const { return *deref(i); }
            /**#@-*/

            /**#@+
             * Unwraps the pointer via an implicit conversion operator.
             * @return The internally wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR operator       T*() noexcept       { return unwrap(); }
            MUSEQA_CUDA_CONSTEXPR operator const T*() const noexcept { return unwrap(); }
            /**#@-*/

            /**#@+
             * Unwraps the container's internal pointer.
             * @return The unwrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       T *unwrap() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR const T *unwrap() const noexcept { return m_ptr; }
            /**#@-*/

            /**
             * Converts the pointer into a foreign-typed pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The foreign-typed unwrapped pointer.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator U*() noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Converts the pointer into a const-qualified foreign-typed pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The const-qualified foreign-typed unwrapped pointer.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator const U*() const noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Checks whether the internal pointer is dereferentiable or not.
             * @return Is the wrapped pointer dereferentiable?
             */
            MUSEQA_CUDA_CONSTEXPR operator bool() const noexcept
            {
                return !empty();
            }

            /**
             * Erases the wrapped pointer and set it back to the default value.
             * @see museqa::memory::pointer::container_t::container_t
             */
            MUSEQA_CUDA_CONSTEXPR void reset() noexcept
            {
                utility::exchange(m_ptr, nullptr);
            }

            /**
             * Swaps the wrapped pointer with another container instance.
             * @param other The container to swap pointers with.
             */
            MUSEQA_CUDA_CONSTEXPR void swap(container_t& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Checks whether the container is empty and therefore non-dereferentiable.
             * @return Is the container currently empty?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_ptr == nullptr;
            }

        protected:
            /**
             * Retrieves a dereferentiable offset of the wrapped pointer.
             * @param offset The offset to be dereferenced by the pointer.
             * @return The deferentiable wrapped pointer offset.
             */
            MUSEQA_CUDA_CONSTEXPR T *deref(ptrdiff_t offset) const MUSEQA_SAFE_EXCEPT
            {
                guard<exception_t>(!empty(), "null pointer is not dereferentiable");
                return m_ptr + offset;
            }
    };

    /**
     * Wraps a non-dereferentiable pointer into an ownership context containter.
     * @since 1.0
     */
    template <>
    class container_t<void>
    {
        public:
            typedef void element_t;
            typedef void *pointer_t;

        protected:
            element_t *m_ptr = nullptr;

        public:
            MUSEQA_CONSTEXPR container_t() noexcept = default;
            MUSEQA_CONSTEXPR container_t(const container_t&) noexcept = default;
            MUSEQA_CONSTEXPR container_t(container_t&&) noexcept = default;

            /**
             * Instantiates a new wrapper from a raw pointer.
             * @param ptr The pointer to be wrapped.
             */
            MUSEQA_CUDA_CONSTEXPR container_t(void *ptr) noexcept
              : m_ptr (ptr)
            {}

            MUSEQA_CONSTEXPR container_t& operator=(const container_t&) noexcept = default;
            MUSEQA_CONSTEXPR container_t& operator=(container_t&&) noexcept = default;

            /**#@+
             * Unwraps the pointer via an implicit conversion operator.
             * @return The internally wrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR operator       void*() noexcept       { return unwrap(); }
            MUSEQA_CUDA_CONSTEXPR operator const void*() const noexcept { return unwrap(); }
            /**#@-*/

            /**#@+
             * Unwraps the container's internal pointer.
             * @return The unwrapped pointer.
             */
            MUSEQA_CUDA_CONSTEXPR       void *unwrap() noexcept       { return m_ptr; }
            MUSEQA_CUDA_CONSTEXPR const void *unwrap() const noexcept { return m_ptr; }
            /**#@-*/

            /**
             * Converts the pointer into a foreign-typed pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The foreign-typed unwrapped pointer.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator U*() noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Converts the pointer into a const-qualified foreign-typed pointer.
             * @tparam U The foreign type to convert the pointer to.
             * @return The const-qualified foreign-typed unwrapped pointer.
             */
            template <typename U>
            MUSEQA_CUDA_CONSTEXPR operator const U*() const noexcept
            {
                return static_cast<U*>(m_ptr);
            }

            /**
             * Checks whether the internal pointer is dereferentiable or not.
             * @return Is the wrapped pointer dereferentiable?
             */
            MUSEQA_CUDA_CONSTEXPR operator bool() const noexcept
            {
                return !empty();
            }

            /**
             * Erases the wrapped pointer and set it back to the default value.
             * @see museqa::memory::pointer::container_t::container_t
             */
            MUSEQA_CUDA_CONSTEXPR void reset() noexcept
            {
                utility::exchange(m_ptr, nullptr);
            }

            /**
             * Swaps the wrapped pointer with another container instance.
             * @param other The container to swap pointers with.
             */
            MUSEQA_CUDA_CONSTEXPR void swap(container_t& other) noexcept
            {
                utility::swap(m_ptr, other.m_ptr);
            }

            /**
             * Checks whether the container is empty.
             * @return Is the container currently empty?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_ptr == nullptr;
            }
    };

    /**
     * Compares the memory addresses pointed by two container instances.
     * @tparam T The first container element type.
     * @tparam U The second container element type.
     * @param a The first container to be compared.
     * @param b The second container to be compared.
     * @return Do both containers point to the same memory address?
     */
    template <typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator==(
        const container_t<T>& a
      , const container_t<U>& b
    ) noexcept {
        return static_cast<void*>(a) == static_cast<void*>(b);
    }

    /**
     * Compares the memory addresses pointed by two container instances.
     * @tparam T The first container element type.
     * @tparam U The second container element type.
     * @param a The first container to be compared.
     * @param b The second container to be compared.
     * @return Do both containers point to different memory addresses?
     */
    template <typename T, typename U>
    MUSEQA_CUDA_CONSTEXPR bool operator!=(
        const container_t<T>& a
      , const container_t<U>& b
    ) noexcept {
        return static_cast<void*>(a) != static_cast<void*>(b);
    }
}

MUSEQA_END_NAMESPACE

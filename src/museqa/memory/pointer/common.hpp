/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The common interface for an automatic pointer implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>

#include <museqa/utility.hpp>
#include <museqa/memory/allocator.hpp>

namespace museqa
{
    namespace memory
    {
        namespace pointer
        {
            /**
             * Wraps a generic pointer into an owning container.
             * @param T The pointer's concrete type.
             * @since 1.0
             */
            template <typename T>
            class pointer
            {
                static_assert(!std::is_function<T>(), "cannot create pointer to a function");
                static_assert(!std::is_reference<T>(), "cannot create pointer to a reference");
                static_assert(!std::is_member_pointer<T>(), "cannot create pointer to a member");
                static_assert(!std::is_member_object_pointer<T>(), "cannot create pointer to an object member");
                static_assert(!std::is_member_function_pointer<T>(), "cannot create pointer to a function member");

              public:
                typedef pure<T> element_type;       /// The pointer's contents type.

              protected:
                element_type *m_ptr = nullptr;      /// The raw encapsulated pointer.

              protected:
                __host__ __device__ inline constexpr pointer() noexcept = default;
                __host__ __device__ inline constexpr pointer(const pointer&) noexcept = default;
                __host__ __device__ inline constexpr pointer(pointer&&) noexcept = default;

                /**
                 * Builds a new instance from a raw pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                __host__ __device__ inline pointer(element_type *ptr) noexcept
                  : m_ptr {ptr}
                {}

                __host__ __device__ inline pointer& operator=(const pointer&) noexcept = default;
                __host__ __device__ inline pointer& operator=(pointer&&) noexcept = default;

              public:
                /**
                 * The pointer dereferencing operator. This operator exposes the
                 * internally owned object via a reference.
                 * @return A reference to the owned object.
                 */
                __host__ __device__ inline element_type& operator*() noexcept
                {
                    return *m_ptr;
                }

                /**
                 * The const-qualified pointer dereferencing operator. This operator
                 * exposes the internally owned object via a reference.
                 * @return A const-qualified reference to the owned object.
                 */
                __host__ __device__ inline const element_type& operator*() const noexcept
                {
                    return *m_ptr;
                }

                /**
                 * The member access operator. This operator exposes the internally
                 * owned pointer for a member access.
                 * @return The owned pointer.
                 */
                __host__ __device__ inline element_type *operator->() noexcept
                {
                    return m_ptr;
                }

                /**
                 * The const-qualified member access operator. This operator exposes
                 * the internally owned pointer for a member access.
                 * @return The const-qualified owned pointer.
                 */
                __host__ __device__ inline const element_type *operator->() const noexcept
                {
                    return m_ptr;
                }

                /**
                 * Converts the pointer wrapper into the element type pointer.
                 * @return The internally owned pointer.
                 */
                __host__ __device__ inline operator element_type*() noexcept
                {
                    return m_ptr;
                }

                /**
                 * Converts the pointer wrapper into the element type pointer.
                 * @return The const-qualified internally owned pointer.
                 */
                __host__ __device__ inline operator const element_type*() const noexcept
                {
                    return m_ptr;
                }

                /**
                 * Converts the owned pointer into a pointer of a convertible type.
                 * @tparam U The target type to convert pointer to.
                 * @return The internally owned pointer.
                 */
                template <typename U>
                __host__ __device__ inline explicit operator U*() noexcept
                {
                    return static_cast<U*>(m_ptr);
                }

                /**
                 * Converts the owned pointer into a pointer of a convertible type.
                 * @tparam U The target type to convert pointer to.
                 * @return The const-qualified internally owned pointer.
                 */
                template <typename U>
                __host__ __device__ inline explicit operator const U*() const noexcept
                {
                    return static_cast<U*>(m_ptr);
                }

                /**
                 * Checks whether the stored pointer is null or not.
                 * @return Is the pointer not null?
                 */
                __host__ __device__ inline explicit operator bool() const noexcept
                {
                    return (nullptr != m_ptr);
                }

                /**
                 * Gives access to the raw owned pointer.
                 * @return The raw encapsulated pointer.
                 */
                __host__ __device__ inline element_type *raw() noexcept
                {
                    return m_ptr;
                }

                /**
                 * Gives access to the raw const-qualified owned pointer.
                 * @return The raw const-qualified encapsulated pointer.
                 */
                __host__ __device__ inline const element_type *raw() const noexcept
                {
                    return m_ptr;
                }
            };

            /**
             * Wraps a type-erased pointer into an owning container.
             * @since 1.0
             */
            template <>
            class pointer<void>
            {
              public:
                typedef void element_type;          /// The generic pointer element type.

              protected:
                void *m_ptr = nullptr;              /// The raw encapsulated pointer.

              protected:
                __host__ __device__ inline constexpr pointer() noexcept = default;
                __host__ __device__ inline constexpr pointer(const pointer&) noexcept = default;
                __host__ __device__ inline constexpr pointer(pointer&&) noexcept = default;

                /**
                 * Builds a new instance from a raw generic pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                __host__ __device__ inline pointer(void *ptr) noexcept
                  : m_ptr {ptr}
                {}

                __host__ __device__ inline pointer& operator=(const pointer&) noexcept = default;
                __host__ __device__ inline pointer& operator=(pointer&&) noexcept = default;

              public:
                /**
                 * Exposes the owned pointer as its original type.
                 * @return The internally owned pointer.
                 */
                __host__ __device__ inline operator void*() noexcept
                {
                    return m_ptr;
                }

                /**
                 * Exposes the owned pointer as its original const-qualified type.
                 * @return The internally owned const-qualified pointer.
                 */
                __host__ __device__ inline operator const void*() const noexcept
                {
                    return m_ptr;
                }

                /**
                 * Reinterprets the owned pointer to a dereferenceable pointer.
                 * @tparam T The type to reinterpret the pointer to.
                 * @return The reinterpreted owned pointer.
                 */
                template <typename T>
                __host__ __device__ inline explicit operator T*() noexcept
                {
                    return static_cast<T*>(m_ptr);
                }

                /**
                 * Reinterprets the owned pointer to a dereferenceable pointer.
                 * @tparam T The type to reinterpret the pointer to.
                 * @return The reinterpreted owned const-qualified pointer.
                 */
                template <typename T>
                __host__ __device__ inline explicit operator const T*() const noexcept
                {
                    return static_cast<T*>(m_ptr);
                }

                /**
                 * Checks whether the stored pointer is null or not.
                 * @return Is the pointer not null?
                 */
                __host__ __device__ inline explicit operator bool() const noexcept
                {
                    return (nullptr != m_ptr);
                }

                /**
                 * Gives access to the raw owned pointer.
                 * @return The raw encapsulated pointer.
                 */
                __host__ __device__ inline void *raw() noexcept
                {
                    return m_ptr;
                }

                /**
                 * Gives access to the raw owned const-qualified pointer.
                 * @return The raw encapsulated const-qualified pointer.
                 */
                __host__ __device__ inline const void *raw() const noexcept
                {
                    return m_ptr;
                }
            };

            /**
             * Wraps a generic array into an owning container.
             * @param T The array pointer's concrete type.
             * @since 1.0
             */
            template <typename T>
            class pointer<T[]> : public pointer<T>
            {
              private:
                typedef pointer<T> underlying_type;             /// The underlying pointer type.

              public:
                using typename underlying_type::element_type;   /// The array's contents type.

              protected:
                using underlying_type::pointer;
                using underlying_type::operator=;

              public:
                /**
                 * Gives access to an element in the array from its offset.
                 * @param offset The requested element offset to be retrieved.
                 * @return The requested array's element reference.
                 */
                __host__ __device__ inline element_type& operator[](ptrdiff_t offset) noexcept
                {
                    return *(this->m_ptr + offset);
                }

                /**
                 * Gives access to an element in the array from its offset.
                 * @param offset The requested element offset to be retrieved.
                 * @return The requested array's const-qualified element reference.
                 */
                __host__ __device__ inline const element_type& operator[](ptrdiff_t offset) const noexcept
                {
                    return *(this->m_ptr + offset);
                }
            };

            /**
             * Wraps a fixed-size array into an owning container.
             * @param T The array pointer's concrete type.
             * @param N The array's fixed size.
             * @since 1.0
             */
            template <typename T, size_t N>
            class pointer<T[N]> : public pointer<T[]>
            {
              private:
                typedef pointer<T[]> underlying_type;           /// The underlying pointer type.

              public:
                using typename underlying_type::element_type;   /// The array's contents type.

              protected:
                using underlying_type::pointer;
                using underlying_type::operator=;
            };
        }
    }
}

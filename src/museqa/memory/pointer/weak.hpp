/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements generic CUDA-compatible non-owning pointers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/utility.hpp>
#include <museqa/memory/pointer/common.hpp>
#include <museqa/memory/pointer/shared.hpp>
#include <museqa/memory/pointer/metadata.hpp>

namespace museqa
{
    namespace memory
    {
        namespace pointer
        {
            /**
             * Encapsulates a non-owning pointer into the common structure of a
             * shared pointer. This object does not perform any validation whether
             * its raw pointer is valid, still allocated whatsoever. Such precautions
             * are up to be done by its user.
             * @tparam T The type of pointer to be held.
             * @since 1.0
             */
            template <typename T>
            class weak : public memory::pointer::shared<T>
            {
              private:
                typedef memory::pointer::shared<T> underlying_type;
                typedef memory::pointer::metadata metadata_type;

              public:
                using typename underlying_type::element_type;

              public:
                __host__ __device__ inline constexpr weak() noexcept = default;
                __host__ __device__ inline constexpr weak(const weak&) noexcept = default;
                __host__ __device__ inline constexpr weak(weak&&) noexcept = default;

                /**
                 * Builds a new weak pointer instance from a raw pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                __host__ __device__ inline explicit weak(element_type *ptr) noexcept
                  : underlying_type {ptr, (metadata_type*) nullptr}
                {}

                /**
                 * Constructs a new weak pointer from a shared pointer by holding
                 * a non-owning reference of the shared pointer.
                 * @tparam U The foreign pointer type to hold to.
                 * @param other The shared pointer instance to hold to.
                 */
                template <typename U>
                __host__ __device__ inline weak(const shared<U>& other) noexcept
                  : weak {static_cast<element_type*>(other)}
                {}

                /**
                 * Constructs a new weak pointer by moving an foreign type instance.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 */
                template <typename U>
                __host__ __device__ inline weak(weak<U>&& other) noexcept
                  : weak {static_cast<element_type*>(other)}
                {}

                __host__ __device__ inline weak& operator=(const weak&) noexcept = default;
                __host__ __device__ inline weak& operator=(weak&&) noexcept = default;

                /**
                 * The copy-assignment operator from a foreign pointer type.
                 * @tparam U The foreign pointer type to be copied.
                 * @param other The foreign instance to be copied.
                 * @return This pointer object.
                 */
                template <typename U>
                __host__ __device__ inline weak& operator=(const shared<U>& other) noexcept
                {
                    return *new (this) weak {static_cast<element_type*>(other)};
                }

                /**
                 * The move-assignment operator from a foreign pointer type.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 * @return This pointer object.
                 */
                template <typename U>
                __host__ __device__ inline weak& operator=(weak<U>&& other) noexcept
                {
                    return *new (this) weak {static_cast<element_type*>(other)};
                }

                /**
                 * Swaps the weak pointer with another pointer instance.
                 * @param other The instance to swap with.
                 */
                __host__ __device__ inline void swap(weak& other) noexcept
                {
                    utility::swap(this->m_ptr, other.m_ptr);
                }

              private:
                using underlying_type::swap;
            };
        }
    }
}

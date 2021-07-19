/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements generic CUDA-compatible shared pointers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/common.hpp>

namespace museqa
{
    namespace memory
    {
        namespace pointer
        {
            /**
             * Implements a shared pointer. This object can be used to represent
             * a pointer that is managed and deleted automatically when all references
             * to it have been destroyed.
             * @tparam T The type of pointer to be held.
             * @since 1.0
             */
            template <typename T>
            class shared : public impl::pointer<T>
            {
                template <typename> friend class shared;

              private:
                typedef impl::pointer<T> underlying_type;
                typedef impl::metadata metadata_type;

              public:
                using typename underlying_type::element_type;
                using allocator_type = memory::allocator;

              private:
                metadata_type *m_meta = nullptr;                /// The pointer's metadata object instance.

              public:
                __host__ __device__ inline constexpr shared() noexcept = default;

                /**
                 * Builds a new instance from a raw pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                inline explicit shared(element_type *ptr) noexcept
                  : shared {factory::allocator<element_type>(), ptr}
                {}

                /**
                 * Builds a new instance from a raw pointer and its allocator.
                 * @param allocator The allocator of given pointer.
                 * @param ptr The pointer to be encapsulated.
                 */
                inline shared(const allocator_type& allocator, element_type *ptr) noexcept
                  : shared {ptr, metadata_type::acquire(ptr, allocator)}
                {}

                /**
                 * The shared pointer's copy constructor.
                 * @param other The instance to be copied.
                 */
                __host__ __device__ inline shared(const shared& other) noexcept
                  : shared {other.m_ptr, metadata_type::acquire(other.m_meta)}
                {}

                /**
                 * The copy constructor from a foreign pointer type.
                 * @tparam U The foreign pointer type to be copied.
                 * @param other The foreign instance to be copied.
                 */
                template <typename U>
                __host__ __device__ inline shared(const shared<U>& other) noexcept
                 : shared {static_cast<element_type*>(other.m_ptr), metadata_type::acquire(other.m_meta)}
                {}

                /**
                 * The shared pointer's move constructor.
                 * @param other The instance to be moved.
                 */
                __host__ __device__ inline shared(shared&& other) noexcept
                {
                    other.swap(*this);
                }

                /**
                 * The move constructor from a foreign pointer type.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 */
                template <typename U>
                __host__ __device__ inline shared(shared<U>&& other) noexcept
                  : shared {static_cast<element_type*>(other.m_ptr), metadata_type::acquire(other.m_meta)}
                {
                    other.reset();
                }

                /**
                 * Releases the ownership of the acquired pointer reference.
                 * @see shared::shared
                 */
                __host__ __device__ inline ~shared()
                {
                    metadata_type::release(m_meta);
                }

                /**
                 * The copy-assignment operator.
                 * @param other The instance to be copied.
                 * @return This pointer object.
                 */
                __host__ __device__ inline shared& operator=(const shared& other)
                {
                    metadata_type::release(m_meta);
                    return *new (this) shared {other};
                }

                /**
                 * The copy-assignment operator from a foreign pointer type.
                 * @tparam U The foreign pointer type to be copied.
                 * @param other The foreign instance to be copied.
                 * @return This pointer object.
                 */
                template <typename U>
                __host__ __device__ inline shared& operator=(const shared<U>& other)
                {
                    metadata_type::release(m_meta);
                    return *new (this) shared {other};
                }

                /**
                 * The move-assignment operator.
                 * @param other The instance to be moved.
                 * @return This pointer object.
                 */
                __host__ __device__ inline shared& operator=(shared&& other)
                {
                    metadata_type::release(m_meta);
                    return *new (this) shared {std::forward<decltype(other)>(other)};
                }

                /**
                 * The move-assignment operator from a foreign pointer type.
                 * @tparam U The foreign pointer type to be moved.
                 * @param other The foreign instance to be moved.
                 * @return This pointer object.
                 */
                template <typename U>
                __host__ __device__ inline shared& operator=(shared<U>&& other)
                {
                    metadata_type::release(m_meta);
                    return *new (this) shared {std::forward<decltype(other)>(other)};
                }

                /**
                 * Creates an instance to an offset of the wrapped pointer.
                 * @param offset The requested offset.
                 * @return The new offset pointer instance.
                 */
                __host__ __device__ inline shared offset(ptrdiff_t offset) noexcept
                {
                    return shared {this->m_ptr + offset, metadata_type::acquire(m_meta)};
                }

                /**
                 * Releases the pointer ownership and returns to an empty state.
                 * @see shared::shared
                 */
                __host__ __device__ inline void reset() noexcept
                {
                    metadata_type::release(m_meta);
                    new (this) shared {};
                }

                /**
                 * Swaps ownership with another pointer instance.
                 * @param other The instance to swap with.
                 */
                __host__ __device__ inline void swap(shared& other) noexcept
                {
                    utility::swap(this->m_ptr, other.m_ptr);
                    utility::swap(m_meta, other.m_meta);
                }

              protected:
                /**
                 * Builds a new instance from a raw pointer and its metadata.
                 * @param ptr The raw pointer object.
                 * @param meta The pointer's metadata.
                 */
                __host__ __device__ inline explicit shared(element_type *ptr, metadata_type *meta) noexcept
                  : underlying_type {ptr}
                  , m_meta {meta}
                {}
            };
        }
    }
}

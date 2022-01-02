/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An automatically managed pointer wrapper implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/wrapper.hpp>
#include <museqa/memory/pointer/detail/metadata.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory::pointer
{
    /**
     * Implements a generic pointer with an automatically managed lifetime duration.
     * The wrapped pointer will have its destructor automatically called when all
     * references to it have been destroyed.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class shared : public memory::pointer::wrapper<T>
    {
        template <typename> friend class shared;

        private:
            typedef memory::pointer::wrapper<T> underlying_type;
            typedef memory::pointer::detail::metadata metadata_type;
            typedef memory::allocator allocator_type;

        public:
            using typename underlying_type::element_type;
            using typename underlying_type::pointer_type;

        private:
            metadata_type *m_meta = nullptr;

        public:
            __host__ __device__ inline constexpr shared() noexcept = default;

            /**
             * Builds a new managed pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            inline explicit shared(pointer_type ptr, const allocator_type& allocator) noexcept
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
             * @param other The foreign pointer instance to be copied.
             */
            template <typename U>
            __host__ __device__ inline shared(const shared<U>& other) noexcept
              : shared {static_cast<pointer_type>(other.m_ptr), metadata_type::acquire(other.m_meta)}
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
             * @param other The foreign pointer instance to be moved.
             */
            template <typename U>
            __host__ __device__ inline shared(shared<U>&& other) noexcept
              : shared {static_cast<pointer_type>(other.m_ptr), metadata_type::acquire(other.m_meta)}
            {
                other.reset();
            }

            /**
             * Releases the ownership of the acquired pointer reference.
             * @see museqa::memory::pointer::shared::shared
             */
            __host__ __device__ inline ~shared() __devicesafe__
            {
                metadata_type::release(m_meta);
            }

            /**
             * The copy-assignment operator.
             * @param other The instance to be copied.
             * @return This pointer object.
             */
            __host__ __device__ inline shared& operator=(const shared& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) return *this;
                metadata_type::release(m_meta); return *new (this) shared {other};
            }

            /**
             * The copy-assignment operator from a foreign pointer type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline shared& operator=(const shared<U>& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) return *this;
                metadata_type::release(m_meta); return *new (this) shared {other};
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return This pointer object.
             */
            __host__ __device__ inline shared& operator=(shared&& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) { other.reset(); return *this; }
                metadata_type::release(m_meta); return *new (this) shared {std::forward<decltype(other)>(other)};
            }

            /**
             * The move-assignment operator from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline shared& operator=(shared<U>&& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) { other.reset(); return *this; }
                metadata_type::release(m_meta); return *new (this) shared {std::forward<decltype(other)>(other)};
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
             * @see museqa::memory::pointer::shared::shared
             */
            __host__ __device__ inline void reset() __devicesafe__
            {
                metadata_type::release(m_meta);
                new (this) shared {nullptr, nullptr};
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
             * Builds a new instance from a raw target pointer and a pointer metadata.
             * @param ptr The raw pointer object.
             * @param meta The pointer's metadata instance.
             */
            __host__ __device__ inline explicit shared(pointer_type ptr, metadata_type *meta) noexcept
              : underlying_type {ptr}
              , m_meta {meta}
            {}
    };

    /*
     * Deduction guides for a generic shared pointer.
     * @since 1.0
     */
    template <typename T> shared(T*) -> shared<T>;
    template <typename T> shared(T*, const memory::allocator&) -> shared<T>;
}

namespace factory::memory::pointer
{
    /**
     * Allocates memory for the given element type into a shared pointer.
     * @tparam T The type to be allocated into a new pointer.
     * @param count The total number of elements to be allocated.
     * @param allocator The allocator to create the elements with.
     * @return The allocated shared memory pointer.
     */
    template <typename T = void>
    inline auto shared(size_t count = 1, const museqa::memory::allocator& allocator = allocator<T>())
    -> typename std::enable_if<
        !std::is_reference<T>::value
      , museqa::memory::pointer::shared<T>
    >::type
    {
        return museqa::memory::pointer::shared<T>(allocator.allocate<T>(count), allocator);
    }
}

MUSEQA_END_NAMESPACE

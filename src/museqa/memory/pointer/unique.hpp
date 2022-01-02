/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An managed unique pointer wrapper implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

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
     * Implements a generic pointer with automatically managed lifetime duration
     * that cannot be owned by more than one unique instance at a time.
     * @tparam T The type of pointer to be wrapped.
     * @since 1.0
     */
    template <typename T>
    class unique : public memory::pointer::wrapper<T>
    {
        template <typename> friend class unique;

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
            __host__ __device__ inline constexpr unique() noexcept = default;
            __host__ __device__ inline constexpr unique(const unique&) noexcept = delete;

            /**
             * Builds a new unique pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            inline explicit unique(pointer_type ptr, const allocator_type& allocator) noexcept
              : unique {ptr, metadata_type::acquire(ptr, allocator)}
            {}

            /**
             * The unique pointer's move constructor.
             * @param other The instance to be moved.
             */
            __host__ __device__ inline unique(unique&& other) noexcept
            {
                other.swap(*this);
            }

            /**
             * The move constructor from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             */
            template <typename U>
            __host__ __device__ inline unique(unique<U>&& other) noexcept
              : unique {static_cast<pointer_type>(other.m_ptr), metadata_type::acquire(other.m_meta)}
            {
                other.reset();
            }

            /**
             * Releases the ownership of the acquired pointer reference.
             * @see museqa::memory::pointer::unique::unique
             */
            __host__ __device__ inline ~unique() __devicesafe__
            {
                metadata_type::release(m_meta);
            }

            __host__ __device__ inline unique& operator=(const unique&) noexcept = delete;

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return This pointer object.
             */
            __host__ __device__ inline unique& operator=(unique&& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) { other.reset(); return *this; }
                metadata_type::release(m_meta); return *new (this) unique {std::forward<decltype(other)>(other)};
            }

            /**
             * The move-assignment operator from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline unique& operator=(unique<U>&& other) __devicesafe__
            {
                if (other.m_ptr == this->m_ptr) { other.reset(); return *this; }
                metadata_type::release(m_meta); return *new (this) unique {std::forward<decltype(other)>(other)};
            }

            /**
             * Releases the pointer ownership and returns to an empty state.
             * @see museqa::memory::pointer::unique::unique
             */
            __host__ __device__ inline void reset() __devicesafe__
            {
                metadata_type::release(m_meta);
                new (this) unique {nullptr, nullptr};
            }

            /**
             * Swaps ownership with another pointer instance.
             * @param other The instance to swap with.
             */
            __host__ __device__ inline void swap(unique& other) noexcept
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
            __host__ __device__ inline explicit unique(pointer_type ptr, metadata_type *meta) noexcept
              : underlying_type {ptr}
              , m_meta {meta}
            {}
    };

    /*
     * Deduction guides for a generic unique pointer.
     * @since 1.0
     */
    template <typename T> unique(T*) -> unique<T>;
    template <typename T> unique(T*, const memory::allocator&) -> unique<T>;
}

namespace factory::memory::pointer
{
    /**
     * Allocates memory for the given element type into an unique pointer.
     * @tparam T The type to be allocated into a new pointer.
     * @param count The total number of elements to be allocated.
     * @param allocator The allocator to create the elements with.
     * @return The allocated unique memory pointer.
     */
    template <typename T = void>
    inline auto unique(size_t count = 1, const museqa::memory::allocator& allocator = allocator<T>())
    -> typename std::enable_if<
        !std::is_reference<T>::value
      , museqa::memory::pointer::unique<T>
    >::type
    {
        return museqa::memory::pointer::unique<T>(allocator.allocate<T>(count), allocator);
    }
}

MUSEQA_END_NAMESPACE

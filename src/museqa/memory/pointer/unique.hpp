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
    class unique_t : public memory::pointer::wrapper_t<T>
    {
        template <typename> friend class unique_t;

        private:
            typedef memory::pointer::wrapper_t<T> underlying_t;
            typedef memory::pointer::detail::metadata_t metadata_t;
            typedef memory::allocator_t allocator_t;

        public:
            using typename underlying_t::element_t;
            using typename underlying_t::pointer_t;

        private:
            metadata_t *m_meta = nullptr;

        public:
            __host__ __device__ inline constexpr unique_t() noexcept = default;
            __host__ __device__ inline constexpr unique_t(const unique_t&) noexcept = delete;

            /**
             * Builds a new unique pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            inline explicit unique_t(pointer_t ptr, const allocator_t& allocator) noexcept
              : unique_t (ptr, metadata_t::acquire(ptr, allocator))
            {}

            /**
             * The unique pointer's move constructor.
             * @param other The instance to be moved.
             */
            __host__ __device__ inline unique_t(unique_t&& other) noexcept
            {
                transfer(std::forward<decltype(other)>(other));
            }

            /**
             * The move constructor from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             */
            template <typename U>
            __host__ __device__ inline unique_t(unique_t<U>&& other) noexcept
            {
                transfer(std::forward<decltype(other)>(other));
            }

            /**
             * Releases the ownership of the acquired pointer reference.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            __host__ __device__ inline ~unique_t() __devicesafe__
            {
                metadata_t::release(m_meta);
            }

            __host__ __device__ inline unique_t& operator=(const unique_t&) noexcept = delete;

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return This pointer object.
             */
            __host__ __device__ inline unique_t& operator=(unique_t&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * The move-assignment operator from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline unique_t& operator=(unique_t<U>&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases the pointer ownership and returns to an empty state.
             * @see museqa::memory::pointer::unique_t::unique_t
             */
            __host__ __device__ inline void reset() __devicesafe__
            {
                metadata_t::release(utility::exchange(m_meta, nullptr));
                underlying_t::reset();
            }

            /**
             * Swaps ownership with another pointer instance.
             * @param other The instance to swap with.
             */
            __host__ __device__ inline void swap(unique_t& other) noexcept
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
            __host__ __device__ inline explicit unique_t(pointer_t ptr, metadata_t *meta) noexcept
              : underlying_t (ptr)
              , m_meta (meta)
            {}

        private:
            template <typename U> __host__ __device__ inline void transfer(unique_t<U>&&) __devicesafe__;
    };

    /**
     * Captures the ownership from a pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be moved.
     * @param other The foreign pointer instance to be moved.
     */
    template <typename T> template <typename U>
    __host__ __device__ inline void unique_t<T>::transfer(unique_t<U>&& other) __devicesafe__
    {
        static_assert(std::is_convertible<U*, T*>::value, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            metadata_t::release(m_meta);
            this->m_ptr  = utility::exchange(other.m_ptr, nullptr);
            this->m_meta = utility::exchange(other.m_meta, nullptr);
        } else {
            other.reset();
        }
    }

    /*
     * Deduction guides for a generic unique pointer.
     * @since 1.0
     */
    template <typename T> unique_t(T*) -> unique_t<T>;
    template <typename T> unique_t(T*, const memory::allocator_t&) -> unique_t<T>;
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
    inline auto unique(size_t count = 1, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> typename std::enable_if<
        !std::is_reference<T>::value
      , museqa::memory::pointer::unique_t<T>
    >::type
    {
        return museqa::memory::pointer::unique_t<T>(allocator.allocate<T>(count), allocator);
    }
}

MUSEQA_END_NAMESPACE

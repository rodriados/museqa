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
    class shared_t : public memory::pointer::wrapper_t<T>
    {
        template <typename> friend class shared_t;

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
            __host__ __device__ inline constexpr shared_t() noexcept = default;

            /**
             * Builds a new managed pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            inline explicit shared_t(pointer_t ptr, const allocator_t& allocator) noexcept
              : shared_t (ptr, metadata_t::acquire(ptr, allocator))
            {}

            /**
             * The shared pointer's copy constructor.
             * @param other The instance to be copied.
             */
            __host__ __device__ inline shared_t(const shared_t& other) __devicesafe__
            {
                share(other);
            }

            /**
             * The copy constructor from a foreign pointer type.
             * @tparam U The foreign pointer type to be copied.
             * @param other The foreign pointer instance to be copied.
             */
            template <typename U>
            __host__ __device__ inline shared_t(const shared_t<U>& other) __devicesafe__
            {
                share(other);
            }

            /**
             * The shared pointer's move constructor.
             * @param other The instance to be moved.
             */
            __host__ __device__ inline shared_t(shared_t&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other));
            }

            /**
             * The move constructor from a foreign pointer type.
             * @tparam U The foreign pointer type to be moved.
             * @param other The foreign pointer instance to be moved.
             */
            template <typename U>
            __host__ __device__ inline shared_t(shared_t<U>&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other));
            }

            /**
             * Releases the ownership of the acquired pointer reference.
             * @see museqa::memory::pointer::shared_t::shared_t
             */
            __host__ __device__ inline ~shared_t() __devicesafe__
            {
                metadata_t::release(m_meta);
            }

            /**
             * The copy-assignment operator.
             * @param other The instance to be shared.
             * @return This pointer object.
             */
            __host__ __device__ inline shared_t& operator=(const shared_t& other) __devicesafe__
            {
                share(other); return *this;
            }

            /**
             * The copy-assignment operator from a foreign pointer type.
             * @tparam U The foreign pointer type to be shared.
             * @param other The foreign pointer instance to be shared.
             * @return This pointer object.
             */
            template <typename U>
            __host__ __device__ inline shared_t& operator=(const shared_t<U>& other) __devicesafe__
            {
                share(other); return *this;
            }

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return This pointer object.
             */
            __host__ __device__ inline shared_t& operator=(shared_t&& other) __devicesafe__
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
            __host__ __device__ inline shared_t& operator=(shared_t<U>&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Creates an instance to an offset of the wrapped pointer.
             * @param offset The requested offset.
             * @return The new offset pointer instance.
             */
            __host__ __device__ inline shared_t offset(ptrdiff_t offset) noexcept
            {
                return shared_t (this->m_ptr + offset, metadata_t::acquire(m_meta));
            }

            /**
             * Releases the pointer ownership and returns to an empty state.
             * @see museqa::memory::pointer::shared_t::shared_t
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
            __host__ __device__ inline void swap(shared_t& other) noexcept
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
            __host__ __device__ inline explicit shared_t(pointer_t ptr, metadata_t *meta) noexcept
              : underlying_t (ptr)
              , m_meta (meta)
            {}

        private:
            template <typename U> __host__ __device__ inline void share(const shared_t<U>&) __devicesafe__;
            template <typename U> __host__ __device__ inline void transfer(shared_t<U>&&) __devicesafe__;
    };

    /**
     * Acquires shared ownership from a pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be shared.
     * @param other The foreign pointer instance to be shared.
     */
    template <typename T> template <typename U>
    __host__ __device__ inline void shared_t<T>::share(const shared_t<U>& other) __devicesafe__
    {
        static_assert(std::is_convertible<U*, T*>::value, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            metadata_t::release(m_meta);
            this->m_ptr  = other.m_ptr;
            this->m_meta = metadata_t::acquire(other.m_meta);
        }
    }

    /**
     * Transfers ownership from a pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be moved.
     * @param other The foreign pointer instance to be moved.
     */
    template <typename T> template <typename U>
    __host__ __device__ inline void shared_t<T>::transfer(shared_t<U>&& other) __devicesafe__
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
     * Deduction guides for a generic shared pointer.
     * @since 1.0
     */
    template <typename T> shared_t(T*) -> shared_t<T>;
    template <typename T> shared_t(T*, const memory::allocator_t&) -> shared_t<T>;
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
    inline auto shared(size_t count = 1, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> typename std::enable_if<
        !std::is_reference<T>::value
      , museqa::memory::pointer::shared_t<T>
    >::type
    {
        return museqa::memory::pointer::shared_t<T>(allocator.allocate<T>(count), allocator);
    }
}

MUSEQA_END_NAMESPACE

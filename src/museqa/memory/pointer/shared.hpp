/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An automatically managed pointer container implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>

#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/container.hpp>
#include <museqa/memory/pointer/unique.hpp>
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
    class shared_t : public memory::pointer::container_t<T>
    {
        template <typename> friend class shared_t;

        public:
            typedef T element_t;
            typedef T *pointer_t;

        private:
            typedef memory::pointer::container_t<T> underlying_t;
            typedef memory::pointer::detail::metadata_t metadata_t;
            typedef memory::allocator_t allocator_t;

        private:
            metadata_t *m_meta = nullptr;

        public:
            MUSEQA_CONSTEXPR shared_t() noexcept = default;

            /**
             * Builds a new managed pointer from a raw pointer and its allocator.
             * @param ptr The raw pointer to be wrapped.
             * @param allocator The given pointer's allocator.
             */
            MUSEQA_INLINE explicit shared_t(T *ptr, const allocator_t& allocator) noexcept
              : shared_t (ptr, metadata_t::acquire(ptr, allocator))
            {}

            /**
             * Share ownership of another container's pointer.
             * @param other The container to share ownership with.
             */
            MUSEQA_CUDA_INLINE shared_t(const shared_t& other) MUSEQA_SAFE_EXCEPT
            {
                share(other);
            }

            /**
             * Share ownership of a foreign-typed container's pointer.
             * @tparam U The foreign container's element type.
             * @param other The foreign-typed container to share ownership with.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t(const shared_t<U>& other) MUSEQA_SAFE_EXCEPT
            {
                share(other);
            }

            /**
             * Acquire ownership of another container's pointer.
             * @param other The container to acquire ownership from.
             */
            MUSEQA_CUDA_INLINE shared_t(shared_t&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other));
            }

            /**
             * Acquire ownership of a foreign-typed container's pointer.
             * @tparam U The foreign container's element type.
             * @param other The foreign-typed container to acquire ownership from.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t(shared_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other));
            }

            /**
             * Acquire ownership of a foreign-typed container's unique pointer.
             * @tparam U The foreign container's element type.
             * @param other The foreign-typed container to acquire ownership from.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other));
            }

            /**
             * Releases ownership of the acquired pointer.
             * @see museqa::memory::pointer::shared_t::shared_t
             */
            MUSEQA_CUDA_INLINE ~shared_t() MUSEQA_SAFE_EXCEPT
            {
                metadata_t::release(m_meta);
            }

            /**
             * Releases the currently owned pointer and acquires shared ownership
             * of another container's instance pointer.
             * @param other The container to be share ownership with.
             * @return The current shared pointer container.
             */
            MUSEQA_CUDA_INLINE shared_t& operator=(const shared_t& other) MUSEQA_SAFE_EXCEPT
            {
                share(other); return *this;
            }

            /**
             * Releases the currently owned pointer and acquires shared ownership
             * of a foreign-typed container's instance pointer.
             * @tparam U The foreign pointer type to be shared.
             * @param other The foreign container to share ownership with.
             * @return The current shared pointer container.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t& operator=(const shared_t<U>& other) MUSEQA_SAFE_EXCEPT
            {
                share(other); return *this;
            }

            /**
             * Releases the currently owned pointer and acquires exclusive ownership
             * of another container's instance pointer.
             * @param other The container acquire ownership from.
             * @return The current shared pointer container.
             */
            MUSEQA_CUDA_INLINE shared_t& operator=(shared_t&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases the currently owned pointer and acquires exclusive ownership
             * of a foreign-typed container's instance pointer.
             * @tparam U The foreign pointer type to acquire.
             * @param other The foreign pointer to acquire ownership from.
             * @return The current shared pointer container.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t& operator=(shared_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Releases the currently owned pointer and acquires exclusive ownership
             * of a foreign-typed container's unique instance pointer.
             * @tparam U The foreign pointer type to acquire.
             * @param other The foreign pointer to acquire ownership from.
             * @return The current shared pointer container.
             */
            template <typename U>
            MUSEQA_CUDA_INLINE shared_t& operator=(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
            {
                acquire(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Creates an instance to an offset of the owned pointer.
             * @param offset The offset to create a share pointer of.
             * @return The new offset shared pointer instance.
             */
            MUSEQA_CUDA_INLINE shared_t offset(ptrdiff_t offset) noexcept
            {
                return shared_t (this->m_ptr + offset, metadata_t::acquire(m_meta));
            }

            /**
             * Releases the pointer ownership and returns to an empty state.
             * @see museqa::memory::pointer::shared_t::shared_t
             */
            MUSEQA_CUDA_INLINE void reset() MUSEQA_SAFE_EXCEPT
            {
                auto ephemeral = shared_t();
                swap(ephemeral);
            }

            /**
             * Swaps ownership with another pointer instance.
             * @param other The instance to swap with.
             */
            MUSEQA_CUDA_INLINE void swap(shared_t& other) noexcept
            {
                underlying_t::swap(other);
                utility::swap(m_meta, other.m_meta);
            }

        protected:
            /**
             * Builds a new instance from a raw target pointer and a pointer metadata.
             * @param ptr The raw pointer object.
             * @param meta The pointer's metadata instance.
             */
            MUSEQA_CUDA_INLINE explicit shared_t(T *ptr, metadata_t *meta) noexcept
              : underlying_t (ptr)
              , m_meta (meta)
            {}

        private:
            template <typename U> MUSEQA_CUDA_INLINE void share(const shared_t<U>&) MUSEQA_SAFE_EXCEPT;
            template <typename U> MUSEQA_CUDA_INLINE void acquire(shared_t<U>&&) MUSEQA_SAFE_EXCEPT;
            template <typename U> MUSEQA_CUDA_INLINE void acquire(unique_t<U>&&) MUSEQA_SAFE_EXCEPT;
    };

    /**
     * Acquires shared ownership from a pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be shared.
     * @param other The foreign pointer instance to be shared.
     */
    template <typename T> template <typename U>
    MUSEQA_CUDA_INLINE void shared_t<T>::share(const shared_t<U>& other) MUSEQA_SAFE_EXCEPT
    {
        static_assert(std::is_convertible_v<U*, T*>, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            metadata_t::release(m_meta);
            this->m_ptr  = other.m_ptr;
            this->m_meta = metadata_t::acquire(other.m_meta);
        }
    }

    /**
     * Captures ownership from a shared pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be moved.
     * @param other The foreign pointer instance to be moved.
     */
    template <typename T> template <typename U>
    MUSEQA_CUDA_INLINE void shared_t<T>::acquire(shared_t<U>&& other) MUSEQA_SAFE_EXCEPT
    {
        static_assert(std::is_convertible_v<U*, T*>, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            metadata_t::release(m_meta);
            this->m_ptr  = utility::exchange(other.m_ptr, nullptr);
            this->m_meta = utility::exchange(other.m_meta, nullptr);
        } else {
            other.reset();
        }
    }

    /**
     * Captures ownership from a unique pointer instance of generic type.
     * @tparam T The type of the target wrapped pointer.
     * @tparam U The foreign pointer type to be moved.
     * @param other The foreign pointer instance to be moved.
     */
    template <typename T> template <typename U>
    MUSEQA_CUDA_INLINE void shared_t<T>::acquire(unique_t<U>&& other) MUSEQA_SAFE_EXCEPT
    {
        static_assert(std::is_convertible_v<U*, T*>, "pointer types are not convertible");

        if (this->m_ptr != other.m_ptr) {
            metadata_t::release(m_meta);
            this->m_ptr = utility::exchange(other.m_ptr, nullptr);
            auto deleter = utility::exchange(other.m_deleter, nullptr);
            this->m_meta = metadata_t::acquire(this->m_ptr, deleter);
        }
    }

    /*
     * Deduction guides for a generic shared pointer.
     * @since 1.0
     */
    template <typename T> shared_t(T*) -> shared_t<T>;
    template <typename T> shared_t(T*, const memory::allocator_t&) -> shared_t<T>;
    template <typename T> shared_t(unique_t<T>&&) -> shared_t<T>;
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
    template <
        typename T = void
      , typename = std::enable_if_t<!std::is_reference_v<T>>>
    MUSEQA_INLINE museqa::memory::pointer::shared_t<T> shared(
        size_t count = 1
      , const museqa::memory::allocator_t& allocator = factory::memory::allocator<T>()
    ) {
        T *ptr = allocator.allocate<T>(count);
        return museqa::memory::pointer::shared_t(ptr, allocator);
    }
}

MUSEQA_END_NAMESPACE

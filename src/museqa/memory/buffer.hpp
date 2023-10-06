/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file An automatically managed memory buffer implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <cstddef>
#include <utility>

#include <museqa/environment.h>
#include <museqa/utility.hpp>
#include <museqa/guard.hpp>

#include <museqa/memory/pointer.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/exception.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Implements a generic automatically-managed contiguous memory buffer specialized
     * for storage of a defined number of elements of the given type.
     * @tparam T The buffer's elements' type.
     * @see std::string
     * @see std::vector
     * @since 1.0
     */
    template <typename T>
    class buffer_t : protected memory::pointer::shared_t<T>
    {
        static_assert(!std::is_array<T>::value, "buffer element cannot be an array");
        static_assert(!std::is_reference<T>::value, "buffer element cannot have reference type");

        private:
            typedef memory::pointer::shared_t<T> underlying_t;
            typedef memory::exception_t exception_t;

        public:
            using typename underlying_t::element_t;
            using typename underlying_t::pointer_t;

        protected:
            size_t m_capacity = 0;

        public:
            __host__ __device__ inline buffer_t() noexcept = default;
            __host__ __device__ inline buffer_t(const buffer_t&) __devicesafe__ = default;

            /**
             * Builds a buffer by acquiring a shared ownership of a buffer pointer.
             * @param ptr The pointer instance to acquire shared ownership of.
             * @param capacity The acquired pointer's buffer capacity.
             */
            __host__ __device__ inline explicit buffer_t(const underlying_t& ptr, size_t capacity) __devicesafe__
              : underlying_t (ptr)
              , m_capacity (capacity)
            {}

            /**
             * Builds a buffer by moving the ownership of a buffer pointer.
             * @param ptr The pointer instance to acquire ownership of.
             * @param capacity The acquired pointer's buffer capacity.
             */
            __host__ __device__ inline explicit buffer_t(underlying_t&& ptr, size_t capacity) __devicesafe__
              : underlying_t (std::forward<decltype(ptr)>(ptr))
              , m_capacity (capacity)
            {}

            /**
             * The buffer's move constructor.
             * @param other The instance to be moved.
             */
            __host__ __device__ inline buffer_t(buffer_t&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other));
            }

            __host__ __device__ inline buffer_t& operator=(const buffer_t&) __devicesafe__ = default;

            /**
             * The move-assignment operator.
             * @param other The instance to be moved.
             * @return This buffer object.
             */
            __host__ __device__ inline buffer_t& operator=(buffer_t&& other) __devicesafe__
            {
                transfer(std::forward<decltype(other)>(other)); return *this;
            }

            /**
             * Accesses an element in the buffer via its offset.
             * @param offset The requested buffer offset to be accessed.
             * @return The element at the requested buffer offset.
             */
            __host__ __device__ inline element_t& operator[](ptrdiff_t offset) __museqasafe__
            {
                return *dereference(offset);
            }

            /**
             * Accesses a const-qualified element in the buffer via its offset.
             * @param offset The requested buffer offset to be accessed.
             * @return The const-qualified element at the requested buffer offset.
             */
            __host__ __device__ inline const element_t& operator[](ptrdiff_t offset) const __museqasafe__
            {
                return *dereference(offset);
            }

            /**
             * Retrieves a slice of the buffer and shares it with a new instance.
             * @param offset The slice offset in relation to original buffer.
             * @param count The number of elements to slice from buffer.
             */
            __host__ __device__ inline buffer_t slice(ptrdiff_t offset, size_t count = 0) __museqasafe__
            {
                museqa::guard<exception_t>(offset >= 0, "buffer slice index offset is invalid");
                museqa::guard<exception_t>((size_t) offset + count <= m_capacity, "buffer slice out of range");
                return buffer_t (this->offset(offset), count ? count : m_capacity - (size_t) offset);
            }

            /**
             * The buffer's initial iterator position.
             * @return The pointer to the start of the buffer iterator.
             */
            __host__ __device__ inline pointer_t begin() noexcept
            {
                return underlying_t::unwrap();
            }

            /**
             * The buffer's initial const-qualified iterator position.
             * @return The pointer to the start of the buffer const-qualified iterator.
             */
            __host__ __device__ inline const pointer_t begin() const noexcept
            {
                return underlying_t::unwrap();
            }

            /**
             * The buffer's final iterator position.
             * @return The pointer to the end of the buffer iterator.
             */
            __host__ __device__ inline pointer_t end() noexcept
            {
                return m_capacity + underlying_t::unwrap();
            }

            /**
             * The buffer's final const-qualified iterator position.
             * @return The pointer to the end of the buffer const-qualified iterator.
             */
            __host__ __device__ inline const pointer_t end() const noexcept
            {
                return m_capacity + underlying_t::unwrap();
            }

            /**
             * Unwraps and exposes the buffer's underlying pointer.
             * @return The buffer's internal underlying pointer.
             */
            __host__ __device__ inline underlying_t& unwrap() noexcept
            {
                return static_cast<underlying_t&>(*this);
            }

            /**
             * Unwraps and exposes the buffer's underlying const-qualified pointer.
             * @return The buffer's internal underlying const-qualified pointer.
             */
            __host__ __device__ inline const underlying_t& unwrap() const noexcept
            {
                return static_cast<const underlying_t&>(*this);
            }

            /**
             * Informs the buffer's current total capacity.
             * @return The maximum capacity of elements in buffer.
             */
            __host__ __device__ inline size_t capacity() const noexcept
            {
                return m_capacity;
            }

            /**
             * Informs whether the buffer is empty and therefore has zero capacity.
             * @return Is the buffer empty?
             */
            __host__ __device__ inline bool empty() const noexcept
            {
                return m_capacity == 0;
            }

            /**
             * Releases the buffer ownership and returns to an empty state.
             * @see museqa::memory::buffer_t::buffer_t
             */
            __host__ __device__ inline void reset() __devicesafe__
            {
                utility::exchange(m_capacity, 0);
                underlying_t::reset();
            }

        protected:
            /**
             * Retrieves a dereferentiable offset of the underlying pointer.
             * @param offset The offset to be dereferenced by the pointer.
             * @return The deferentiable wrapped pointer offset.
             */
            __host__ __device__ inline constexpr pointer_t dereference(ptrdiff_t offset) const __museqasafe__
            {
                museqa::guard<exception_t>(offset >= 0, "cannot access buffer negative offset");
                museqa::guard<exception_t>((size_t) offset < m_capacity, "buffer offset is out of range");
                return underlying_t::dereference(offset);
            }

        private:
            /**
             * Acquires ownership of a buffer instance.
             * @param other The buffer instance to acquire ownership of.
             */
            __host__ __device__ inline void transfer(buffer_t&& other) __devicesafe__
            {
                m_capacity = utility::exchange(other.m_capacity, 0);
                underlying_t::operator=(std::forward<decltype(other)>(other));
            }
    };

    /**
     * Copies data from a source to a target buffer.
     * @tparam T The buffers' content type.
     * @param target The buffer to copy data into.
     * @param source The buffer to copy data from.
     */
    template <typename T = void>
    inline void copy(memory::buffer_t<T>& target, const memory::buffer_t<T>& source) noexcept
    {
        const auto count = utility::min(target.capacity(), source.capacity());
        memory::copy(target.unwrap(), source.unwrap(), count);
    }

    /**
     * Initializes a buffer with zeroes.
     * @tparam T The buffer's content type.
     * @param target The buffer to be zero-initialized.
     */
    template <typename T = void>
    inline void zero(memory::buffer_t<T>& target) noexcept
    {
        memory::zero(target.unwrap(), target.capacity());
    }
}

namespace factory::memory
{
    /**
     * Allocates a buffer with the given capacity using an allocator.
     * @tparam T The buffer's elements type.
     * @param capacity The buffer's total element capacity.
     * @param allocator The allocator to create the new buffer with.
     * @return The allocated buffer.
     */
    template <typename T>
    inline auto buffer(size_t capacity = 1, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> museqa::memory::buffer_t<T>
    {
        return museqa::memory::buffer_t(pointer::shared<T>(capacity, allocator), capacity);
    }

    /**
     * Builds a new buffer by copying data from a raw pointer.
     * @tparam T The buffer's elements type.
     * @param ptr The target pointer to copy data from.
     * @param count The number of elements to be copied.
     * @param allocator The allocator to create the new buffer with.
     * @return The allocated buffer.
     */
    template <typename T>
    inline auto buffer(const T *ptr, size_t count = 1, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> museqa::memory::buffer_t<T>
    {
        auto buffer = factory::memory::buffer<T>(count, allocator);
        museqa::memory::copy<T>(buffer.unwrap(), ptr, count);
        return buffer;
    }

    /**
     * Builds a new buffer from a list of elements
     * @tparam T The buffer's elements type.
     * @param list The list of elements to fill the buffer with.
     * @param allocator The allocator to create the new buffer with.
     * @return The allocated buffer.
     */
    template <typename T>
    inline auto buffer(std::initializer_list<T> list, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> museqa::memory::buffer_t<T>
    {
        return factory::memory::buffer<T>(list.begin(), list.size(), allocator);
    }

    /**
     * Builds a new buffer by copying from an already existing buffer instance.
     * @tparam T The buffer's elements type.
     * @param buffer The buffer to copy the contents from.
     * @param allocator The allocator to create the new buffer with.
     * @return The allocated buffer.
     */
    template <typename T>
    inline auto buffer(
        const museqa::memory::buffer_t<T>& buffer
      , const museqa::memory::allocator_t& allocator = allocator<T>()
    ) -> museqa::memory::buffer_t<T>
    {
        return factory::memory::buffer<T>(buffer.unwrap(), buffer.capacity(), allocator);
    }

    /**
     * Builds a new buffer by copying from a vector instance.
     * @tparam T The buffer's elements type.
     * @param vector The vector to copy the contents from.
     * @param allocator The allocator to create the new buffer with.
     * @return The allocated buffer.
     */
    template <typename T>
    inline auto buffer(const std::vector<T>& vector, const museqa::memory::allocator_t& allocator = allocator<T>())
    -> museqa::memory::buffer_t<T>
    {
        return factory::memory::buffer<T>(vector.data(), vector.size(), allocator);
    }
}

MUSEQA_END_NAMESPACE

/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file A generic CUDA-compatible memory buffer.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <utility>

#include <fmt/format.h>

#include <museqa/utility.hpp>
#include <museqa/exception.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/shared.hpp>

namespace museqa
{
    namespace memory
    {
        /**
         * Represents a contiguous memory region reserved for the storage of a defined
         * number of elements of the given type.
         * @see std::string
         * @see std::vector
         * @since 1.0
         */
        template <typename T>
        class buffer
        {
            static_assert(!std::is_array<T>::value, "buffer elements cannot have array type");

          public:
            typedef T element_type;                                 /// The buffer's element type.
            typedef pointer::shared<element_type[]> pointer_type;   /// The buffer's internal pointer type.

          protected:
            pointer_type m_ptr;         /// The buffer's shared pointer instance.
            size_t m_capacity = 0;      /// The buffer's total capacity.

          public:
            __host__ __device__ inline buffer() noexcept = default;
            __host__ __device__ inline buffer(const buffer&) noexcept = default;
            __host__ __device__ inline buffer(buffer&&) noexcept = default;

            /**
             * Acquires ownership of a buffer pointer.
             * @param ptr The pointer instance to acquire ownership of.
             * @param capacity The acquired pointer's total capacity
             */
            __host__ __device__ inline explicit buffer(pointer_type&& ptr, size_t capacity) noexcept
              : m_ptr {std::forward<decltype(ptr)>(ptr)}
              , m_capacity {capacity}
            {}

            __host__ __device__ inline buffer& operator=(const buffer&) = default;
            __host__ __device__ inline buffer& operator=(buffer&&) = default;

            /**
             * Gives access to an element in the buffer by its offset.
             * @param offset The requested buffer offset to be retrieved.
             * @return The element at the requested buffer offset.
             */
            __host__ __device__ inline element_type& operator[](ptrdiff_t offset) noexcept(!safe)
            {
                museqa::assert(offset >= 0 && (size_t) offset < m_capacity, "buffer offset out of range");
                return m_ptr[offset];
            }

            /**
             * Gives access to a const-qualified element in the buffer by its offset.
             * @param offset The requested buffer offset to be retrieved.
             * @return The const-qualified element at the requested buffer offset.
             */
            __host__ __device__ inline const element_type& operator[](ptrdiff_t offset) const noexcept(!safe)
            {
                museqa::assert(offset >= 0 && (size_t) offset < m_capacity, "buffer offset out of range");
                return m_ptr[offset];
            }

            /**
             * Retrieves a slice of the buffer as a new buffer instance.
             * @param offset The initial buffer slice offset.
             * @param count The number of elements to slice from buffer.
             * @return The buffer's requested slice.
             */
            __host__ __device__ inline buffer slice(ptrdiff_t offset, size_t count = 0) noexcept(!safe)
            {
                museqa::assert(offset >= 0 && (size_t) offset < m_capacity, "buffer offset out of range");
                museqa::assert((size_t) offset + count <= m_capacity, "buffer slice out of range");
                return buffer {m_ptr.offset(offset), 0 == count ? m_capacity - (size_t) offset : count};
            }

            /**
             * The buffer's initial point as an iterator.
             * @return The pointer to the first element of iterator.
             */
            __host__ __device__ inline element_type *begin() noexcept
            {
                return (element_type*) m_ptr;
            }

            /**
             * The buffer's initial point as a const-qualified iterator.
             * @return The pointer to the first element of const-qualified iterator.
             */
            __host__ __device__ inline const element_type *begin() const noexcept
            {
                return (const element_type*) m_ptr;
            }

            /**
             * The buffer's final point as an iterator.
             * @return The pointer to the last element of iterator.
             */
            __host__ __device__ inline element_type *end() noexcept
            {
                return ((element_type*) m_ptr) + m_capacity;
            }

            /**
             * The buffer's final point as a const-qualified iterator.
             * @return The pointer to the last element of const-qualified iterator.
             */
            __host__ __device__ inline const element_type *end() const noexcept
            {
                return ((const element_type*) m_ptr) + m_capacity;
            }

            /**
             * Gives access to the buffer's internal pointer.
             * @return The buffer's internal pointer.
             */
            __host__ __device__ inline pointer_type& raw() noexcept
            {
                return m_ptr;
            }

            /**
             * Gives access to the buffer's internal pointer.
             * @return The buffer's internal const-qualified pointer.
             */
            __host__ __device__ inline const pointer_type& raw() const noexcept
            {
                return m_ptr;
            }

            /**
             * Informs the buffer's capacity.
             * @return The maximum number of elements in buffer.
             */
            __host__ __device__ inline size_t capacity() const noexcept
            {
                return m_capacity;
            }
        };
    }

    namespace factory
    {
        /**
         * Allocates a new buffer with the given allocator and capacity.
         * @tparam T The buffer's elements type.
         * @param allocator The allocator to create the new buffer with.
         * @param capacity The new buffer's total capacity.
         * @return The new allocated buffer.
         */
        template <typename T>
        inline auto buffer(const memory::allocator& allocator, size_t capacity = 1) noexcept -> memory::buffer<T>
        {
            auto raw = (T*) allocator.allocate<T>(capacity);
            auto ptr = typename memory::buffer<T>::pointer_type {raw, allocator};
            return memory::buffer<T> {std::move(ptr), capacity};
        }

        /**
         * Allocates a new buffer with the requested capacity.
         * @tparam T The buffer's elements type.
         * @param capacity The new buffer's total capacity.
         * @return The new allocated buffer.
         */
        template <typename T>
        inline auto buffer(size_t capacity = 1) noexcept -> memory::buffer<T>
        {
            auto allocator = factory::allocator<T[]>();
            return factory::buffer<T>(allocator, capacity);
        }

        /**
         * Copies data from a raw pointer into a buffer instance.
         * @tparam T The buffer's elements type.
         * @param ptr The target pointer to copy data from.
         * @param count The number of elements to be copied.
         * @return The new buffer with copied elements. 
         */
        template <typename T>
        inline auto buffer(const T *ptr, size_t count = 1) noexcept -> memory::buffer<T>
        {
            auto buffer = factory::buffer<T>(count);
            memcpy(buffer.begin(), ptr, sizeof(T) * count);
            return buffer;
        }

        /**
         * Creates a new buffer with the given content.
         * @tparam T The buffer's elements type.
         * @param list The list of elements to fill the buffer with.
         * @return The new buffer with the given contents.
         */
        template <typename T>
        inline auto buffer(std::initializer_list<T> list) noexcept -> memory::buffer<T>
        {
            return factory::buffer(list.begin(), list.size());
        }

        /**
         * Copies data from an already existing buffer instance.
         * @tparam T The buffer's elements type.
         * @param buffer The buffer to be copied into a new instance.
         * @return The new buffer with copied contents.
         */
        template <typename T>
        inline auto buffer(const memory::buffer<T>& buffer) noexcept -> memory::buffer<T>
        {
            return factory::buffer(buffer.begin(), buffer.capacity());
        }

        /**
         * Copies data from a vector instance.
         * @tparam T The buffer's elements type.
         * @param vector The vector to copy the contents from.
         * @return The new buffer with the vector's contents.
         */
        template <typename T>
        inline auto buffer(const std::vector<T>& vector) noexcept -> memory::buffer<T>
        {
            return factory::buffer(vector.data(), vector.size());
        }
    }
}

/**
 * Implements a string formatter for a generic buffer type, thus giving generic
 * buffer instances ease of printing whenever its contents type is printable.
 * @tparam T The buffer's contents type.
 * @since 1.0
 */
template <typename T>
class fmt::formatter<museqa::memory::buffer<T>>
{
  private:
    typedef museqa::memory::buffer<T> target_type;

  public:
    /**
     * Evaluates the formatter's parsing context.
     * @tparam P The parsing context type.
     * @param ctx The parsing context instance.
     * @return The processed and evaluated parsing context.
     */
    template <typename P>
    constexpr auto parse(P& ctx) -> decltype(ctx.begin())
    {
        return ctx.begin();
    }

    /**
     * Formats the buffer into a printable string.
     * @tparam F The formatting context type.
     * @param buffer The buffer to be formatted into a string.
     * @param ctx The formatting context instance.
     * @return The formatting context instance.
     */
    template <typename F>
    auto format(const target_type& buffer, F& ctx) -> decltype(ctx.out())
    {
        return fmt::format_to(ctx.out(), "[{}]", fmt::join(buffer.begin(), buffer.end(), ", "));
    }
};

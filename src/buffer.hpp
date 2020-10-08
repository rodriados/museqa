/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Generic CUDA-compatible buffer implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

#include "utils.hpp"
#include "pointer.hpp"
#include "allocator.hpp"
#include "exception.hpp"

namespace museqa
{
    /**
     * Creates a general-purpose buffer. The buffer's idea is to store all of its
     * data contiguously in memory. Originally, the buffer is not dynamic.
     * @tparam T The buffer contents type.
     * @see std::string
     * @see std::vector
     * @since 0.1.1
     */
    template <typename T>
    class buffer
    {
        static_assert(!std::is_array<T>::value, "buffer element cannot be of array type");

        public:
            using element_type = T;                         /// The buffer's element type.
            using pointer_type = pointer<element_type[]>;   /// The buffer's pointer type.

        protected:
            pointer_type m_ptr;     /// The pointer to the corresponding buffer's memory area.
            size_t m_size = 0;      /// The number of elements in buffer.

        public:
            __host__ __device__ inline buffer() noexcept = default;
            __host__ __device__ inline buffer(const buffer&) noexcept = default;
            __host__ __device__ inline buffer(buffer&&) noexcept = default;

            /**
             * Acquires the ownership of a raw buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param size The size of buffer to acquire.
             */
            inline explicit buffer(element_type *ptr, size_t size) noexcept
            :   m_ptr {ptr}
            ,   m_size {size}
            {}

            /**
             * Acquires the ownership of a buffer pointer.
             * @param ptr The buffer pointer to acquire.
             * @param size The size of buffer to acquire.
             */
            __host__ __device__ inline explicit buffer(pointer_type&& ptr, size_t size) noexcept
            :   m_ptr {std::forward<decltype(ptr)>(ptr)}
            ,   m_size {size}
            {}

            __host__ __device__ inline buffer& operator=(const buffer&) = default;
            __host__ __device__ inline buffer& operator=(buffer&&) = default;

            /**
             * Gives access to a specific location in buffer's data.
             * @param offset The requested buffer offset.
             * @return The buffer's position pointer.
             */
            __host__ __device__ inline element_type& operator[](ptrdiff_t offset)
            {
                enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
                return m_ptr[offset];
            }

            /**
             * Gives constant access to a specific location in buffer's data.
             * @param offset The requested buffer offset.
             * @return The buffer's position constant pointer.
             */
            __host__ __device__ inline const element_type& operator[](ptrdiff_t offset) const
            {
                enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
                return m_ptr[offset];
            }

            /**
             * Allows the buffer to be traversed as an iterator.
             * @return The pointer to the first element in the buffer.
             */
            __host__ __device__ inline element_type *begin() noexcept
            {
                return raw();
            }

            /**
             * Allows the buffer to be traversed as a const-iterator.
             * @return The const-pointer to the first element in the buffer.
             */
            __host__ __device__ inline const element_type *begin() const noexcept
            {
                return raw();
            }

            /**
             * Returns the pointer to the end of buffer.
             * @return The pointer after the last element in buffer.
             */
            __host__ __device__ inline element_type *end() noexcept
            {
                return raw() + size();
            }

            /**
             * Returns the const-pointer to the end of buffer.
             * @return The const-pointer after the last element in buffer.
             */
            __host__ __device__ inline const element_type *end() const noexcept
            {
                return raw() + size();
            }

            /**
             * Gives access to raw buffer's data.
             * @return The buffer's internal pointer.
             */
            __host__ __device__ inline element_type *raw() noexcept
            {
                return &m_ptr;
            }

            /**
             * Gives constant access to raw buffer's data.
             * @return The buffer's internal constant pointer.
             */
            __host__ __device__ inline const element_type *raw() const noexcept
            {
                return m_ptr.get();
            }

            /**
             * Gives access to an offset of the buffer's pointer.
             * @param offset The requested buffer offset.
             * @return The buffer's offset pointer.
             */
            __host__ __device__ inline pointer_type offset(ptrdiff_t offset) noexcept
            {
                enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
                return m_ptr.offset(offset);
            }

            /**
             * Exposes the buffer's internal allocator instance.
             * @return The buffer's internal allocator.
             */
            __host__ __device__ inline msa::allocator allocator() const noexcept
            {
                return m_ptr.allocator();
            }

            /**
             * Informs the buffer's number of elements.
             * @return The number of elements in buffer.
             */
            __host__ __device__ inline size_t size() const noexcept
            {
                return m_size;
            }

            /**
             * Copies data from an existing buffer instance.
             * @param buf The target buffer to copy data from.
             * @return A newly created buffer instance.
             */
            static inline buffer copy(const buffer& buf) noexcept
            {
                return make(buf.size()).copy_from(buf.raw());
            }

            /**
             * Copies data from a vector instance.
             * @param vector The target vector instance to copy data from.
             * @return A newly created buffer instance.
             */
            static inline buffer copy(const std::vector<element_type>& vector) noexcept
            {
                return make(vector.size()).copy_from(vector.data());
            }

            /**
             * Copies data from an existing pointer.
             * @param ptr The target pointer to copy from.
             * @param count The number of elements to copy.
             * @return A newly created buffer instance.
             */
            static inline buffer copy(const element_type *ptr, size_t count) noexcept
            {
                return make(count).copy_from(ptr);
            }

            /**
             * Creates a new buffer of given size.
             * @param size The buffer's number of elements.
             * @return The newly created buffer instance.
             */
            static inline buffer make(size_t size = 1) noexcept
            {
                return buffer {pointer_type::make(size), size};
            }

            /**
             * Creates a new buffer of given size with an allocator.
             * @param allocator The allocator to be used to new buffer.
             * @param size The buffer's number of elements.
             * @return The newly created buffer instance.
             */
            static inline buffer make(const msa::allocator& allocator, size_t size = 1) noexcept
            {
                return buffer {pointer_type::make(allocator, size), size};
            }
            
        protected:
            /**
             * Effectively copies data from the given pointer.
             * @param ptr The pointer to copy data from.
             * @return The current buffer instance.
             */
            inline buffer copy_from(const element_type *ptr) noexcept
            {
                memcpy(raw(), ptr, sizeof(element_type) * size());
                return *this;
            }
    };

    /**
     * Manages a slice of a buffer. The buffer must have already been initialized
     * and will have boundaries checked according to slice pointers.
     * @tparam T The buffer contents type.
     * @since 0.1.1
     */
    template <typename T>
    class slice_buffer : public buffer<T>
    {
        protected:
            using underlying_buffer = buffer<T>;    /// The underlying buffer type.

        protected:
            ptrdiff_t m_displ = 0;      /// The slice displacement in relation to original buffer.

        public:
            inline slice_buffer() noexcept = default;
            inline slice_buffer(const slice_buffer&) noexcept = default;
            inline slice_buffer(slice_buffer&&) noexcept = default;

            /**
             * Instantiates a new slice buffer.
             * @param tgt The target buffer to which the slice shall relate to.
             * @param displ The initial displacement of slice.
             * @param size The number of elements in the slice.
             */
            inline slice_buffer(underlying_buffer& tgt, ptrdiff_t displ = 0, size_t size = 0)
            :   underlying_buffer {std::move(tgt.offset(displ)), size}
            ,   m_displ {displ}
            {
                enforce(size_t(m_displ + size) <= tgt.size(), "slice out of buffer's range");
            }

            /**
             * Instantiates a slice by copying slice pointers into another buffer.
             * @param tgt The target buffer to which the slice shall relate to.
             * @param base The slice data to be put into the new target.
             */
            inline slice_buffer(underlying_buffer& tgt, const slice_buffer& base)
            :   underlying_buffer {std::move(tgt.offset(base.displ())), base.size()}
            ,   m_displ {base.displ()}
            {
                enforce(size_t(m_displ + base.size()) <= tgt.size(), "slice out of buffer's range");
            }

            inline slice_buffer& operator=(const slice_buffer&) = default;
            inline slice_buffer& operator=(slice_buffer&&) = default;

            /**
             * Informs the pointer displacement in relation to the original buffer.
             * @return The buffer's slice pointer displacement.
             */
            __host__ __device__ inline ptrdiff_t displ() const noexcept
            {
                return m_displ;
            }

        private:
            using underlying_buffer::copy;
            using underlying_buffer::make;
    };
}

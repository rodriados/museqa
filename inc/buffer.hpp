/** 
 * Multiple Sequence Alignment buffer header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef BUFFER_HPP_INCLUDED
#define BUFFER_HPP_INCLUDED

#include <cstdint>
#include <cstring>
#include <vector>

#include <utils.hpp>
#include <pointer.hpp>
#include <exception.hpp>

/**
 * Creates a general-purpose buffer. The buffer's idea is to store all of its data
 * contiguously in memory. Originally, the buffer is not growable.
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
        using element_type = T;                             /// The buffer's element size.
        using pointer_type = pointer<element_type[]>;       /// The buffer's pointer type.
        using allocator_type = allocatr<element_type[]>;    /// The buffer's allocator type.

    protected:
        pointer_type mptr;              /// The pointer to the corresponding buffer's memory area.
        size_t msize = 0;               /// The number of elements in buffer.

    public:
        inline buffer() noexcept = default;
        inline buffer(const buffer&) noexcept = default;
        inline buffer(buffer&&) noexcept = default;

        /**
         * Acquires the ownership of a raw buffer pointer.
         * @param ptr The buffer pointer to acquire.
         * @param size The size of buffer to acquire.
         */
        inline explicit buffer(element_type *ptr, size_t size)
        :   mptr {ptr}
        ,   msize {size}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The buffer pointer to acquire.
         * @param size The size of buffer to acquire.
         */
        inline explicit buffer(pointer_type&& ptr, size_t size)
        :   mptr {std::forward<decltype(ptr)>(ptr)}
        ,   msize {size}
        {}

        inline buffer& operator=(const buffer&) = default;
        inline buffer& operator=(buffer&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        __host__ __device__ inline element_type& operator[](ptrdiff_t offset)
        {
            enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
            return mptr[offset];
        }

        /**
         * Gives constant access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position constant pointer.
         */
        __host__ __device__ inline const element_type& operator[](ptrdiff_t offset) const
        {
            enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
            return mptr[offset];
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
            return begin() + size();
        }

        /**
         * Returns the const-pointer to the end of buffer.
         * @return The const-pointer after the last element in buffer.
         */
        __host__ __device__ inline const element_type *end() const noexcept
        {
            return begin() + size();
        }

        /**
         * Gives access to raw buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__ inline element_type *raw() noexcept
        {
            return &mptr;
        }

        /**
         * Gives constant access to raw buffer's data.
         * @return The buffer's internal constant pointer.
         */
        __host__ __device__ inline const element_type *raw() const noexcept
        {
            return mptr.get();
        }

        /**
         * Gives access to an offset of the buffer's pointer.
         * @param offset The requested buffer offset.
         * @return The buffer's offset pointer.
         */
        __host__ __device__ inline pointer_type offset(ptrdiff_t offset)
        {
            enforce(offset >= 0 && size_t(offset) < size(), "buffer offset out of range");
            return mptr.offset(offset);
        }

        /**
         * Informs the buffer's number of elements.
         * @return The number of elements in buffer.
         */
        __host__ __device__ inline size_t size() const noexcept
        {
            return msize;
        }

        /**
         * Copies data from an existing buffer instance.
         * @param buf The target buffer to copy data from.
         * @return A newly created buffer instance.
         */
        static inline buffer copy(const buffer& buf)
        {
            return make(buf.size()).copy_from(buf.raw());
        }

        /**
         * Copies data from a vector instance.
         * @param vector The target vector instance to copy data from.
         * @return A newly created buffer instance.
         */
        static inline buffer copy(const std::vector<element_type>& vector)
        {
            return make(vector.size()).copy_from(vector.data());
        }

        /**
         * Copies data from an existing pointer.
         * @param ptr The target pointer to copy from.
         * @param count The number of elements to copy.
         * @return A newly created buffer instance.
         */
        static inline buffer copy(const element_type *ptr, size_t count)
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
            return make(allocator_type {}, size);
        }

        /**
         * Creates a new buffer of given size with an allocator.
         * @param alloc The allocator to be used to new buffer.
         * @param size The buffer's number of elements.
         * @return The newly created buffer instance.
         */
        static inline buffer make(const allocator_type& alloc, size_t size = 1) noexcept
        {
            return buffer {pointer_type::make(alloc, size), size};
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
        ptrdiff_t mdispl = 0;       /// The slice displacement in relation to original buffer.

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
        ,   mdispl {displ}
        {
            enforce(size_t(mdispl + size) <= tgt.size(), "slice out of buffer's range");
        }

        /**
         * Instantiates a slice by copying slice pointers into another buffer.
         * @param tgt The target buffer to which the slice shall relate to.
         * @param base The slice data to be put into the new target.
         */
        inline slice_buffer(underlying_buffer& tgt, const slice_buffer& base)
        :   underlying_buffer {std::move(tgt.offset(base.displ())), base.size()}
        ,   mdispl {base.displ()}
        {
            enforce(size_t(mdispl + base.size()) <= tgt.size(), "slice out of buffer's range");
        }

        inline slice_buffer& operator=(const slice_buffer&) = default;
        inline slice_buffer& operator=(slice_buffer&&) = default;

        /**
         * Informs the pointer displacement in relation to the original buffer.
         * @return The buffer's slice pointer displacement.
         */
        __host__ __device__ inline ptrdiff_t displ() const noexcept
        {
            return mdispl;
        }

    private:
        using underlying_buffer::copy;
        using underlying_buffer::make;
};

#endif
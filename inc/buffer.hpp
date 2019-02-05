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

#include "utils.hpp"
#include "pointer.hpp"

/**
 * The base of a buffer class.
 * @since 0.1.alpha
 */
template <typename T>
class BaseBuffer
{
    protected:
        AutoPointer<T[]> buffer;    /// The buffer being encapsulated.
        size_t size = 0;            /// The number of buffer blocks.

    public:
        BaseBuffer() = default;
        BaseBuffer(const BaseBuffer<T>&) = default;
        BaseBuffer(BaseBuffer<T>&&) = default;

        /**
         * Encapsulates a buffer pointer.
         * @param ptr The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline BaseBuffer(const RawPointer<T>& ptr, size_t size)
        :   buffer {ptr}
        ,   size {size}
        {}

        /**
         * Encapsulates a buffer pointer.
         * @param buffer The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline BaseBuffer(const Pointer<T> buffer, size_t size, const Deleter<T> dfunc = nullptr)
        :   buffer {buffer, dfunc}
        ,   size {size}
        {}

        BaseBuffer<T>& operator=(const BaseBuffer<T>&) = default;
        BaseBuffer<T>& operator=(BaseBuffer<T>&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        cudadecl inline Pure<T>& operator[](ptrdiff_t offset) const
        {
            return buffer.getOffset(offset);
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        cudadecl inline const Pointer<T> getBuffer() const
        {
            return buffer.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        cudadecl inline const AutoPointer<T[]>& getPointer() const
        {
            return buffer;
        }

        /**
         * Informs the buffer's number of blocks.
         * @return The number of buffer blocks.
         */
        cudadecl inline size_t getSize() const
        {
            return size;
        }

    protected:
        /**
         * Instantiates a new buffer from an already existing pointer.
         * @param ptr The buffer pointer.
         * @param size The size of buffer.
         */
        inline explicit BaseBuffer(const AutoPointer<T[]>& ptr, size_t size)
        :   buffer {ptr}
        ,   size {size}
        {}
};

/**
 * Creates a general-purpose buffer. The buffer created is constant and
 * cannot be changed. For mutable buffers, please use strings or vectors.
 * @see std::string
 * @see std::vector
 * @since 0.1.alpha
 */
template <typename T>
class Buffer : public BaseBuffer<T>
{
    public:
        Buffer() = default;
        Buffer(const Buffer<T>&) = default;
        Buffer(Buffer<T>&&) = default;

        using BaseBuffer<T>::BaseBuffer;

        /**
         * Constructs a new buffer from an already existing base buffer.
         * @param buffer The buffer to be copied.
         */
        inline Buffer(const BaseBuffer<T>& buffer)
        :   BaseBuffer<T> {}
        {
            copy(buffer.getBuffer(), buffer.getSize());
        }

        /**
         * Constructs a new buffer from an already existing one.
         * @param buffer The buffer to be copied.
         * @param size The number of buffer blocks being copied.
         */
        inline Buffer(const Pointer<T> buffer, size_t size)
        :   BaseBuffer<T> {}
        {
            copy(buffer, size);
        }

        /**
         * Constructs a new buffer from a vector.
         * @param vector The vector from which the buffer will be created.
         */
        inline Buffer(const std::vector<T>& vector)
        :   BaseBuffer<T> {}
        {
            copy(vector.data(), vector.size());
        }

        Buffer<T>& operator=(const Buffer<T>&) = default;
        Buffer<T>& operator=(Buffer<T>&&) = default;

    protected:
        /**
         * Copies an existing buffer's data.
         * @param buffer The buffer to be copied.
         * @param size The buffer's data blocks number.
         */
        inline void copy(const Pointer<T> buffer, size_t size)
        {
            this->size = size;
            this->buffer = new T[size];
            memcpy(this->buffer.get(), buffer, sizeof(T) * size);
        }
};

/**
 * Represents a slice of a buffer.
 * @since 0.1.alpha
 */
template <typename T>
class BufferSlice : public BaseBuffer<T>
{
    protected:
        ptrdiff_t displ = 0;    /// The displacement in relation to buffer start.

    public:
        BufferSlice() = default;
        BufferSlice(const BufferSlice<T>&) = default;
        BufferSlice(BufferSlice<T>&&) = default;

        /**
         * Constructs a new buffer slice.
         * @param target The target buffer to which the slice is related to.
         * @param displ The displacement of the slice.
         * @param size The number of blocks of the slice.
         */
        inline BufferSlice(const BaseBuffer<T>& target, ptrdiff_t displ = 0, size_t size = 0)
        :   BaseBuffer<T> {target.getPointer() + displ, size}
        ,   displ {displ}
        {}

        /**
         * Constructs a slice from an already existing instance into another buffer.
         * @param target The target buffer to which the slice is related to.
         * @param slice The slice data to be put into the new target.
         */
        inline BufferSlice(const BaseBuffer<T>& target, const BufferSlice<T>& slice)
        :   BaseBuffer<T> {target.getPointer() + slice.getDispl(), slice.getSize()}
        ,   displ {slice.getDispl()}
        {}

        BufferSlice<T>& operator=(const BufferSlice<T>&) = default;
        BufferSlice<T>& operator=(BufferSlice<T>&&) = default;

        /**
         * Informs the displacement of data pointed by the slice.
         * @return The buffer's slice displacement.
         */
        cudadecl inline ptrdiff_t getDispl() const
        {
            return displ;
        }
};

#endif

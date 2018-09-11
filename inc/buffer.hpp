/** 
 * Multiple Sequence Alignment buffer header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef BUFFER_HPP_INCLUDED
#define BUFFER_HPP_INCLUDED

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "device.cuh"

/**
 * The base of a buffer class.
 * @since 0.1.alpha
 */
template<typename T>
class BaseBuffer
{
    protected:
        size_t size = 0;            /// The number of buffer blocks.
        std::shared_ptr<T> buffer;  /// The buffer being encapsulated.

    public:
        BaseBuffer() = default;
        BaseBuffer(const BaseBuffer<T>&) = default;
        BaseBuffer(BaseBuffer<T>&&) = default;

        /**
         * Encapsulates a buffer pointer.
         * @param buffer The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(T *buffer, size_t size)
        :   buffer(buffer)
        ,   size(size) {}

        BaseBuffer<T>& operator=(const BaseBuffer<T>&) = default;
        BaseBuffer<T>& operator=(BaseBuffer<T>&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        cudadecl inline const T& operator[](ptrdiff_t offset) const
        {
            return this->buffer.get()[offset];
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        cudadecl inline T *getBuffer() const
        {
            return this->buffer.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        cudadecl inline std::shared_ptr<T>& getPointer() const
        {
            return this->buffer;
        }

        /**
         * Informs the buffer's number of blocks.
         * @return The number of buffer blocks.
         */
        cudadecl inline size_t getSize() const
        {
            return this->size;
        }
};

/**
 * Creates a general-purpose buffer. The buffer created is constant and
 * cannot be changed. For mutable buffers, please use strings or vectors.
 * @see std::string
 * @see std::vector
 * @since 0.1.alpha
 */
template<typename T>
class Buffer : public BaseBuffer<T>
{
    public:
        Buffer() = default;
        Buffer(const Buffer<T>&) = default;
        Buffer(Buffer<T>&&) = default;

        /**
         * Constructs a new buffer from an already existing one.
         * @param buffer The buffer to be copied.
         * @param size The number of buffer blocks being copied.
         */
        inline Buffer(const T *buffer, size_t size)
        :   BaseBuffer<T>()
        {
            this->copy(buffer, size);
        }

        /**
         * Constructs a new buffer from an already existing base buffer.
         * @param buffer The buffer to be copied.
         */
        inline Buffer(const BaseBuffer<T>& buffer)
        :   BaseBuffer<T>()
        {
            this->copy(buffer.getBuffer(), buffer.getSize());
        }

        /**
         * Constructs a new buffer from a vector.
         * @param vector The vector from which the buffer will be created.
         */
        inline Buffer(const std::vector<T>& vector)
        :   BaseBuffer<T>()
        {
            this->copy(vector.data(), vector.size());
        }

        Buffer<T>& operator=(const Buffer<T>&) = default;
        Buffer<T>& operator=(Buffer<T>&&) = default;

    protected:
        /**
         * Copies an existing buffer's data.
         * @param buffer The buffer to be copied.
         * @param size The buffer's data blocks number.
         */
        inline void copy(const T *buffer, size_t size)
        {
            this->size = size;
            this->buffer = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
            memcpy(this->buffer.get(), buffer, sizeof(T) * size);
        }
};

/**
 * Represents a slice of a buffer.
 * @since 0.1.alpha
 */
template<typename T>
class BufferSlice : public BaseBuffer<T>
{
    protected:
        ptrdiff_t displ = 0;    /// The slice displacement in relation to the buffer.

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
        :   BaseBuffer<T>(target)
        ,   displ(displ)
        {
            this->size = this->size < this->displ + size
                ? this->size - this->displ
                : size;
        }

        /**
         * Constructs a slice from an already existing instance into another buffer.
         * @param target The target buffer to which the slice is related to.
         * @param slice The slice data to be put into the new target.
         */
        inline BufferSlice(const BaseBuffer<T>& target, const BufferSlice<T>& slice)
        :   BaseBuffer<T>(target)
        ,   displ(slice.getDispl())
        {
            this->size = this->size < this->displ + slice.getSize()
                ? this->size - this->displ
                : slice.getSize();
        }

        BufferSlice<T>& operator=(const BufferSlice<T>&) = default;
        BufferSlice<T>& operator=(BufferSlice<T>&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        cudadecl inline const T& operator[](ptrdiff_t offset) const
        {
            return this->buffer.get()[this->displ + offset];
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        cudadecl inline T *getBuffer() const
        {
            return this->buffer.get() + this->displ;
        }

        /**
         * Informs the displacement of data pointed by the slice.
         * @return The buffer's slice displacement.
         */
        cudadecl inline ptrdiff_t getDispl() const
        {
            return this->displ;
        }
};

#endif

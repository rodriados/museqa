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
#include "exception.hpp"

/**
 * The base of a general-purpose buffer. The buffer's idea is to store all of
 * its data contiguously in memory.
 * @tparam T The buffer contents type.
 * @since 0.1.1
 */
template <typename T>
class BaseBuffer
{
    protected:
        Pointer<T[]> ptr;       /// The pointer to the buffer being encapsulated.
        size_t size = 0;        /// The number of elements in buffer.

    public:
        inline BaseBuffer() noexcept = default;
        inline BaseBuffer(const BaseBuffer&) noexcept = default;
        inline BaseBuffer(BaseBuffer&&) noexcept = default;

        /**
         * Creates a new buffer by allocating memory.
         * @param size The number of elements to allocate memory for.
         */
        inline explicit BaseBuffer(size_t size)
        :   ptr {new T[size]}
        ,   size {size}
        {
            enforce(ptr != nullptr, "could not allocate buffer memory");
        }

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(const Pointer<T[]>& ptr, size_t size) noexcept
        :   ptr {ptr}
        ,   size {size}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The raw pointer instance to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(const RawPointer<T>& ptr, size_t size) noexcept
        :   ptr {ptr}
        ,   size {size}
        {}

        inline BaseBuffer& operator=(const BaseBuffer&) = default;
        inline BaseBuffer& operator=(BaseBuffer&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        __host__ __device__ inline T& operator[](ptrdiff_t offset)
        {
            enforce(0 <= offset && static_cast<size_t>(offset) < getSize(), "buffer offset out of range");
            return ptr[offset];
        }

        /**
         * Gives constant access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position constant pointer.
         */
        __host__ __device__ inline const T& operator[](ptrdiff_t offset) const
        {
            enforce(0 <= offset && static_cast<size_t>(offset) < getSize(), "buffer offset out of range");
            return const_cast<const T&>(ptr[offset]);
        }

        /**
         * Allows the buffer to be traversed as an iterator.
         * @return The pointer to the first element in the buffer.
         */
        __host__ __device__ inline T *begin() noexcept
        {
            return getBuffer();
        }

        /**
         * Allows the buffer to be traversed as a const-iterator.
         * @return The const-pointer to the first element in the buffer.
         */
        __host__ __device__ inline const T *begin() const noexcept
        {
            return getBuffer();
        }

        /**
         * Returns the pointer to the end of buffer.
         * @return The pointer after the last element in buffer.
         */
        __host__ __device__ inline T *end() noexcept
        {
            return begin() + size;
        }

        /**
         * Returns the const-pointer to the end of buffer.
         * @return The const-pointer after the last element in buffer.
         */
        __host__ __device__ inline const T *end() const noexcept
        {
            return begin() + size;
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__ inline T *getBuffer() noexcept
        {
            return &ptr;
        }

        /**
         * Gives constant access to buffer's data.
         * @return The buffer's internal constant pointer.
         */
        __host__ __device__ inline const T *getBuffer() const noexcept
        {
            return ptr.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        __host__ __device__ inline const Pointer<T[]>& getPointer() const noexcept
        {
            return ptr;
        }

        /**
         * Gives access to an offset pointer.
         * @param offset The offset to apply to pointer.
         * @return The buffer's offset pointer.
         */
        inline Pointer<T[]> getOffsetPointer(ptrdiff_t offset)
        {
            enforce(0 <= offset && static_cast<size_t>(offset) < getSize(), "buffer offset out of range");
            return ptr.getOffsetPointer(offset);
        }

        /**
         * Informs the buffer's number of blocks.
         * @return The number of buffer blocks.
         */
        __host__ __device__ inline size_t getSize() const noexcept
        {
            return size;
        }
};

/**
 * Creates a general-purpose contiguous buffer.
 * @tparam T The buffer contents type.
 * @see std::string
 * @see std::vector
 * @since 0.1.1
 */
template <typename T>
class Buffer : public BaseBuffer<T>
{
    public:
        inline Buffer() noexcept = default;
        inline Buffer(const Buffer&) noexcept = default;
        inline Buffer(Buffer&&) noexcept = default;

        using BaseBuffer<T>::BaseBuffer;

        /** 
         * Constructs a new buffer from an different buffer type instance.
         * @param buffer The buffer to be copied.
         */ 
        inline Buffer(const BaseBuffer<T>& buffer)
        :   BaseBuffer<T> {buffer.getSize()}
        {   
            copy(buffer.getBuffer());
        }

        /**
         * Copies the contents of a not-ownable already existing buffer.
         * @param ptr The pointer of buffer to be copied.
         * @param size The size of buffer to encapsulate.
         */
        inline Buffer(const T *ptr, size_t size)
        :   BaseBuffer<T> {size}
        {
            copy(ptr);
        }

        /**
         * Constructs a new buffer from a vector.
         * @param vector The vector from which the buffer will be created.
         */
        inline Buffer(const std::vector<T>& vector)
        :   BaseBuffer<T> {vector.size()}
        {
            copy(vector.data());
        }

        inline Buffer& operator=(const Buffer&) = default;
        inline Buffer& operator=(Buffer&&) = default;

    protected:
        /**
         * Copies an existing buffer's data.
         * @param ptr The pointer of buffer to be copied.
         */
        inline void copy(const T *ptr) noexcept
        {
            memcpy(this->getBuffer(), ptr, sizeof(T) * this->getSize());
        }
};

/**
 * Manages a slice of a buffer. The buffer must have already been initialized
 * and will have boundaries checked according to slice pointers.
 * @tparam T The buffer contents type.
 * @since 0.1.1
 */
template <typename T>
class BufferSlice : public BaseBuffer<T>
{
    protected:
        ptrdiff_t displ = 0;    /// The slice displacement in relation to original buffer.

    public:
        inline BufferSlice() noexcept = default;
        inline BufferSlice(const BufferSlice&) noexcept = default;
        inline BufferSlice(BufferSlice&&) noexcept = default;

        /**
         * Instantiates a new buffer slice.
         * @param target The target buffer to which the slice shall relate to.
         * @param displ The initial displacement of slice.
         * @param size The number of elements in the slice.
         */
        inline BufferSlice(BaseBuffer<T>& target, ptrdiff_t displ = 0, size_t size = 0)
        :   BaseBuffer<T> {target.getOffsetPointer(displ), size}
        ,   displ {displ}
        {
            enforce(
                static_cast<size_t>(size + displ) < target.getSize()
            ,   "slice initialized out of range"
            );
        }

        /**
         * Instantiates a slice by copying slice pointers into another buffer.
         * @param target The target buffer to which the slice shall relate to.
         * @param slice The slice data to be put into the new target.
         */
        inline BufferSlice(BaseBuffer<T>& target, const BufferSlice& slice)
        :   BaseBuffer<T> {target.getOffsetPointer(slice.getDispl()), slice.getSize()}
        ,   displ {slice.getDispl()}
        {
            enforce(
                static_cast<size_t>(slice.getSize() + slice.getDispl()) < target.getSize()
            ,   "slice initialized out of range"
            );
        }

        inline BufferSlice& operator=(const BufferSlice&) = default;
        inline BufferSlice& operator=(BufferSlice&&) = default;

        /**
         * Informs the displacement pointer in relation to original buffer.
         * @return The buffer's slice displacement.
         */
        __host__ __device__ inline ptrdiff_t getDispl() const noexcept
        {
            return displ;
        }
};

#endif
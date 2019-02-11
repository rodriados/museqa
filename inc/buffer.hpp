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
 * The base of a general-purpose buffer.
 * @tparam T The buffer type.
 * @since 0.1.1
 */
template <typename T>
class BaseBuffer
{
    protected:
        AutoPointer<T[]> ptr;       /// The pointer to the buffer being encapsulated.
        uint32_t size = 0;          /// The number of elements in the buffer.

    public:
        BaseBuffer() = default;
        BaseBuffer(const BaseBuffer<T>&) = default;
        BaseBuffer(BaseBuffer<T>&&) = default;

        /**
         * Creates a new buffer by allocating memory.
         * @param size The number of elements to allocate memory for.
         */
        inline explicit BaseBuffer(size_t size)
        :   ptr {new T[size]}
        ,   size {static_cast<uint32_t>(size)}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(const AutoPointer<T[]>& ptr, size_t size)
        :   ptr {ptr}
        ,   size {size}
        {}

        BaseBuffer<T>& operator=(const BaseBuffer<T>&) = default;
        BaseBuffer<T>& operator=(BaseBuffer<T>&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        __host__ __device__ inline Pure<T>& operator[](ptrdiff_t offset) const
        {
#ifdef msa_compile_cython
            if(static_cast<unsigned>(offset) >= getSize())
                throw Exception("Buffer offset out of range");
#endif
            return ptr.getOffset(offset);
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__ inline Pointer<T> getBuffer() const
        {
            return ptr.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        __host__ __device__ inline const AutoPointer<T[]>& getPointer() const
        {
            return ptr;
        }

        /**
         * Informs the buffer's number of blocks.
         * @return The number of buffer blocks.
         */
        __host__ __device__ inline size_t getSize() const
        {
            return size;
        }
};

/**
 * Creates a general-purpose buffer. The buffer created is constant and
 * cannot be changed. For mutable buffers, please use strings or vectors.
 * @tparam T The buffer type.
 * @see std::string
 * @see std::vector
 * @since 0.1.1
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
         * Constructs a new buffer from an different buffer type instance.
         * @param buffer The buffer to be copied.
         */ 
        inline Buffer(const BaseBuffer<T>& buffer)
        :   BaseBuffer<T> {buffer.getSize()}
        {   
            copy(buffer.getBuffer());
        }

        /**
         * Copies the contents of a not-owning already existing buffer.
         * @param ptr The pointer of buffer to be copied.
         * @param size The size of buffer to encapsulate.
         */
        inline Buffer(Pointer<const T> ptr, size_t size)
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

        Buffer<T>& operator=(const Buffer<T>&) = default;
        Buffer<T>& operator=(Buffer<T>&&) = default;

    protected:
        /**
         * Copies an existing buffer's data.
         * @param ptr The pointer of buffer to be copied.
         */
        inline void copy(Pointer<const T> ptr)
        {
            memcpy(this->ptr, ptr, sizeof(T) * this->size);
        }
};

#endif
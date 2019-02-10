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
 * Creates a general-purpose buffer. The buffer created is constant and
 * cannot be changed. For mutable buffers, please use strings or vectors.
 * @tparam T The buffer type.
 * @see std::string
 * @see std::vector
 * @since 0.1.1
 */
template <typename T>
class Buffer
{
    protected:
        AutoPointer<T[]> ptr;       /// The pointer to the buffer being encapsulated.
        size_t size = 0;            /// The number of elements in the buffer.

    public:
        Buffer() = default;
        Buffer(const Buffer<T>&) = default;
        Buffer(Buffer<T>&&) = default;

        /**
         * Creates a new buffer by allocating memory.
         * @param size The number of elements to allocate memory for.
         */
        inline explicit Buffer(size_t size)
        :   ptr {new T[size]}
        ,   size {size}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline Buffer(const AutoPointer<T[]>& ptr, size_t size)
        :   ptr {ptr}
        ,   size {size}
        {}

        /**
         * Copies the contents of a not-owning already existing buffer.
         * @param ptr The pointer of buffer to be copied.
         * @param size The size of buffer to encapsulate.
         */
        inline Buffer(Pointer<const T> ptr, size_t size)
        :   Buffer {size}
        {
            copy(ptr);
        }

        /**
         * Constructs a new buffer from a vector.
         * @param vector The vector from which the buffer will be created.
         */
        inline Buffer(const std::vector<T>& vector)
        :   Buffer {vector.size()}
        {
            copy(vector.data());
        }

        Buffer<T>& operator=(const Buffer<T>&) = default;
        Buffer<T>& operator=(Buffer<T>&&) = default;

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        cudadecl inline Pure<T>& operator[](ptrdiff_t offset) const
        {
#ifdef msa_compile_cython
            if(offset >= (signed) getSize())
                throw Exception("buffer offset out of range");
#endif
            return ptr.getOffset(offset);
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        cudadecl inline Pointer<const T> getBuffer() const
        {
            return ptr.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        cudadecl inline const AutoPointer<T[]>& getPointer() const
        {
            return ptr;
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
         * Copies an existing buffer's data.
         * @param ptr The pointer of buffer to be copied.
         */
        inline void copy(Pointer<const T> ptr)
        {
            memcpy(this->ptr, ptr, sizeof(T) * size);
        }
};

#endif
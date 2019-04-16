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
 * @tparam T The buffer type.
 * @since 0.1.1
 */
template <typename T>
class BaseBuffer
{
    protected:
        Pointer<T[]> ptr;       /// The pointer to the buffer being encapsulated.
        size_t size = 0;        /// The number of elements in buffer.

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
        ,   size {size}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(const Pointer<T[]>& ptr, size_t size)
        :   ptr {ptr}
        ,   size {size}
        {}

        /**
         * Acquires the ownership of a buffer pointer.
         * @param ptr The raw pointer instance to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(const RawPointer<T>& ptr, size_t size)
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
        __host__ __device__ inline T& operator[](ptrdiff_t offset) const
        {
#if defined(msa_compile_cython) && !defined(msa_compile_cuda)
            if(offset < 0 || static_cast<unsigned>(offset) > getSize())
                throw Exception("buffer offset out of range");
#endif
            return ptr.getOffset(offset);
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__ inline T *getBuffer() const
        {
            return ptr.get();
        }

        /**
         * Gives access to buffer's pointer.
         * @return The buffer's smart pointer.
         */
        __host__ __device__ inline const Pointer<T[]>& getPointer() const
        {
            return ptr;
        }

        /**
         * Gives access to an offset pointer.
         * @param offset The offset to apply to pointer.
         * @return The buffer's offset pointer.
         */
        inline const Pointer<T[]> getOffsetPointer(ptrdiff_t offset) const
        {
            return ptr.getOffsetPointer(offset);
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
 * Creates a general-purpose contiguous buffer.
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

        Buffer<T>& operator=(const Buffer<T>&) = default;
        Buffer<T>& operator=(Buffer<T>&&) = default;

    protected:
        /**
         * Copies an existing buffer's data.
         * @param ptr The pointer of buffer to be copied.
         */
        inline void copy(const T *ptr)
        {
            memcpy(this->getBuffer(), ptr, sizeof(T) * this->getSize());
        }
};

/**
 * Manages a slice of a buffer. The buffer must have already been initialized
 * and will have boundaries checked according to slice pointers.
 * @tparam T The buffer type.
 * @since 0.1.1
 */
template <typename T>
class BufferSlice : public BaseBuffer<T>
{
    protected:
        ptrdiff_t displ = 0;    /// The slice displacement in relation to original buffer.

    public:
        BufferSlice() = default;
        BufferSlice(const BufferSlice<T>&) = default;
        BufferSlice(BufferSlice<T>&&) = default;

        /**
         * Instantiates a new buffer slice.
         * @param target The target buffer to which the slice shall relate to.
         * @param displ The initial displacement of slice.
         * @param size The number of elements in the slice.
         */
        inline BufferSlice(const BaseBuffer<T>& target, ptrdiff_t displ = 0, size_t size = 0)
        :   BaseBuffer<T> {target.getOffsetPointer(displ), size}
        ,   displ {displ}
        {
#if defined(msa_compile_cython)
            if(size + displ < 0 || static_cast<unsigned>(size + displ) >= target.getSize())
                throw Exception("slice initialized out of range");
#endif
        }

        /**
         * Instantiates a slice by copying slice pointers into another buffer.
         * @param target The target buffer to which the slice shall relate to.
         * @param slice The slice data to be put into the new target.
         */
        inline BufferSlice(const BaseBuffer<T>& target, const BufferSlice<T>& slice)
        :   BaseBuffer<T> {target.getOffsetPointer(slice.getDispl()), slice.getSize()}
        ,   displ {slice.getDispl()}
        {
#if defined(msa_compile_cython)
            if(static_cast<unsigned>(slice.getSize() + slice.getDispl()) >= target.getSize())
                throw Exception("slice initialized out of range");
#endif
        }

        BufferSlice<T>& operator=(const BufferSlice<T>&) = default;
        BufferSlice<T>& operator=(BufferSlice<T>&&) = default;

        /**
         * Informs the displacement pointer in relation to original buffer.
         * @return The buffer's slice displacement.
         */
        __host__ __device__ inline ptrdiff_t getDispl() const
        {
            return displ;
        }
};

#endif
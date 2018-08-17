/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

#include <cstdint>
#include <cstring>
#include <vector>

/**
 * The base of a buffer class.
 * @since 0.1.alpha
 */
template<typename T>
class BaseBuffer
{
    protected:
        T *buffer = nullptr;    /// The buffer being encapsulated.
        uint32_t size = 0;      /// The number of buffer blocks.

    public:
        /**
         * Encapsulates a buffer pointer.
         * @param buffer The buffer pointer to encapsulate.
         * @param size The size of buffer to encapsulate.
         */
        inline explicit BaseBuffer(T *buffer, uint32_t size)
        :   buffer(buffer)
        ,   size(size) {}

        /**
         * Copies a buffer into a new instance.
         * @param buff The buffer to be copied.
         */
        inline explicit BaseBuffer(const BaseBuffer<T>& buffer)
        :   buffer(buffer.getBuffer())
        ,   size(buffer.getSize()) {}

        /**
         * Gives access to a specific location in buffer's data.
         * @param offset The requested buffer offset.
         * @return The buffer's position pointer.
         */
        inline const T& operator[](uint32_t offset) const
        {
            return this->buffer[offset];
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        inline T *getBuffer() const
        {
            return this->buffer;
        }

        /**
         * Informs the buffer's number of blocks.
         * @return The number of buffer blocks.
         */
        inline uint32_t getSize() const
        {
            return this->size;
        }

    protected:
        BaseBuffer() = default;
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
        /**
         * Constructs a new buffer from an already existing one.
         * @param buffer The buffer to be copied.
         * @param size The number of buffer blocks being copied.
         */
        inline Buffer(const T *buffer, uint32_t size)
        :   BaseBuffer<T>()
        {
            this->copy(buffer, size);
        }

        /**
         * Constructs a new buffer from an already existing instance.
         * @param buffer The instance from which data will be copied.
         */
        inline Buffer(const Buffer<T>& buffer)
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

        /**
         * Destroys the buffer created by this instance.
         */
        inline ~Buffer() noexcept
        {
            if(this->buffer != nullptr)
                delete[] this->buffer;
        }

    protected:
        Buffer() = default;

        /**
         * Copies an existing buffer's data.
         * @param buffer The buffer to be copied.
         * @param size The buffer's data blocks number.
         */
        inline void copy(const T *buffer, uint32_t size)
        {
            this->size = size;
            this->buffer = new T[size];
            std::memcpy(this->buffer, buffer, sizeof(T) * size);
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
        uint32_t displ = 0;    /// The slice displacement in relation to the buffer.

    public:
        /**
         * Constructs a new buffer slice.
         * @param target The target buffer to which the slice is related to.
         * @param displ The displacement of the slice.
         * @param size The number of blocks of the slice.
         */
        inline BufferSlice(const BaseBuffer<T>& target, uint32_t displ = 0, uint32_t size = 0)
        :   displ(displ)
        {
            this->buffer = target.getBuffer() + this->displ;
            this->size = size;
        }

        /**
         * Constructs a slice from an already existing instance into another buffer.
         * @param target The target buffer to which the slice is related to.
         * @param slice The slice data to be put into the new target.
         */
        inline BufferSlice(const BaseBuffer<T>& target, const BufferSlice<T>& slice)
        :   displ(slice.getDispl())
        {
            this->buffer = target.getBuffer() + this->displ;
            this->size = slice.getSize();
        }

        /**
         * Constructs a slice from an already existing instance.
         * @param slice The slice to be copied.
         */
        inline BufferSlice(const BufferSlice<T>& slice)
        :   displ(slice.getDispl())
        {
            this->buffer = slice.getBuffer();
            this->size = slice.getSize();
        }

        /**
         * Informs the displacement of data pointed by the slice.
         * @return The buffer's slice displacement.
         */
        inline uint32_t getDispl() const
        {
            return this->displ;
        }

    protected:
        BufferSlice() = default;
};

#endif

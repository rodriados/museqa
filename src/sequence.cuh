/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _SEQUENCE_CUH_
#define _SEQUENCE_CUH_

#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "device.cuh"

/**
 * Representation of a buffer pointer.
 * @since 0.1.alpha
 */
template<typename T>
class BufferPtr
{
    protected:
        T *buffer = nullptr;    /// The buffer being encapsulated.
        uint32_t length = 0;     /// The buffer's current number of data blocks.

    public:
        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__
        inline const T *operator&() const
        {
            return this->buffer;
        }

        /**
         * Gives access to a specific location in buffer's data.
         * @return The buffer's position pointer.
         */
        __host__ __device__
        inline const T& operator[](uint32_t offset) const
        {
            return this->buffer[offset];
        }

        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        __host__ __device__
        inline const T *getBuffer() const
        {
            return this->buffer;
        }

        /**
         * Informs the number of data blocks stored in buffer.
         * @return The buffer's number of blocks.
         */
        __host__ __device__
        inline uint32_t getLength() const
        {
            return this->length;
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
class Buffer : public BufferPtr<T>
{
    public:
        Buffer() = delete;

        /**
         * Constructs a new buffer from an already existing one.
         * @param buffer The buffer to be copied.
         * @param length The number of buffer blocks being copied.
         */
        inline Buffer(const T *buffer, uint32_t length)
        {
            this->copy(buffer, length);
        }

        /**
         * Constructs a new buffer from a vector.
         * @param vector The vector from which the buffer will be created.
         */
        inline Buffer(const std::vector<T>& vector)
        {
            this->copy(vector.data(), vector.size());
        }

        /**
         * Constructs a new buffer from an already existing instance.
         * @param buffer The instance from which data will be copied.
         */
        inline Buffer(const BufferPtr<T>& buffer)
        {
            this->copy(buffer.getBuffer(), buffer.getLength());
        }

        /**
         * Destroys the buffer created by this instance.
         */
        inline ~Buffer() noexcept
        {
            if(this->buffer != nullptr)
                delete[] this->buffer;
        }

    private:
        /**
         * Copies an existing buffer's data.
         * @param buffer The buffer to be copied.
         * @param length The buffer's data blocks number.
         */
        inline void copy(const T *buffer, uint32_t length)
        {
            this->length = length;
            this->buffer = new T[length];
            memcpy(this->buffer, buffer, sizeof(T) * length);
        }
};

/**
 * Represents a slice of a buffer.
 * @since 0.1.alpha
 */
template<typename T>
class BufferSlice : public BufferPtr<T>
{
    using BufferPtr<T>::BufferPtr;

    protected:
        uint32_t offset = 0;    /// The slice offset in relation to the buffer.

    public:
        BufferSlice() = delete;

        /**
         * Constructs a new buffer slice.
         * @param target The target buffer to which the slice is related to.
         * @param offset The offset of the slice.
         * @param length The number of blocks of the slice.
         */
        inline BufferSlice(const BufferPtr<T>& target, uint32_t offset = 0, uint32_t length = 0)
        {
            this->buffer = &target[offset];
            this->length = length;
            this->offset = offset;
        }

        /**
         * Constructs a slice from an already existing instance.
         * @param slice The slice to be copied.
         */
        inline BufferSlice(const BufferSlice<T>& slice)
        {
            this->buffer = slice.getBuffer();
            this->length = slice.getLength();
            this->offset = slice.getOffset();
        }

        /**
         * Informs the offset of data pointed by the slice.
         * @return The buffer's slice offset.
         */
        __host__ __device__
        inline int getOffset() const
        {
            return this->offset;
        }

};

/**
 * Creates an sequence. This sequence is a buffer an any modification to
 * it shall be implemented by inherited methods.
 * @since 0.1.alpha
 */
class Sequence : public Buffer<char>
{
    using Buffer::Buffer;

    public:
        /**
         * Instantiates a new sequence.
         * @param string The string containing this sequence's data.
         */
        inline Sequence(const std::string& string)
        :   Buffer<char>(string.c_str(), string.size())
        {}

    friend std::ostream& operator<<(std::ostream&, const BufferPtr<char>&);
};

/**
 * This function allows buffers to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to print.
 */
inline std::ostream& operator<<(std::ostream& os, const BufferPtr<char>& sequence)
{
    for(uint32_t i = 0; i < sequence.getLength(); ++i)
        os << sequence[i];

    return os;
}

#endif
/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _SEQUENCE_HPP_
#define _SEQUENCE_HPP_

#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

/**
 * Representation of a buffer pointer.
 * @since 0.1.alpha
 */
template<typename T>
class BufferPtr
{
    protected:
        T *buffer = nullptr;    /// The buffer being encapsulated.
        uint32_t length = 0;    /// The buffer's current length.

    public:
        /**
         * Gives access to buffer's data.
         * @return The buffer's internal pointer.
         */
        inline const T *operator&() const
        {
            return this->buffer;
        }

        /**
         * Gives access to a specific location in buffer's data.
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
        inline const T *getBuffer() const
        {
            return this->buffer;
        }

        /**
         * Informs the length of data stored in buffer.
         * @return The buffer's length.
         */
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
         * @param length The length of buffer being copied.
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
            delete[] this->buffer;
        }

    private:
        /**
         * Copies an existing buffer's data.
         * @param buffer The buffer to be copied.
         * @param length The buffer's length.
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
    protected:
        uint32_t offset = 0;    /// The slice offset in relation to the buffer.

    public:
        BufferSlice() = delete;

        /**
         * Constructs a new buffer slice.
         * @param target The target buffer to which the slice is related to.
         * @param offset The offset of the slice.
         * @param length The length of the slice.
         */
        inline BufferSlice(const BufferPtr<T>& target, uint32_t offset = 0, uint32_t length = 0)
        {
            this->offset = offset;
            this->length = length;
            this->buffer = &target[offset];
        }

        /**
         * Constructs a slice from an already existing instance.
         * @param slice The slice to be copied.
         */
        inline BufferSlice(const BufferSlice<T>& slice)
        {
            this->offset = slice.getOffset();
            this->length = slice.getLength();
            this->buffer = slice.getBuffer();
        }

        /**
         * Informs the offset of data pointed by the slice.
         * @return The buffer's slice offset.
         */
        inline uint32_t getOffset() const
        {
            return this->offset;
        }

};

/**
 * Creates an immutable sequence. This sequence is a buffer that should not be
 * changed after its instantiation.
 * @since 0.1.alpha
 */
class Sequence
{
    protected:
        Buffer<char> buffer;

    public:
        Sequence(const std::string&);
        Sequence(const BufferPtr<char>&);
        Sequence(const char *, uint32_t);

        virtual const Sequence& operator=(const Sequence&);
        virtual const Sequence& operator=(const BufferPtr<char>&);

        /**
         * Gives access to a specific location in buffer's data.
         * @return The buffer's position pointer.
         */
        inline const char& operator[](uint32_t offset) const
        {
            return this->buffer[offset];
        }

        /**
         * Informs the length of sequence.
         * @return The sequence's length.
         */
        inline uint32_t getLength() const
        {
            return this->buffer.getLength();
        }

    protected:
        Sequence() = default;

        /**
         * Gives access to the sequence's internal buffer's data.
         * @return The buffer's internal pointer.
         */
        inline const char *getBuffer() const
        {
            return this->buffer.getBuffer();
        }

    friend std::ostream& operator<<(std::ostream&, const Sequence&);
};

#endif
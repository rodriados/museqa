/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _SEQUENCE_HPP_
#define _SEQUENCE_HPP_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

/**
 * Creates a general-purpose buffer. This buffer should only be written via
 * inherited classes' instances.
 * @since 0.1.alpha
 */
template<typename T>
class Buffer
{
    protected:
        T *buffer = nullptr;    /// The buffer being encapsulated.
        uint32_t length = 0;    /// The buffer's current length.

    public:
        /**
         * Gives access to buffer's data.
         * @return Buffer's pointer.
         */
        inline T *operator&() const
        {
            return this->buffer;
        }

        /**
         * Gives access to a specific location in buffer's data.
         * @return Buffer's position pointer.
         */
        inline virtual T& operator[](uint32_t offset) const
        {
            return this->buffer[offset];
        }

        /**
         * Gives access to buffer's data.
         * @return Buffer's pointer.
         */
        inline T *getBuffer() const
        {
            return this->buffer;
        }

        /**
         * Informs the length of data stored in buffer.
         * @return Buffer's length.
         */
        inline virtual uint32_t getLength() const
        {
            return this->length;
        }

};

/**
 * Creates an immutable sequence. This sequence is a buffer that should not be
 * changed after its instantiation.
 * @since 0.1.alpha
 */
class Sequence : public Buffer<char>
{
    public:
        Sequence(const std::string&);
        Sequence(const Buffer<char>&);
        Sequence(const char *, uint32_t);
        virtual ~Sequence() noexcept;

        virtual const Sequence& operator=(const Buffer<char>&);

    protected:
        Sequence() = default;

        void copy(const char *, uint32_t);

    friend std::ostream& operator<<(std::ostream&, const Sequence&);
};

#endif
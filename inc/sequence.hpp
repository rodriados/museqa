/** 
 * Multiple Sequence Alignment sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _SEQUENCE_HPP_
#define _SEQUENCE_HPP_

#include <ostream>
#include <string>

#include "buffer.hpp"

/**
 * Creates an sequence. This sequence is a buffer an any modification to
 * it shall be implemented by inherited methods.
 * @since 0.1.alpha
 */
class Sequence : public Buffer<char>
{
    using Buffer<char>::Buffer;

    public:
        /**
         * Instantiates a new sequence.
         * @param string The string containing this sequence's data.
         */
        inline Sequence(const std::string& string)
        :   Buffer<char>(string.c_str(), string.size()) {}

        /**
         * Constructs a new sequence from an already existing instance.
         * @param buffer The instance from which data will be copied.
         */
        inline Sequence(const BaseBuffer<char>& buffer)
        :   Buffer<char>(buffer.getBuffer(), buffer.getSize()) {}

        /**
         * Informs the length of the sequence.
         * @return The sequence's length.
         */
        inline uint32_t getLength() const
        {
            return this->getSize();
        }
};

/**
 * This function allows buffers to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to print.
 */
inline std::ostream& operator<<(std::ostream& os, const BaseBuffer<char>& sequence)
{
    for(uint32_t i = 0; i < sequence.getSize(); ++i)
        os << sequence[i];

    return os;
}

#endif
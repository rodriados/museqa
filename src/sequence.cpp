/** 
 * Multiple Sequence Alignment sequence file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>

#include "sequence.hpp"

/**
 * Instantiates a new immutable sequence.
 * @param string The string containing this sequence's data.
 */
Sequence::Sequence(const std::string& string)
:   buffer(string.c_str(), string.size())
{}

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 */
Sequence::Sequence(const BufferPtr<char>& buffer)
:   buffer(buffer)
{}

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 * @param size The size of the buffer.
 */
Sequence::Sequence(const char *buffer, uint32_t size)
:   buffer(buffer, size)
{}

/**
 * Copy assignment operator.
 * @param sequence The sequence to be copied.
 */
const Sequence& Sequence::operator=(const Sequence& sequence)
{
    this->buffer = Buffer<char>(sequence.buffer);
    return *this;
}

/**
 * Copy assignment operator.
 * @param buffer The buffer of which data will be copied from.
 */
const Sequence& Sequence::operator=(const BufferPtr<char>& buffer)
{
    this->buffer = Buffer<char>(buffer);
    return *this;
}

/**
 * This function allows buffers to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to print.
 */
std::ostream& operator<< (std::ostream& os, const Sequence& sequence)
{
    for(uint32_t i = 0; i < sequence.getLength(); ++i)
        os << sequence[i];

    return os;
}
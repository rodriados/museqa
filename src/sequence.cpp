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
{
    this->copy(string.c_str(), string.size());
}

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 */
Sequence::Sequence(const Buffer<char>& buffer)
{
    this->copy(buffer.getBuffer(), buffer.getLength());
}

/**
 * Instantiates a new immutable sequence.
 * @param buffer The buffer of which data will be copied from.
 * @param size The size of the buffer.
 */
Sequence::Sequence(const char *buffer, uint32_t size)
{
    this->copy(buffer, size);
}

/**
 * Destroys a sequence instance.
 */
Sequence::~Sequence() noexcept
{
    delete[] this->buffer;
}

/**
 * Copy assignment operator.
 * @param buffer The buffer of which data will be copied from.
 */
const Sequence& Sequence::operator=(const Buffer<char>& buffer)
{
    delete[] this->buffer;

    this->copy(buffer.getBuffer(), buffer.getLength());
    return *this;
}

/**
 * Copies a buffer into the sequence.
 * @param buffer The buffer to be copied.
 * @param size The buffer's size.
 */
void Sequence::copy(const char *buffer, uint32_t size)
{
    this->length = size;
    this->buffer = new char [size];

    memcpy(this->buffer, buffer, sizeof(char) * size);
}

/**
 * This function allows buffers to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to print.
 */
std::ostream& operator<< (std::ostream& os, const Sequence& sequence)
{
    for(uint32_t i = 0; i < sequence.length; ++i)
        os << sequence.buffer[i];

    return os;
}
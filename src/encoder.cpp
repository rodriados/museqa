/**
 * Multiple Sequence Alignment sequence encoder file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <cctype>
#include <string>
#include <vector>
#include <ostream>

#include "buffer.hpp"
#include "encoder.hpp"
#include "exception.hpp"

#if defined(__GNUC__)
  #pragma GCC push_options
  #pragma GCC optimize ("unroll-loops")
#endif

/**
 * The translate table used for encoding a plain text sequences.
 * @since 0.1.1
 */
static constexpr uint8_t encodeTable[26] = {
    0x00, 0x14, 0x01, 0x06, 0x08, 0x0E, 0x03, 0x09, 0x0A, 0x15, 0x0C, 0x0B, 0x0D
,   0x05, 0x17, 0x0F, 0x07, 0x04, 0x10, 0x02, 0x17, 0x13, 0x11, 0x17, 0x12, 0x16
};

/**
 * The translate table used for decoding an encoded block.
 * @since 0.1.1
 */
static constexpr char decodeTable[26] = {
    'A', 'C', 'T', 'G', 'R', 'N', 'D', 'Q', 'E', 'H', 'I', 'L', 'K'
,   'M', 'F', 'P', 'S', 'W', 'Y', 'V', 'B', 'J', 'Z', 'X', '*', '-'
};

/**
 * The batch shifting indeces.
 * @since 0.1.1
 */
static constexpr uint8_t shift[6] = {1, 6, 11, 17, 22, 27};

/**
 * Encodes a single sequence character.
 * @param letter The character to encode.
 * @return The encoded character.
 */
uint8_t encoder::encode(uint8_t letter)
{
    letter = toupper(letter);

    return 'A' <= letter && letter <= 'Z'
        ? encodeTable[letter - 'A']
        : encoder::end;
}

/**
 * Compresses the string into a buffer of encoded blocks.
 * @param ptr The pointer to string to encode.
 * @param size The size of given string.
 * @return The buffer of enconded blocks.
 */
Buffer<encoder::EncodedBlock> encoder::encode(const char *ptr, size_t size)
{
    std::vector<encoder::EncodedBlock> vector;

    for(size_t i = 0, n = 0; n < size; ++i) {
        encoder::EncodedBlock block = 0;

        for(uint8_t j = 0; j < encoder::batchSize; ++j, ++n)
            block |= (n < size ? encoder::encode(ptr[n]) : encoder::end) << shift[j];

        vector.push_back(block | (n >= size));
    }

    return {vector};
}

/**
 * Decodes a single element into a character.
 * @param element The element to be decoded.
 * @return The decoded character.
 */
char encoder::decode(uint8_t element)
{
#ifdef msa_compile_cython
    if(element >= 26)
        throw Exception("cannot convert invalid sequence element:", static_cast<int>(element));
#endif
    return decodeTable[element];
}

/**
 * Decodes an encoded buffer to a human-friendly string.
 * @param buffer The target buffer to decode.
 * @return The decoded string.
 */
std::string encoder::decode(const BaseBuffer<encoder::EncodedBlock>& buffer)
{
    int size = buffer.getSize();
    std::string string;

    string.reserve(size * encoder::batchSize);

    for(int i = 0; i < size; ++i) 
        for(int j = 0; j < encoder::batchSize; ++j)
            string.append(1, decodeTable[access(buffer[i], j)]);

    return string;
}

/**
 * This function allows encoded buffers to be directly printed into an ostream instance.
 * @param os The output stream object.
 * @param buffer The sequence buffer to be printed.
 * @return The ostream instance for chaining.
 */
std::ostream& operator<<(std::ostream& os, const BaseBuffer<encoder::EncodedBlock>& buffer)
{
    os << encoder::decode(buffer);
    return os;
}

#if defined(__GNUC__)
  #pragma GCC pop_options
#endif

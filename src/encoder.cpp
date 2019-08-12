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
uint8_t encoder::encode(uint8_t letter) noexcept
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
    using namespace encoder;
    Buffer<EncodedBlock> buffer {(size / batchSize) + !!(size % batchSize)};

    for(size_t i = 0, n = 0; n < size; ++i) {
        EncodedBlock block = 0;

        for(uint8_t j = 0; j < batchSize; ++j, ++n)
            block |= (n < size ? encode(ptr[n]) : end) << shift[j];

        buffer[i] = block | (n >= size);
    }

    return buffer;
}

/**
 * Decodes a single element into a character.
 * @param element The element to be decoded.
 * @return The decoded character.
 */
char encoder::decode(uint8_t element)
{
    enforce(element < 26, "cannot convert invalid sequence element: %d", static_cast<int>(element));
    return decodeTable[element];
}

/**
 * Decodes an encoded buffer to a human-friendly string.
 * @param buffer The target buffer to decode.
 * @return The decoded string.
 */
std::string encoder::decode(const BaseBuffer<encoder::EncodedBlock>& buffer)
{
    std::string str;
    
    str.reserve(buffer.getSize() * encoder::batchSize);

    for(const auto& block : buffer)
        for(int j = 0; j < encoder::batchSize; ++j)
            str.append(1, decodeTable[access(block, j)]);

    return str;
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

/**
 * Multiple Sequence Alignment pairwise sequence file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <ostream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "sequence.hpp"
#include "pairwise/sequence.cuh"

/*
 * Declaring the translation tables for sequence symbols.
 */
static const uint8_t translate[26] = {
    0x00, 0x14, 0x01, 0x06, 0x08, 0x0E, 0x03, 0x09, 0x0A, 0x15, 0x0C, 0x0B, 0x0D,
    0x05, 0x17, 0x0F, 0x07, 0x04, 0x10, 0x02, 0x17, 0x13, 0x11, 0x17, 0x12, 0x16,
};

static const char untranslate[26] = {
    'A', 'C', 'T', 'G', 'R', 'N', 'D', 'Q', 'E', 'H', 'I', 'L', 'K',
    'M', 'F', 'P', 'S', 'W', 'Y', 'V', 'B', 'J', 'Z', 'X', '*', '-'
};

/*
 * Declaring constant values for compressing and decompressing the sequence.
 */
__constant__ uint8_t dshift[6] = {2, 7, 12, 17, 22, 27};
static const uint8_t hshift[6] = {2, 7, 12, 17, 22, 27};

namespace pw = pairwise;

/**
 * Initializes a new compressed sequence.
 * @param sequence The sequence to be copied.
 */
pw::Sequence::Sequence(const pw::Sequence& sequence)
{
    this->copy(sequence.getBuffer(), sequence.getLength());
}

/**
 * Initializes a new compressed sequence.
 * @param string The string from which the sequence will be created.
 */
pw::Sequence::Sequence(const std::string& string)
{
    this->compress(string.c_str(), string.size());
}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer from which the sequence will be created.
 */
pw::Sequence::Sequence(const Buffer<char>& buffer)
{
    this->compress(buffer.getBuffer(), buffer.getLength());
}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer to create the sequence from.
 * @param size The buffer's size.
 */
pw::Sequence::Sequence(const char *buffer, uint32_t size)
{
    this->compress(buffer, size);
}

/**
 * Initializes a new compressed sequence. An internal constructor possibility.
 * @param list Creates the sequence from a list of blocks.
 */
pw::Sequence::Sequence(const std::vector<uint32_t>& list)
{
    this->copy(list.data(), list.size());
}

/**
 * Initializes a new compressed sequence. An internal constructor possibility.
 * @param buffer Creates the sequence from a buffer of blocks.
 * @param size The buffer's size.
 */
pw::Sequence::Sequence(const uint32_t *buffer, uint32_t size)
{
    this->copy(buffer, size);
}

/**
 * Destroys the sequence and frees memory used by it.
 */
pw::Sequence::~Sequence() noexcept
{
    delete[] this->buffer;
}

/**
 * Copies the contents of a buffer into the sequence.
 * @param buffer The buffer to be copied.
 * @return This sequence instance.
 */
const pw::Sequence& pw::Sequence::operator=(const Buffer<char>& buffer)
{
    delete[] this->buffer;

    this->compress(buffer.getBuffer(), buffer.getLength());
    return *this;
}

/**
 * Copies the contents of a sequence instance into the sequence.
 * @param sequence The sequence to be copied.
 * @return This sequence instance.
 */
const pw::Sequence& pw::Sequence::operator=(const pw::Sequence& sequence)
{
    delete[] this->buffer;

    this->copy(sequence.getBuffer(), sequence.getLength());
    return *this;
}

/**
 * Uncompresses the sequence into a printable string.
 * @return The uncompressed and readable sequence.
 */
std::string pw::Sequence::uncompress() const
{
    std::string result;

    for(uint32_t i = 0, n = this->getLength(); i < n; ++i)
        for(uint8_t j = 0; j < 6; ++j)
            result += untranslate[(this->buffer[i] >> hshift[j]) & 0x1F];

    return result;
}

/**
 * Compresses the buffer into the sequence.
 * @param buffer The buffer to be compressed.
 * @param size The buffer's size.
 */
void pw::Sequence::compress(const char *buffer, uint32_t size)
{
    this->length = (size / 6) + bool(size % 6);
    this->buffer = new uint32_t [this->length];

    for(uint32_t i = 0, n = 0; i < this->length; ++i) {
        uint32_t block = 0;

        for(uint8_t j = 0; j < 6; ++j, ++n)
            if(n < size && 'A' <= buffer[n] && buffer[n] <= 'Z')
                block |= translate[buffer[n] - 'A'] << hshift[j];
            else
                block |= 0x18 << hshift[j];

        block |= (i != 0)
            ? (i == this->length - 1) ? 0x2 : 0x3
            : 0x1;

        this->buffer[i] = block;
    }
}

/**
 * Copies a buffer's contents into the sequence.
 * @param buffer The buffer to be copied.
 * @param size The buffer's size.
 */
void pw::Sequence::copy(const uint32_t *buffer, uint32_t size)
{
    this->length = size;
    this->buffer = new uint32_t [size];

    memcpy(this->buffer, buffer, sizeof(uint32_t) * size);
}

/**
 * Initializes a sequence list from a fasta file.
 * @param fasta The fasta file containing the sequences to be pushed.
 */
pw::SequenceList::SequenceList(const Fasta& fasta)
{
    for(uint16_t i = 0, n = fasta.getCount(); i < n; ++i)
        this->list.push_back(new pw::Sequence(fasta[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The list of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pw::SequenceList::SequenceList(const Buffer<char> *list, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pw::Sequence(list[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The vector of character buffers to be pushed.
 */
pw::SequenceList::SequenceList(const std::vector<Buffer<char>>& list)
{
    for(const Buffer<char>& buffer : list)
        this->list.push_back(new pw::Sequence(buffer));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 * @param count The number of selected sequence.
 */
pw::SequenceList::SequenceList(const pw::SequenceList& list, const uint16_t *selected, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pw::Sequence(list[selected[i]]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
pw::SequenceList::SequenceList(const pw::SequenceList& list, const std::vector<uint16_t>& selected)
{
    for(uint16_t index : selected)
        this->list.push_back(new pw::Sequence(list[index]));
}

/**
 * Destroys a sequence list instance.
 */
pw::SequenceList::~SequenceList() noexcept
{
    for(const pw::Sequence *sequence : this->list)
        delete sequence;
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @param count The number of selected sequences.
 * @return The new list of selected sequences.
 */
pw::SequenceList pw::SequenceList::select(const uint16_t *selected, uint16_t count) const
{
    return pw::SequenceList(*this, selected, count);
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @return The new list of selected sequences.
 */
pw::SequenceList pw::SequenceList::select(const std::vector<uint16_t>& selected) const
{
    return pw::SequenceList(*this, selected);
}

/**
 * Consolidates and compresses the sequence list.
 * @return The new compressed sequence list.
 */
pw::CompressedSequenceList pw::SequenceList::compress() const
{
    return CompressedSequenceList(*this);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 */
pw::CompressedSequenceList::CompressedSequenceList(const pw::SequenceList& list)
:   pw::Sequence(pw::CompressedSequenceList::merge(list, list.getCount()))
,   internal(new Buffer<uint32_t> [list.getCount()])
,   count(list.getCount())
{
    this->init(list, list.getCount());
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 * @param count The number of sequences in the list.
 */
pw::CompressedSequenceList::CompressedSequenceList(const pw::Sequence *list, uint16_t count)
:   pw::Sequence(pw::CompressedSequenceList::merge(list, count))
,   internal(new Buffer<uint32_t> [count])
,   count(count)
{
    this->init(list, count);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list An array of sequences of which data will be copied from.
 */
pw::CompressedSequenceList::CompressedSequenceList(const std::vector<pw::Sequence>& list)
:   pw::Sequence(pw::CompressedSequenceList::merge(list.data(), list.size()))
,   internal(new Buffer<uint32_t> [list.size()])
,   count(list.size())
{
    this->init(list.data(), list.size());
}

/**
 * Destroys the compressed sequence list.
 */
pw::CompressedSequenceList::~CompressedSequenceList() noexcept
{
    delete[] this->internal;
}

/**
 * This function allows sequences to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to be printed.
 */
std::ostream& operator<<(std::ostream& os, const pw::Sequence& sequence)
{
    os << sequence.uncompress();
    return os;
}

/**
 * This function allows an element to be retrieved out of a compressed sequence.
 * @param sequence The sequence to be accessed.
 * @param offset The requested offset.
 * @return The value stored in given offset.
 */
__host__ __device__ uint8_t operator%(const pw::Sequence& sequence, uint32_t offset)
{
    uint32_t block = offset / 6;
    uint8_t bindex = offset % 6;

#ifdef __CUDA_ARCH__
    return (sequence[block] >> dshift[bindex]) & 0x1F;
#else
    return (sequence[block] >> hshift[bindex]) & 0x1F;
#endif
}

/**
 * This function allows an element to be retrieved out of a sequence buffer.
 * @param buffer The buffer to be accessed.
 * @param offset The requested offset.
 * @return The value stored in given offset.
 */
__host__ __device__ uint8_t operator%(const Buffer<uint32_t>& buffer, uint32_t offset)
{
    uint32_t block = offset / 6;
    uint8_t bindex = offset % 6;

#ifdef __CUDA_ARCH__
    return (buffer[block] >> dshift[bindex]) & 0x1F;
#else
    return (buffer[block] >> hshift[bindex]) & 0x1F;
#endif
}

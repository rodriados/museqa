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
#include "device.cuh"
#include "sequence.cuh"
#include "pairwise/sequence.cuh"

/*
 * Declaring the translation tables for sequence symbols.
 */
static const char toChar[26] = {
    'A', 'C', 'T', 'G', 'R', 'N', 'D', 'Q', 'E', 'H', 'I', 'L', 'K',
    'M', 'F', 'P', 'S', 'W', 'Y', 'V', 'B', 'J', 'Z', 'X', '*', '-'
};

static const uint8_t toCompressed[26] = {
    0x00, 0x14, 0x01, 0x06, 0x08, 0x0E, 0x03, 0x09, 0x0A, 0x15, 0x0C, 0x0B, 0x0D,
    0x05, 0x17, 0x0F, 0x07, 0x04, 0x10, 0x02, 0x17, 0x13, 0x11, 0x17, 0x12, 0x16,
};

/*
 * Declaring constant values for compressing and decompressing the sequence.
 */
__constant__ uint8_t dShift[6] = {2, 7, 12, 17, 22, 27};
static const uint8_t hShift[6] = {2, 7, 12, 17, 22, 27};

/**
 * Initializes a new compressed sequence.
 * @param string The string from which the sequence will be created.
 */
pairwise::dSequence::dSequence(const std::string& string)
:   Buffer(compress(string.c_str(), string.size())) {}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer from which the sequence will be created.
 */
pairwise::dSequence::dSequence(const BaseBuffer<char>& buffer)
:   Buffer(compress(buffer.getBuffer(), buffer.getSize())) {}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer to create the sequence from.
 * @param size The buffer's size.
 */
pairwise::dSequence::dSequence(const char *buffer, uint32_t size)
:   Buffer(compress(buffer, size)) {}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param buffer Creates the sequence from a buffer of blocks.
 */
pairwise::dSequence::dSequence(const BaseBuffer<uint32_t>& buffer)
:   Buffer(buffer.getBuffer(), buffer.getSize()) {}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param list Creates the sequence from a list of blocks.
 */
pairwise::dSequence::dSequence(const std::vector<uint32_t>& list)
:   Buffer(list) {}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param buffer Creates the sequence from a buffer of blocks.
 * @param size The buffer's size.
 */
pairwise::dSequence::dSequence(const uint32_t *buffer, uint32_t size)
:   Buffer(buffer, size) {}

/**
 * Uncompresses the sequence into a printable string.
 * @return The uncompressed and readable sequence.
 */
std::string pairwise::dSequence::toString() const
{
    std::string result;

    for(uint32_t i = 0, n = this->getSize(); i < n; ++i)
        for(uint8_t j = 0; j < 6; ++j)
            result += toChar[(this->buffer[i] >> hShift[j]) & 0x1F];

    return result;
}

/**
 * Compresses the buffer into the sequence.
 * @param buffer The buffer to be compressed.
 * @param size The buffer's size.
 */
std::vector<uint32_t> pairwise::dSequence::compress(const char *buffer, uint32_t size)
{
    std::vector<uint32_t> blocks;

    for(uint32_t i = 0, n = 0; n < size; ++i) {
        uint32_t actual = 0;

        for(uint8_t j = 0; j < 6; ++j, ++n)
            actual |= (n < size && 'A' <= buffer[n] && buffer[n] <= 'Z')
                ? toCompressed[buffer[n] - 'A'] << hShift[j]
                : 0x18 << hShift[j];

        actual |= (i != 0)
            ? (n >= size) ? 0x2 : 0x3
            : 0x1;

        blocks.push_back(actual);
    }

    return blocks;
}

/**
 * Initializes a sequence list from a fasta file.
 * @param fasta The fasta file containing the sequences to be pushed.
 */
pairwise::SequenceList::SequenceList(const Fasta& fasta)
{
    for(uint16_t i = 0, n = fasta.getCount(); i < n; ++i)
        this->list.push_back(new pairwise::dSequence(fasta[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The vector of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<char> *list, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pairwise::dSequence(list[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The list of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<uint32_t> *list, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pairwise::dSequence(list[i]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 * @param count The number of selected sequence.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const uint16_t *selected, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pairwise::dSequence(list[selected[i]]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const std::vector<uint16_t>& selected)
{
    for(uint16_t index : selected)
        this->list.push_back(new pairwise::dSequence(list[index]));
}

/**
 * Destroys a sequence list instance.
 */
pairwise::SequenceList::~SequenceList() noexcept
{
    for(const pairwise::dSequence *sequence : this->list)
        delete sequence;
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @param count The number of selected sequences.
 * @return The new list of selected sequences.
 */
pairwise::SequenceList pairwise::SequenceList::select(const uint16_t *selected, uint16_t count) const
{
    return pairwise::SequenceList(*this, selected, count);
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @return The new list of selected sequences.
 */
pairwise::SequenceList pairwise::SequenceList::select(const std::vector<uint16_t>& selected) const
{
    return pairwise::SequenceList(*this, selected);
}

/**
 * Consolidates and compresses the sequence list.
 * @return The new compressed sequence list.
 */
pairwise::dSequenceList pairwise::SequenceList::compress() const
{
    return pairwise::dSequenceList(*this);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 */
pairwise::dSequenceList::dSequenceList(const pairwise::SequenceList& list)
:   pairwise::dSequence(merge(list, list.getCount()))
,   slice(new dSequenceSlice[list.getCount()])
,   count(list.getCount())
{
    this->init(list, list.getCount());
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 * @param count The number of sequences in the list.
 */
pairwise::dSequenceList::dSequenceList(const pairwise::dSequence *list, uint16_t count)
:   pairwise::dSequence(merge(list, count))
,   slice(new dSequenceSlice[count])
,   count(count)
{
    this->init(list, count);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list An array of sequences of which data will be copied from.
 */
pairwise::dSequenceList::dSequenceList(const std::vector<pairwise::dSequence>& list)
:   pairwise::dSequence(merge(list.data(), list.size()))
,   slice(new dSequenceSlice[list.size()])
,   count(list.size())
{
    this->init(list.data(), list.size());
}

/**
 * Destroys the compressed sequence list.
 */
pairwise::dSequenceList::~dSequenceList() noexcept
{
    delete[] this->slice;
}

/**
 * Sends the compressed sequence list to the device.
 * @return The object representing the sequence in the device.
 */
pairwise::hSequenceList pairwise::dSequenceList::toDevice() const
{
    return pairwise::hSequenceList(*this);
}

/**
 * Creates a compressed sequence list into the device global memory.
 * @param list The list to be sent to device.
 */
pairwise::hSequenceList::hSequenceList(const pairwise::dSequenceList& list)
{
    this->size = list.getSize();
    this->count = list.getCount();

    __cudacall(cudaMalloc(&this->buffer, sizeof(uint32_t) * this->size));
    __cudacall(cudaMalloc(&this->slice, sizeof(dSequenceSlice) * this->count));
    __cudacall(cudaMemcpy(this->buffer, list.getBuffer(), sizeof(uint32_t) * this->size, cudaMemcpyHostToDevice));
    __cudacall(cudaMemcpy(this->slice, list.slice, sizeof(dSequenceSlice) * this->count, cudaMemcpyHostToDevice));
}

/**
 * Destroys the compressed sequence list in the device.
 */
pairwise::hSequenceList::~hSequenceList() noexcept
{
    __cudacall(cudaFree(this->buffer));
    __cudacall(cudaFree(this->slice));
}

/**
 * Decodes an offset of a block.
 * @param block The target block.
 * @param offset The requested offset.
 * @return The buffer's position pointer.
 */
__cudadecl__ uint8_t pairwise::blockDecode(uint32_t block, uint8_t offset)
{
#ifdef __CUDA_ARCH__
    return (block >> dShift[offset]) & 0x1F;
#else
    return (block >> hShift[offset]) & 0x1F;
#endif
}

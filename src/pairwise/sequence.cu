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

/**
 * Initializes a new compressed sequence.
 * @param string The string from which the sequence will be created.
 */
pairwise::Sequence::Sequence(const std::string& string)
:   Buffer(compress(string.c_str(), string.size()))
{}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer from which the sequence will be created.
 */
pairwise::Sequence::Sequence(const BufferPtr<char>& buffer)
:   Buffer(compress(buffer.getBuffer(), buffer.getLength()))
{}

/**
 * Initializes a new compressed sequence.
 * @param buffer The buffer to create the sequence from.
 * @param size The buffer's size.
 */
pairwise::Sequence::Sequence(const char *buffer, uint32_t size)
:   Buffer(compress(buffer, size))
{}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param buffer Creates the sequence from a buffer of blocks.
 */
pairwise::Sequence::Sequence(const BufferPtr<uint32_t>& buffer)
:   Buffer(buffer)
{}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param list Creates the sequence from a list of blocks.
 */
pairwise::Sequence::Sequence(const std::vector<uint32_t>& list)
:   Buffer(list)
{}

/**
 * Initializes a new compressed sequence. An internal constructor option.
 * @param buffer Creates the sequence from a buffer of blocks.
 * @param size The buffer's size.
 */
pairwise::Sequence::Sequence(const uint32_t *buffer, uint32_t size)
:   Buffer(buffer, size)
{}

/**
 * Gives access to a specific location in buffer's data.
 * @return The buffer's position pointer.
 */
__host__ __device__ const uint8_t pairwise::Sequence::operator[](uint32_t offset) const
{
    register uint32_t block = offset / 6;
    register uint8_t bindex = offset % 6;

#ifdef __CUDA_ARCH__
    return (this->buffer[block] >> dshift[bindex]) & 0x1F;
#else
    return (this->buffer[block] >> hshift[bindex]) & 0x1F;
#endif
}

/**
 * Uncompresses the sequence into a printable string.
 * @return The uncompressed and readable sequence.
 */
std::string pairwise::Sequence::uncompress() const
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
static std::vector<uint32_t> pairwise::Sequence::compress(const char *buffer, uint32_t size)
{
    uint32_t length = (size / 6) + bool(size % 6);
    std::vector<uint32_t> vblocks;

    for(uint32_t n = 0; n < size; ++i) {
        uint32_t actual = 0;

        for(uint8_t i = 0; i < 6; ++i, ++n)
            actual |= (n < size && 'A' <= buffer[n] && buffer[n] <= 'Z')
                ? translate[buffer[n] - 'A'] << hshift[i]
                : 0x18 << hshift[i];

        actual |= (i != 0)
            ? (n >= size) ? 0x2 : 0x3
            : 0x1

        vblocks.push_back(actual);
    }

    return vblocks;
}

/**
 * Initializes a sequence list from a fasta file.
 * @param fasta The fasta file containing the sequences to be pushed.
 */
pairwise::SequenceList::SequenceList(const Fasta& fasta)
{
    for(uint16_t i = 0, n = fasta.getCount(); i < n; ++i)
        this->list.push_back(new pairwise::Sequence(fasta[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The list of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const Buffer<char> *list, uint16_t count)
{
    for(uint16_t i = 0; i < count; ++i)
        this->list.push_back(new pairwise::Sequence(list[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The vector of character buffers to be pushed.
 */
pairwise::SequenceList::SequenceList(const std::vector<Buffer<char>>& list)
{
    for(const Buffer<char>& buffer : list)
        this->list.push_back(new pairwise::Sequence(buffer));
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
        this->list.push_back(new pairwise::Sequence(list[selected[i]]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const std::vector<uint16_t>& selected)
{
    for(uint16_t index : selected)
        this->list.push_back(new pairwise::Sequence(list[index]));
}

/**
 * Destroys a sequence list instance.
 */
pairwise::SequenceList::~SequenceList() noexcept
{
    for(const pairwise::Sequence *sequence : this->list)
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
pairwise::CompressedSequenceList pairwise::SequenceList::compress() const
{
    return CompressedSequenceList(*this);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 */
pairwise::CompressedSequenceList::CompressedSequenceList(const pairwise::SequenceList& list)
:   pairwise::Sequence(merge(list, list.getCount()))
,   offset(new BufferOffset<uint32_t>[list.getCount()])
,   count(list.getCount())
{
    this->init(list, list.getCount());
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 * @param count The number of sequences in the list.
 */
pairwise::CompressedSequenceList::CompressedSequenceList(const pairwise::Sequence *list, uint16_t count)
:   pairwise::Sequence(merge(list, count))
,   offset(new BufferOffset<uint32_t>[count])
,   count(count)
{
    this->init(list, count);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list An array of sequences of which data will be copied from.
 */
pairwise::CompressedSequenceList::CompressedSequenceList(const std::vector<pairwise::Sequence>& list)
:   pairwise::Sequence(merge(list.data(), list.size()))
,   offset(new BufferOffset<uint32_t>[list.size()])
,   count(list.size())
{
    this->init(list.data(), list.size());
}

/**
 * Destroys the compressed sequence list.
 */
pairwise::CompressedSequenceList::~CompressedSequenceList() noexcept
{
    delete[] this->offset;
}

/**
 * Sends the compressed sequence list to the device.
 * @return The object representing the sequence in the device.
 */
pairwise::DeviceSequenceList pairwise::CompressedSequenceList::toDevice() const
{
    return pairwise::DeviceSequenceList(*this);
}

/**
 * Creates a compressed sequence list into the device global memory.
 * @param list The list to be sent to device.
 */
pairwise::DeviceSequenceList::DeviceSequenceList(const pairwise::CompressedSequenceList& list)
{
    this->count = list.getCount();
    this->length = list.getLength();

    __cudacall(cudaMalloc(&this->buffer, sizeof(uint32_t) * this->length));
    __cudacall(cudaMalloc(&this->internal, sizeof(pairwise::BlockBuffer) * this->count));
    __cudacall(cudaMemcpy(this->buffer, list.getBuffer(), sizeof(uint32_t) * this->length, cudaMemcpyHostToDevice));
    __cudacall(cudaMemcpy(this->internal, list.internal, sizeof(pairwise::BlockBuffer) * this->count, cudaMemcpyHostToDevice));
}

/**
 * Destroys the compressed sequence list in the device.
 */
pairwise::DeviceSequenceList::~DeviceSequenceList() noexcept
{
    __cudacall(cudaFree(this->buffer));
    __cudacall(cudaFree(this->internal));
}

/**
 * This function allows sequences to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to be printed.
 */
std::ostream& operator<<(std::ostream& os, const pairwise::Sequence& sequence)
{
    os << sequence.uncompress();
    return os;
}

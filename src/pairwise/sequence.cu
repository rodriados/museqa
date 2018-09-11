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
#include "buffer.hpp"
#include "device.cuh"
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
 * Uncompresses the sequence into a printable string.
 * @return The uncompressed and readable sequence.
 */
std::string pairwise::dSequence::toString() const
{
    std::string result;

    for(size_t i = 0, n = this->getSize(); i < n; ++i)
        for(uint8_t j = 0; j < 6; ++j)
            result += toChar[((*this)[i] >> hShift[j]) & 0x1F];

    return result;
}

/**
 * Compresses the buffer into the sequence.
 * @param buffer The buffer to be compressed.
 * @param size The buffer's size.
 */
std::vector<block_t> pairwise::dSequence::compress(const char *buffer, size_t size)
{
    std::vector<block_t> blocks;

    for(size_t i = 0, n = 0; n < size; ++i) {
        block_t actual = 0;

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
    for(size_t i = 0, n = fasta.getCount(); i < n; ++i)
        this->list.push_back(pairwise::dSequence(fasta[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The vector of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<char> *list, size_t count)
{
    for(size_t i = 0; i < count; ++i)
        this->list.push_back(pairwise::dSequence(list[i]));
}

/**
 * Initializes a new sequence list.
 * @param list The list of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<block_t> *list, size_t count)
{
    for(size_t i = 0; i < count; ++i)
        this->list.push_back(pairwise::dSequence(list[i]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 * @param count The number of selected sequence.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const ptrdiff_t *selected, size_t count)
{
    for(size_t i = 0; i < count; ++i)
        this->list.push_back(pairwise::dSequence(list[selected[i]]));
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const std::vector<ptrdiff_t>& selected)
{
    for(ptrdiff_t index : selected)
        this->list.push_back(pairwise::dSequence(list[index]));
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @param count The number of selected sequences.
 * @return The new list of selected sequences.
 */
pairwise::SequenceList pairwise::SequenceList::select(const ptrdiff_t *selected, size_t count) const
{
    return pairwise::SequenceList(*this, selected, count);
}

/**
 * Creates a new sequence list based on a selection of sequences.
 * @param selected The sequences' indices to be sent a new list.
 * @return The new list of selected sequences.
 */
pairwise::SequenceList pairwise::SequenceList::select(const std::vector<ptrdiff_t>& selected) const
{
    return pairwise::SequenceList(*this, selected);
}

/**
 * Consolidates and compresses the sequence list.
 * @return The new compressed sequence list.
 */
pairwise::CompressedList pairwise::SequenceList::compress() const
{
    return pairwise::CompressedList(*this);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 */
pairwise::CompressedList::CompressedList(const pairwise::SequenceList& list)
:   pairwise::dSequence(merge(list, list.getCount()))
,   slice(new dSequenceSlice[list.getCount()], std::default_delete<dSequenceSlice[]>())
,   count(list.getCount())
{
    this->init(list, list.getCount());
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 * @param count The number of sequences in the list.
 */
pairwise::CompressedList::CompressedList(const pairwise::dSequence *list, size_t count)
:   pairwise::dSequence(merge(list, count))
,   slice(new dSequenceSlice[count], std::default_delete<dSequenceSlice[]>())
,   count(count)
{
    this->init(list, count);
}

/**
 * Instantiates a new compressed sequence list.
 * @param list An array of sequences of which data will be copied from.
 */
pairwise::CompressedList::CompressedList(const std::vector<pairwise::dSequence>& list)
:   pairwise::dSequence(merge(list.data(), list.size()))
,   slice(new dSequenceSlice[list.size()], std::default_delete<dSequenceSlice[]>())
,   count(list.size())
{
    this->init(list.data(), list.size());
}

/**
 * Sends the compressed sequence list to the device.
 * @return The object representing the sequence in the device.
 */
pairwise::dSequenceList pairwise::CompressedList::toDevice() const
{
    return pairwise::dSequenceList(*this);
}

/**
 * Creates a compressed sequence list into the device global memory.
 * @param list The list to be sent to device.
 */
pairwise::dSequenceList::dSequenceList(const pairwise::CompressedList& list)
:   pairwise::CompressedList(list)
{
    block_t *buffer;
    dSequenceSlice *slice;
    std::vector<dSequenceSlice> adapted;

    cudacall(cudaMalloc(&buffer, sizeof(block_t) * this->size));
    cudacall(cudaMalloc(&slice, sizeof(dSequenceSlice) * this->count));

    for(size_t i = 0; i < this->count; ++i)
        adapted.push_back(dSequenceSlice(*this, list[i]));

    cudacall(cudaMemcpy(buffer, list.getBuffer(), sizeof(block_t) * this->size, cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(slice, adapted.data(), sizeof(dSequenceSlice) * this->count, cudaMemcpyHostToDevice));

    this->buffer = std::shared_ptr<block_t>(buffer, [](block_t *ptr){cudacall(cudaFree(ptr));});
    this->slice = std::shared_ptr<dSequenceSlice>(slice, [](dSequenceSlice *ptr){cudacall(cudaFree(ptr));});
}

/**
 * Decodes an offset of a block.
 * @param block The target block.
 * @param offset The requested offset.
 * @return The buffer's position pointer.
 */
cudadecl uint8_t pairwise::blockDecode(block_t block, uint8_t offset)
{
#ifdef __CUDA_ARCH__
    return (block >> dShift[offset]) & 0x1F;
#else
    return (block >> hShift[offset]) & 0x1F;
#endif
}

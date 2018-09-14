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
#include "pointer.hpp"
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
 * Initializes a sequence list from a fasta file.
 * @param fasta The fasta file containing the sequences to be pushed.
 */
pairwise::SequenceList::SequenceList(const Fasta& fasta)
{
    for(size_t i = 0, n = fasta.getCount(); i < n; ++i)
        this->list.push_back(fasta[i]);
}

/**
 * Initializes a new sequence list.
 * @param list The vector of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<char> *list, size_t count)
{
    for(size_t i = 0; i < count; ++i)
        this->list.push_back(list[i]);
}

/**
 * Initializes a new sequence list.
 * @param list The list of character buffers to be pushed.
 * @param count The number of elements in the list.
 */
pairwise::SequenceList::SequenceList(const BaseBuffer<block_t> *list, size_t count)
{
    for(size_t i = 0; i < count; ++i)
        this->list.push_back(list[i]);
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
        this->list.push_back(list[selected[i]]);
}

/**
 * Instantiates a new sequence list based on a subset of a list.
 * @param list A sequence list of which data will be copied from.
 * @param selected The selected sequences from the list.
 */
pairwise::SequenceList::SequenceList(const pairwise::SequenceList& list, const std::vector<ptrdiff_t>& selected)
{
    for(ptrdiff_t index : selected)
        this->list.push_back(list[index]);
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
:   pairwise::dSequence(merge(list.getRaw(), list.getCount()))
,   slice(new dSequenceSlice[list.getCount()])
,   count(list.getCount())
{
    this->init(list.getRaw(), list.getCount());
}

/**
 * Instantiates a new compressed sequence list.
 * @param list A list of sequences of which data will be copied from.
 * @param count The number of sequences in the list.
 */
pairwise::CompressedList::CompressedList(const pairwise::dSequence *list, size_t count)
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
pairwise::CompressedList::CompressedList(const std::vector<pairwise::dSequence>& list)
:   pairwise::dSequence(merge(list.data(), list.size()))
,   slice(new dSequenceSlice[list.size()])
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
 * Sets up the buffers responsible for keeping track of internal sequences.
 * @param list The list of original sequences being consolidated.
 * @param count The number of sequences to be compressed.
 */
void pairwise::CompressedList::init(const pairwise::dSequence *list, size_t count)
{
    for(size_t i = 0, off = 0; i < count; ++i) {
        this->slice[i] = {*this, off, list[i].getSize()};
        off += list[i].getSize();
    }
}

/**
 * Merges all sequences from the list into a single sequnces.
 * @param list The list of original sequences to be merged.
 * @param count The number of sequences to be merged.
 * @return The merged sequences.
 */
std::vector<block_t> pairwise::CompressedList::merge(const pairwise::dSequence *list, size_t count)
{
    std::vector<block_t> merged;

    for(size_t i = 0; i < count; ++i) {
        const block_t *ref = list[i].getBuffer();
        merged.insert(merged.end(), ref, ref + list[i].getSize());
    }

    return merged;
}

/**
 * Creates a compressed sequence list into the device global memory.
 * @param list The list to be sent to device.
 */
pairwise::dSequenceList::dSequenceList(const pairwise::CompressedList& list)
:   pairwise::CompressedList()
{
    block_t *buffer;
    dSequenceSlice *slices;
    std::vector<dSequenceSlice> adapt;

    this->size = list.getSize();
    this->count = list.getCount();

    cudacall(cudaMalloc(&buffer, sizeof(block_t) * this->size));
    cudacall(cudaMalloc(&slices, sizeof(dSequenceSlice) * this->count));

    this->buffer = SharedPointer<block_t[]>(buffer, pairwise::dSequenceList::deleteBuffer);
    this->slice = SharedPointer<dSequenceSlice[]>(slices, pairwise::dSequenceList::deleteSlices);

    for(size_t i = 0; i < this->count; ++i)
        adapt.push_back({*this, list[i]});

    cudacall(cudaMemcpy(buffer, list.getBuffer(), sizeof(block_t) * this->size, cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(slices, adapt.data(), sizeof(dSequenceSlice) * this->count, cudaMemcpyHostToDevice));
}

/**
 * Frees device memory from buffer data.
 * @param ptr The pointer to be freed.
 */
void pairwise::dSequenceList::deleteBuffer(block_t *ptr)
{
    cudacall(cudaFree(ptr));
}

/**
 * Frees device memory from sequence slices data.
 * @param ptr The pointer to be freed.
 */
void pairwise::dSequenceList::deleteSlices(dSequenceSlice *ptr)
{
    cudacall(cudaFree(ptr));
}

/**
 * Compresses the buffer into the sequence.
 * @param buffer The buffer to be compressed.
 * @param size The buffer's size.
 * @return The list of blocks of encoded sequence.
 */
std::vector<block_t> pairwise::encode(const char *buffer, size_t size)
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
 * Decodes an offset of a block.
 * @param block The target block.
 * @param offset The requested offset.
 * @return The buffer's position pointer.
 */
cudadecl uint8_t pairwise::decode(block_t block, uint8_t offset)
{
#ifdef __CUDA_ARCH__
    return (block >> dShift[offset]) & 0x1F;
#else
    return (block >> hShift[offset]) & 0x1F;
#endif
}

/**
 * This function allows sequences to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to be printed.
 * @return The ostream instance for chaining.
 */
std::ostream& operator<<(std::ostream& os, const pairwise::dSequence& sequence)
{
    for(size_t i = 0, n = sequence.getSize(); i < n; ++i)
        #pragma unroll
        for(uint8_t j = 0; j < 6; ++j)
            os << toChar[pairwise::decode(sequence.getBlock(i), j)];

    return os;
}

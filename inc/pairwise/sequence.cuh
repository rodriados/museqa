/**
 * Multiple Sequence Alignment pairwise sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef PW_SEQUENCE_CUH_INCLUDED
#define PW_SEQUENCE_CUH_INCLUDED

#pragma once

#include <ostream>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "device.cuh"
#include "sequence.cuh"

/**
 * Type alias to represent a sequence block.
 * @since 0.1.alpha
 */
typedef uint32_t block_t;

namespace pairwise
{
    /*
     * Declaring namespace helper functions.
     */
    cudadecl uint8_t blockDecode(block_t, uint8_t);

    /**
     * Represents a compressed sequence. The characters are encoded in
     * such a way that it saves one third of the space it would require.
     * @since 0.1.alpha
     */
    class dSequence : public Buffer<block_t>
    {
        public:
            dSequence() = default;
            dSequence(const dSequence&) = default;
            dSequence(dSequence&&) = default;

            /**
             * Initializes a new compressed sequence.
             * @param string The string from which the sequence will be created.
             */
            inline dSequence(const std::string& string)
            :   Buffer<block_t>(compress(string.c_str(), string.size())) {}

            /**
             * Initializes a new compressed sequence.
             * @param buffer The buffer from which the sequence will be created.
             */
            inline dSequence(const BaseBuffer<char>& buffer)
            :   Buffer<block_t>(compress(buffer.getBuffer(), buffer.getSize())) {}

            /**
             * Initializes a new compressed sequence.
             * @param buffer The buffer to create the sequence from.
             * @param size The buffer's size.
             */
            inline explicit dSequence(const char *buffer, size_t size)
            :   Buffer<block_t>(compress(buffer, size)) {}

            dSequence& operator=(const dSequence&) = default;
            dSequence& operator=(dSequence&&) = default;

            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            cudadecl inline uint8_t operator[](ptrdiff_t offset) const
            {
                return blockDecode(this->getBlock(offset / 6), offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            cudadecl inline block_t getBlock(ptrdiff_t id) const
            {
                return this->buffer.get()[id];
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            cudadecl inline size_t getLength() const
            {
                return this->getSize() * 6;
            }

            std::string toString() const;

        protected:
            /**
             * Initializes a new compressed sequence. An internal constructor option.
             * @param buffer Creates the sequence from a buffer of blocks.
             */
            inline dSequence(const BaseBuffer<block_t>& buffer)
            :   Buffer<block_t>(buffer) {}

            /**
             * Initializes a new compressed sequence. An internal constructor option.
             * @param list Creates the sequence from a list of blocks.
             */
            inline dSequence(const std::vector<block_t>& list)
            :   Buffer<block_t>(list) {}

            /**
             * Initializes a new compressed sequence. An internal constructor option.
             * @param buffer Creates the sequence from a buffer of blocks.
             * @param size The buffer's size.
             */
            inline dSequence(const block_t *buffer, size_t size)
            :   Buffer<block_t>(buffer, size) {}

        private:
            static std::vector<block_t> compress(const char *, size_t);

        friend class SequenceList;
    };

    /**
     * Represents a slice of a sequence.
     * @since 0.1.alpha
     */
    class dSequenceSlice : public BufferSlice<block_t>
    {
        public:
            dSequenceSlice() = default;
            dSequenceSlice(const dSequenceSlice&) = default;
            dSequenceSlice(dSequenceSlice&&) = default;
            
            using BufferSlice<block_t>::BufferSlice;

            dSequenceSlice& operator=(const dSequenceSlice&) = default;
            dSequenceSlice& operator=(dSequenceSlice&&) = default;

            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            cudadecl inline uint8_t operator[](ptrdiff_t offset) const
            {
                return blockDecode(this->getBlock(offset / 6), offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            cudadecl inline block_t getBlock(ptrdiff_t id) const
            {
                return this->buffer.get()[id];
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            cudadecl inline size_t getLength() const
            {
                return this->getSize() * 6;
            }

        friend class CompressedList;
    };

    /**
     * Creates a sequence list. This sequence list is responsible for keeping
     * track of sequences within its scope. Once a sequence is put into the
     * list, it cannot leave.
     * @since 0.1.alpha
     */
    class SequenceList
    {
        private:
            std::vector<dSequence> list;

        public:
            SequenceList() = default;
            SequenceList(const SequenceList&) = default;
            SequenceList(SequenceList&&) = default;

            SequenceList(const Fasta&);
            SequenceList(const BaseBuffer<char> *, size_t);
            SequenceList(const BaseBuffer<block_t> *, size_t);
            SequenceList(const SequenceList&, const ptrdiff_t *, size_t);
            SequenceList(const SequenceList&, const std::vector<ptrdiff_t>&);

            SequenceList& operator=(const SequenceList&) = delete;
            SequenceList& operator=(SequenceList&&) = default;

            /**
             * Gives access to a specific sequence of the list.
             * @param offset The offset of requested sequence.
             * @return The requested sequence.
             */
            inline const dSequence& operator[](ptrdiff_t offset) const
            {
                return this->list.at(offset);
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            inline size_t getCount() const
            {
                return this->list.size();
            }

            SequenceList select(const ptrdiff_t *, size_t) const;
            SequenceList select(const std::vector<ptrdiff_t>&) const;
            class CompressedList compress() const;
    };

    /**
     * Creates a compressed sequence list. This immutable sequence list keeps
     * all of its sequences together in memory. This is useful not to worry about
     * moving these sequences separately.
     * @since 0.1.alpha
     */
    class CompressedList : public dSequence
    {
        protected:
            size_t count = 0;                       /// The number of slices in list.
            std::shared_ptr<dSequenceSlice> slice;  /// The list of slices.

        public:
            CompressedList() = default;
            CompressedList(const CompressedList&) = default;
            CompressedList(CompressedList&&) = default;

            CompressedList(const SequenceList&);
            CompressedList(const dSequence *, size_t);
            CompressedList(const std::vector<dSequence>&);

            CompressedList& operator=(const CompressedList&) = default;
            CompressedList& operator=(CompressedList&&) = default;

            /**
             * Gives access to a specific sequence buffer offset of the list.
             * @param offset The requested sequence offset.
             * @return The requested sequence buffer.
             */
            cudadecl inline const dSequenceSlice& operator[](ptrdiff_t offset) const
            {
                return this->slice.get()[offset];
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            cudadecl inline size_t getCount() const
            {
                return this->count;
            }

            class dSequenceList toDevice() const;

        protected:
            /**
             * Sets up the buffers responsible for keeping track of internal sequences.
             * @param list The list of original sequences being consolidated.
             * @param count The number of sequences to be compressed.
             */
            template<typename T>
            void init(const T& list, size_t count)
            {
                for(size_t i = 0, off = 0; i < count; ++i) {
                    this->slice.get()[i] = dSequenceSlice(*this, off, list[i].getSize());
                    off += list[i].getSize();
                }
            }

            /**
             * Merges all sequences from the list into a single sequnces.
             * @param list The list of original sequences to be merged.
             * @param count The number of sequences to be merged.
             * @return The merged sequences.
             */
            template<typename T>
            static std::vector<block_t> merge(const T& list, size_t count)
            {
                std::vector<block_t> merged;

                for(size_t i = 0; i < count; ++i) {
                    const block_t *ref = list[i].getBuffer();
                    merged.insert(merged.end(), ref, ref + list[i].getSize());
                }

                return merged;
            }

        friend class dSequenceList;
    };

    /**
     * Sends a list of sequences to the device. This list of sequences are immutable
     * and can only be read from the device.
     * @since 0.1.alpha
     */
    class dSequenceList : public CompressedList
    {
        public:
            dSequenceList(const CompressedList&);
    };
};

/**
 * This function allows sequences to be directly printed into a ostream instance.
 * @param os The output stream object.
 * @param sequence The sequence to be printed.
 */
inline std::ostream& operator<<(std::ostream& os, const pairwise::dSequence& sequence)
{
    os << sequence.toString();
    return os;
}

#endif
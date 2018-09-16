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
#include <string>
#include <vector>

#include "fasta.hpp"
#include "buffer.hpp"
#include "device.cuh"
#include "pointer.hpp"

/**
 * Type alias to represent a sequence block.
 * @since 0.1.alpha
 */
using Block = uint32_t;

namespace pairwise
{
    /*
     * Declaring namespace helper functions.
     */
    extern std::vector<Block> encode(const char *, size_t);
    extern cudadecl uint8_t decode(Block, uint8_t);

    /**
     * Represents a compressed sequence. The characters are encoded in
     * such a way that it saves one third of the space it would require.
     * @since 0.1.alpha
     */
    class dSequence : public Buffer<Block>
    {
        public:
            dSequence() = default;
            dSequence(const dSequence&) = default;
            dSequence(dSequence&&) = default;

            /**
             * Initializes a new compressed sequence.
             * @param buffer Creates the sequence from a buffer of blocks.
             */
            inline dSequence(const BaseBuffer<Block>& buffer)
            :   Buffer<Block>(buffer) {}

            /**
             * Initializes a new compressed sequence.
             * @param buffer The buffer from which the sequence will be created.
             */
            inline dSequence(const BaseBuffer<char>& buffer)
            :   Buffer<Block>(encode(buffer.getBuffer(), buffer.getSize())) {}

            /**
             * Initializes a new compressed sequence.
             * @param string The string from which the sequence will be created.
             */
            inline dSequence(const std::string& string)
            :   Buffer<Block>(encode(string.c_str(), string.size())) {}

            /**
             * Initializes a new compressed sequence.
             * @param buffer The buffer to create the sequence from.
             * @param size The buffer's size.
             */
            inline explicit dSequence(const char *buffer, size_t size)
            :   Buffer<Block>(encode(buffer, size)) {}

            dSequence& operator=(const dSequence&) = default;
            dSequence& operator=(dSequence&&) = default;

            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            cudadecl inline uint8_t operator[](ptrdiff_t offset) const
            {
                return decode(this->getBlock(offset / 6), offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            cudadecl inline Block getBlock(ptrdiff_t id) const
            {
                return this->buffer.getOffset(id);
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            cudadecl inline size_t getLength() const
            {
                return this->getSize() * 6;
            }

        protected:
            /**
             * Initializes a new compressed sequence. An internal constructor option.
             * @param list Creates the sequence from a list of blocks.
             */
            inline dSequence(const std::vector<Block>& list)
            :   Buffer<Block>(list) {}

            /**
             * Initializes a new compressed sequence. An internal constructor option.
             * @param buffer Creates the sequence from a buffer of blocks.
             * @param size The buffer's size.
             */
            inline dSequence(const Block *buffer, size_t size)
            :   Buffer<Block>(buffer, size) {}

        friend class SequenceList;
    };

    /**
     * Represents a slice of a sequence.
     * @since 0.1.alpha
     */
    class dSequenceSlice : public BufferSlice<Block>
    {
        public:
            dSequenceSlice() = default;
            dSequenceSlice(const dSequenceSlice&) = default;
            dSequenceSlice(dSequenceSlice&&) = default;
            
            using BufferSlice<Block>::BufferSlice;

            dSequenceSlice& operator=(const dSequenceSlice&) = default;
            dSequenceSlice& operator=(dSequenceSlice&&) = default;

            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            cudadecl inline uint8_t operator[](ptrdiff_t offset) const
            {
                return decode(this->getBlock(offset / 6), offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            cudadecl inline Block getBlock(ptrdiff_t id) const
            {
                return this->buffer.getOffset(id);
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            cudadecl inline size_t getLength() const
            {
                return this->getSize() * 6;
            }
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
            SequenceList(const BaseBuffer<Block> *, size_t);

            SequenceList(const SequenceList&, const ptrdiff_t *, size_t);
            SequenceList(const SequenceList&, const std::vector<ptrdiff_t>&);

            SequenceList& operator=(const SequenceList&) = default;
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
             * Gives access to the raw list pointer.
             * @return The raw sequence list pointer.
             */
            inline const dSequence *getRaw() const
            {
                return this->list.data();
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
            uint16_t count = 0;                     /// The number of slices in list.
            SharedPointer<dSequenceSlice[]> slice;  /// The list of slices.

        public:
            CompressedList() = default;
            CompressedList(const CompressedList&) = default;
            CompressedList(CompressedList&&) = default;

            CompressedList(const SequenceList&);
            CompressedList(const dSequence *, uint16_t);
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
                return this->slice.getOffset(offset);
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            cudadecl inline uint16_t getCount() const
            {
                return this->count;
            }

            class dSequenceList toDevice() const;

        protected:
            void init(const dSequence *, uint16_t);
            static std::vector<Block> merge(const dSequence *, uint16_t);

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

/*
 * Declaring global function.
 */
extern std::ostream& operator<<(std::ostream&, const pairwise::dSequence&);

#endif
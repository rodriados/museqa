/**
 * Multiple Sequence Alignment pairwise sequence header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#ifndef _PW_SEQUENCE_CUH_
#define _PW_SEQUENCE_CUH_

#include <ostream>
#include <cstdint>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "device.cuh"
#include "sequence.cuh"

namespace pairwise
{
    /*
     * Declaring namespace helper functions.
     */
    __cudadecl__ uint8_t blockDecode(uint32_t, uint8_t);

    /**
     * Represents a compressed sequence. The characters are encoded in
     * such a way that it saves one third of the space it would require.
     * @since 0.1.alpha
     */
    class dSequence : public Buffer<uint32_t>
    {
        public:
            dSequence(const std::string&);
            dSequence(const BaseBuffer<char>&);
            dSequence(const char *, uint32_t);

            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            __cudadecl__ inline uint8_t operator[](uint32_t offset) const
            {
                return blockDecode(this->buffer[offset / 6], offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            __cudadecl__ inline uint32_t getBlock(uint32_t id) const
            {
                return this->buffer[id];
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            __cudadecl__ inline uint32_t getLength() const
            {
                return this->getSize() * 6;
            }

            std::string toString() const;

        protected:
            dSequence() = default;
            dSequence(const BaseBuffer<uint32_t>&);
            dSequence(const std::vector<uint32_t>&);
            dSequence(const uint32_t *, uint32_t);

        private:
            static std::vector<uint32_t> compress(const char *, uint32_t);

        friend class SequenceList;
    };

    /**
     * Represents a slice of a sequence.
     * @since 0.1.alpha
     */
    class dSequenceSlice : public BufferSlice<uint32_t>
    {
        using BufferSlice<uint32_t>::BufferSlice;

        public:
            /**
             * Decodes the character at the given offset.
             * @param offset The requested offset.
             * @return The character in the specified offset.
             */
            __cudadecl__ inline uint8_t operator[](uint32_t offset) const
            {
                return blockDecode(this->buffer[offset / 6], offset % 6);
            }

            /**
             * Gives access to a encoded character block of the sequence.
             * @param id The index of the requested block.
             * @return The requested block.
             */
            __cudadecl__ inline uint32_t getBlock(uint32_t id) const
            {
                return this->buffer[id];
            }

            /**
             * Informs the length of the sequence.
             * @return The sequence's length.
             */
            __cudadecl__ inline uint32_t getLength() const
            {
                return this->getSize() * 6;
            }

        friend class dSequenceList;
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
            std::vector<dSequence*> list;

        public:
            SequenceList() = delete;
            SequenceList(const Fasta&);
            SequenceList(const BaseBuffer<char> *, uint16_t);
            SequenceList(const BaseBuffer<uint32_t> *, uint16_t);

            SequenceList(const SequenceList&, const uint16_t *, uint16_t);
            SequenceList(const SequenceList&, const std::vector<uint16_t>&);

            ~SequenceList() noexcept;

            /**
             * Gives access to a specific sequence of the list.
             * @param offset The offset of requested sequence.
             * @return The requested sequence.
             */
            inline const dSequence& operator[](uint16_t offset) const
            {
                return *(this->list.at(offset));
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            inline uint16_t getCount() const
            {
                return this->list.size();
            }

            SequenceList select(const uint16_t *, uint16_t) const;
            SequenceList select(const std::vector<uint16_t>&) const;
            class dSequenceList compress() const;
    };

    /**
     * Creates a compressed sequence list. This immutable sequence list keeps
     * all of its sequences together in memory. This is useful not to worry about
     * moving these sequences separately.
     * @since 0.1.alpha
     */
    class dSequenceList : public dSequence
    {
        protected:
            dSequenceSlice *slice = nullptr;
            uint16_t count = 0;

        public:
            dSequenceList() = default;
            dSequenceList(const SequenceList&);
            dSequenceList(const dSequence *, uint16_t);
            dSequenceList(const std::vector<dSequence>&);

            ~dSequenceList() noexcept;

            /**
             * Gives access to a specific sequence buffer offset of the list.
             * @param offset The requested sequence offset.
             * @return The requested sequence buffer.
             */
            __cudadecl__ inline const dSequenceSlice& operator[](uint16_t offset) const
            {
                return this->slice[offset];
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            __cudadecl__ inline uint16_t getCount() const
            {
                return this->count;
            }

            class hSequenceList toDevice() const;

        protected:
            /**
             * Sets up the buffers responsible for keeping track of internal sequences.
             * @param list The list of original sequences being consolidated.
             * @param count The number of sequences to be compressed.
             */
            template<typename T>
            void init(const T& list, uint16_t count)
            {
                for(uint32_t i = 0, off = 0; i < count; ++i) {
                    this->slice[i].displ = off;
                    this->slice[i].buffer = &this->buffer[off];
                    off += this->slice[i].size = list[i].getSize();
                }
            }

            /**
             * Merges all sequences from the list into a single sequnces.
             * @param list The list of original sequences to be merged.
             * @param count The number of sequences to be merged.
             * @return The merged sequences.
             */
            template<typename T>
            static std::vector<uint32_t> merge(const T& list, uint16_t count)
            {
                std::vector<uint32_t> merged;

                for(uint16_t i = 0; i < count; ++i) {
                    const uint32_t *ref = list[i].getBuffer();
                    merged.insert(merged.end(), ref, ref + list[i].getSize());
                }

                return merged;
            }

        friend class hSequenceList;
    };

    /**
     * Sends a list of sequences to the device. This list of sequences are immutable
     * and can only be read from the device.
     * @since 0.1.alpha
     */
    class hSequenceList : public dSequenceList
    {
        public:
            hSequenceList(const dSequenceList&);
            ~hSequenceList() noexcept;
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
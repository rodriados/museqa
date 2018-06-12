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
#include "sequence.hpp"

namespace pairwise
{
    /**
     * Represents a buffer of blocks to be used as base for sequences
     * available for usage in the device.
     * @since 0.1.alpha
     */
    class BlockBuffer : public Buffer<uint32_t>
    {
        public:
            /**
             * Constructs a new buffer instance.
             * @param buffer The buffer's pointer.
             * @param length The buffer's length.
             */
            BlockBuffer(uint32_t *buffer, uint32_t length)
            {
                this->buffer = buffer;
                this->length = length;
            }

            BlockBuffer() = default;

            /**
             * Gives access to buffer's data.
             * @return Buffer's pointer.
             */
            __host__ __device__
            inline uint32_t *operator&() const
            {
                return this->buffer;
            }

            /**
             * Gives access to a block in the buffer.
             * @param offset The requested block offset.
             * @return The requested block.
             */
            __host__ __device__
            inline uint32_t& operator[](uint32_t offset) const
            {
                return this->buffer[offset];
            }

            /**
             * Gives access to buffer's data.
             * @return Buffer's pointer.
             */
            __host__ __device__
            inline uint32_t *getBuffer() const
            {
                return this->buffer;
            }

            /**
             * Informs the length of data stored in buffer.
             * @return Buffer's length.
             */
            __host__ __device__
            inline uint32_t getLength() const
            {
                return this->length;
            }            
    };

    /**
     * Represents a compressed sequence. The characters are encoded in
     * such a way that it saves one third of the space it would require.
     * @since 0.1.alpha
     */
    class Sequence : public BlockBuffer
    {
        protected:
            using Buffer<uint32_t>::buffer;
            using Buffer<uint32_t>::length;

        public:
            Sequence(const Sequence&);
            Sequence(const std::string&);
            Sequence(const Buffer<char>&);
            Sequence(const char *, uint32_t);
            ~Sequence() noexcept;

            const Sequence& operator=(const Buffer<char>&);
            const Sequence& operator=(const Sequence&);

            std::string uncompress() const;
            
        protected:
            Sequence() = default;
            Sequence(const std::vector<uint32_t>&);
            Sequence(const uint32_t *, uint32_t);

            void copy(const uint32_t *, uint32_t);

        private:
            void compress(const char *, uint32_t);

        friend class CompressedSequenceList;
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
            std::vector<Sequence *> list;

        public:
            SequenceList() = default;
            SequenceList(const Fasta&);
            SequenceList(const Buffer<char> *, uint16_t);
            SequenceList(const std::vector<Buffer<char>>&);

            SequenceList(const SequenceList&, const uint16_t *, uint16_t);
            SequenceList(const SequenceList&, const std::vector<uint16_t>&);

            ~SequenceList() noexcept;

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            inline uint16_t getCount() const
            {
                return this->list.size();
            }

            /**
             * Gives access to a specific sequence of the list.
             * @return The requested sequence.
             */
            inline const Sequence& operator[](uint16_t offset) const
            {
                return *(this->list.at(offset));
            }

            SequenceList select(const uint16_t *, uint16_t) const;
            SequenceList select(const std::vector<uint16_t>&) const;
            class CompressedSequenceList compress() const;
    };

    /**
     * Creates a compressed sequence list. This immutable sequence list keeps
     * all of its sequences together in memory. This is useful not to worry about
     * moving these sequences separately.
     * @since 0.1.alpha
     */
    class CompressedSequenceList : public Sequence
    {
        protected:
            BlockBuffer *internal = nullptr;
            uint16_t count = 0;

        public:
            CompressedSequenceList() = default;
            CompressedSequenceList(const SequenceList&);
            CompressedSequenceList(const Sequence *, uint16_t);
            CompressedSequenceList(const std::vector<Sequence>&);

            ~CompressedSequenceList() noexcept;

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            __host__ __device__
            inline uint16_t getCount() const
            {
                return this->count;
            }

            /**
             * Gives access to a specific sequence buffer offset of the list.
             * @return The requested sequence buffer.
             */
            __host__ __device__
            inline const BlockBuffer operator[](uint16_t offset) const
            {
                return BlockBuffer {
                    this->buffer + (intptr_t)this->internal[offset].buffer
                ,   this->internal[offset].length
                };
            }

            class DeviceSequenceList toDevice() const;

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
                    this->internal[i].buffer = (uint32_t *)off;
                    off += this->internal[i].length = list[i].getLength();
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
                    uint32_t *ref = list[i].getBuffer();
                    merged.insert(merged.end(), ref, ref + list[i].getLength());
                }

                return merged;
            }

        friend class DeviceSequenceList;
    };

    /**
     * Sends a list of sequences to the device. This list of sequences are immutable
     * and can only be read from the device.
     * @since 0.1.alpha
     */
    class DeviceSequenceList : public CompressedSequenceList
    {
        public:
            DeviceSequenceList(const CompressedSequenceList&);
            ~DeviceSequenceList() noexcept;
    };
};

/*
 * Declaring global functions, related to compressed sequences.
 */
extern std::ostream& operator<<(std::ostream&, const pairwise::Sequence&);
extern __host__ __device__ uint8_t operator%(const pairwise::Sequence&, uint32_t);
extern __host__ __device__ uint8_t operator%(const pairwise::BlockBuffer&, uint32_t);

#endif
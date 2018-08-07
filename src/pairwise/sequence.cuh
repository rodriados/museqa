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
    /**
     * Represents a compressed sequence. The characters are encoded in
     * such a way that it saves one third of the space it would require.
     * @since 0.1.alpha
     */
    class Sequence : public Buffer<uint32_t>
    {
        public:
            Sequence(const std::string&);
            Sequence(const BufferPtr<char>&);
            Sequence(const char *, uint32_t);

            __host__ __device__ uint8_t operator[](uint32_t) const;

            std::string uncompress() const;

        protected:
            Sequence() = default;
            Sequence(const BufferPtr<uint32_t>&);
            Sequence(const std::vector<uint32_t>&);
            Sequence(const uint32_t *, uint32_t);

        private:
            static std::vector<uint32_t> compress(const char *, uint32_t);

        friend class SequenceList;
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
            std::vector<Sequence*> list;

        public:
            SequenceList() = default;
            SequenceList(const Fasta&);
            SequenceList(const BufferPtr<char> *, uint16_t);
            SequenceList(const BufferPtr<uint32_t> *, uint16_t);

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

    class SequenceSlice : public BufferSlice<uint32_t>
    {
        using BufferSlice::BufferSlice;
        friend class CompressedSequenceList;
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
            SequenceSlice *slice = nullptr;
            uint16_t count = 0;

        public:
            CompressedSequenceList() = default;
            CompressedSequenceList(const SequenceList&);
            CompressedSequenceList(const Sequence *, uint16_t);
            CompressedSequenceList(const std::vector<Sequence>&);

            ~CompressedSequenceList() noexcept;

            /**
             * Gives access to a specific sequence buffer offset of the list.
             * @return The requested sequence buffer.
             */
            __host__ __device__
            inline const BufferPtr<uint32_t>& operator[](uint16_t offset) const
            {
                return this->slice[offset];
            }

            /**
             * Informs the number of sequences in the list.
             * @return The list's number of sequences.
             */
            __host__ __device__
            inline uint16_t getCount() const
            {
                return this->count;
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
                    this->slice[i].offset = off;
                    this->slice[i].buffer = &this->buffer[off];
                    off += this->slice[i].length = list[i].getLength();
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

#endif
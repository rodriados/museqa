/**
 * Multiple Sequence Alignment pairwise database header file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#pragma once

#ifndef PW_DATABASE_CUH_INCLUDED
#define PW_DATABASE_CUH_INCLUDED

#include <set>
#include <vector>
#include <cstdint>

#include "buffer.hpp"
#include "encoder.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "sequence.hpp"
#include "exception.hpp"

namespace pairwise
{
    /**
     * Manages a slice of a compressed database sequence.
     * @since 0.1.1
     */
    class SequenceSlice : public BufferSlice<encoder::EncodedBlock>
    {
        public:
            SequenceSlice() = default;
            SequenceSlice(const SequenceSlice&) = default;
            SequenceSlice(SequenceSlice&&) = default;

            using BufferSlice<encoder::EncodedBlock>::BufferSlice;

            SequenceSlice& operator=(const SequenceSlice&) = default;
            SequenceSlice& operator=(SequenceSlice&&) = default;

            /**
             * Retrieves the element at given offset.
             * @param offset The requested offset.
             * @return The element in the specified offset.
             */
            __host__ __device__ inline uint8_t operator[](ptrdiff_t offset) const
            {
                return encoder::access(*this, offset);
            }

            /**
             * Retrieves an encoded character block from sequence slice.
             * @param offset The index of the requested block.
             * @return The requested block.
             */
            __host__ __device__ inline encoder::EncodedBlock getBlock(ptrdiff_t offset) const
            {
                return BufferSlice<encoder::EncodedBlock>::operator[](offset);
            }

            /**
             * Informs the length of the sequence slice.
             * @return The sequence slice's length.
             */
            __host__ __device__ inline size_t getLength() const
            {
                return this->getSize() * encoder::batchSize;
            }

            /**
             * Transforms the sequence slice into a string.
             * @return The sequence slice representation as a string.
             */
            inline std::string toString() const
            {
                return encoder::decode(*this);
            }
    };

    /**
     * Stores a list of sequences from a database as a single contiguous sequence.
     * No description nor any other metadata will be stored alongside the sequences,
     * which will all solely be identified by their indexes. These indeces will be
     * kept the same as in the original database.
     * @since 0.1.1
     */
    class Database : public Sequence
    {
        protected:
            Buffer<SequenceSlice> slice;            /// The buffer of sequence slices.

        public:
            Database() = default;
            Database(const Database&) = default;
            Database(Database&&) = default;

            Database(const ::Database&);

            Database(const ::Database&, const std::set<ptrdiff_t>&);
            Database(const ::Database&, const std::vector<ptrdiff_t>&);

            Database& operator=(const Database&) = default;
            Database& operator=(Database&&) = default;

            /**
             * Gives access to a specific sequence in the database.
             * @param offset The offset of requested sequence.
             * @return The requested sequence.
             */
            __host__ __device__ inline const SequenceSlice& operator[](ptrdiff_t offset) const
            {
#if defined(msa_compile_cython) && !defined(msa_compile_cuda)
                if(static_cast<unsigned>(offset) >= getCount())
                    throw Exception("database index out of range");
#endif
                return slice[offset];
            }

            /**
             * Informs the number of sequences in database.
             * @return The database's number of sequences.
             */
            __host__ __device__ inline size_t getCount() const
            {
                return slice.getSize();
            }

            Database toDevice() const;

        private:
            /**
             * Initializes a new database from already compressed sequences and their respective slices.
             * @param blocks The encoded blocks of compressed database sequences.
             * @param slices The sequence slices upon the compressed sequences.
             */
            inline Database(const BaseBuffer<encoder::EncodedBlock>& blocks, const BaseBuffer<SequenceSlice>& slices)
            {
                slice.BaseBuffer<SequenceSlice>::operator=(slices);
                BaseBuffer<encoder::EncodedBlock>::operator=(blocks);
            }

            void init(const ::Database&);
            static std::vector<encoder::EncodedBlock> merge(const ::Database&);
    };
};

#endif
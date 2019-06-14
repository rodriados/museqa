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
     * Stores a list of sequences from a database as a single contiguous sequence.
     * No description nor any other metadata will be stored alongside the sequences,
     * which will all solely be identified by their indexes. These indeces will be
     * kept the same as in the original database.
     * @since 0.1.1
     */
    class Database : public Sequence
    {
        protected:
            Buffer<SequenceView> view;          /// The buffer of sequence views.

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
            __host__ __device__ inline const SequenceView& operator[](ptrdiff_t offset) const
            {
                enforce(offset < 0 || unsigned(offset) >= getCount(), "database index out of range");
                return view[offset];
            }

            /**
             * Informs the number of sequences in database.
             * @return The database's number of sequences.
             */
            __host__ __device__ inline size_t getCount() const
            {
                return view.getSize();
            }

            Database toDevice() const;

        private:
            /**
             * Initializes a new database from already compressed sequences and their respective views.
             * @param blocks The encoded blocks of compressed database sequences.
             * @param views The sequence views upon the compressed sequences.
             */
            inline Database(const BaseBuffer<encoder::EncodedBlock>& blocks, const BaseBuffer<SequenceView>& views)
            {
                view.BaseBuffer<SequenceView>::operator=(views);
                BaseBuffer<encoder::EncodedBlock>::operator=(blocks);
            }

            void init(const ::Database&);
            static std::vector<encoder::EncodedBlock> merge(const ::Database&);
    };
};

#endif
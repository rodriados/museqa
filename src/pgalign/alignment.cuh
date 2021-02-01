/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the profile-aligner module's alignment data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include "buffer.hpp"
#include "database.hpp"

#include "pgalign/sequence.cuh"

namespace museqa
{
    namespace pgalign
    {
        /**
         * Represents the alignment between one or more sequences. The alignment's
         * sequences should be organized in such a way that sequences are grouped
         * together by the order they should be aligned according to the alignment's
         * guiding-tree. This is done so we can create the global alignment's instance
         * first, and then efficiently create sub-alignments and sequence subsets,
         * and align them individually and in parallel, without having to worry
         * about moving or managing their position in memory.
         * @since 0.1.1
         */
        class alignment
        {
            private:
                using sequence_type = sequence;             /// The alignment's sequence type.
                using buffer_type = buffer<sequence_type>;  /// The alignment's sequence buffer type.

            private:
                buffer_type m_buffer;                       /// The internal buffer of sequences.

            public:
                inline alignment() noexcept = default;
                inline alignment(const alignment&) noexcept = default;
                inline alignment(alignment&&) noexcept = default;

                /**
                 * Instantiates a new alignment from a buffer of sequences.
                 * @param buffer The list of sequences to create the alignment with.
                 */
                inline alignment(const buffer_type& buffer) noexcept
                :   m_buffer {buffer}
                {}

                inline alignment& operator=(const alignment&) = default;
                inline alignment& operator=(alignment&&) = default;

                /**
                 * Creates a sub-alignment by selecting a slice of the alignment.
                 * @param displ The slice displament in relation to the alignment.
                 * @param size The number of sequences on the new sub-alignment.
                 */
                inline alignment slice(ptrdiff_t displ, size_t size)
                {
                    return buffer_slice<sequence_type> {m_buffer, displ, size};
                }

            private:
                /**
                 * Creates a new alignment from a subset of sequences in a larger
                 * sequence. This allows the creation of sub-alignments from an
                 * already existing alignment instance.
                 * @param slice An alignment slice.
                 */
                inline alignment(buffer_slice<sequence_type>&& slice)
                :   m_buffer {std::move(slice)}
                {}
        };
    }
}

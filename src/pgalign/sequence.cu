/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the profile-aligner module's sequence data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#include <string>
#include <cstdint>

#include "buffer.hpp"
#include "format.hpp"
#include "encoder.hpp"

#include "pgalign/sequence.cuh"

namespace museqa
{
    /**
     * Fetches a character from a random offset on the sequence. As the sequence
     * is stored using the Yale format for sparse matrices, it is not ideal to heavily
     * rely on this method for accessing the sequence, as a binary search must be
     * performed every time a new random offset is requested.
     * @param offset The offset to be fetched from sequence.
     * @return The character found at the fetched offset.
     */
    __host__ __device__ encoder::unit pgalign::sequence::operator[](ptrdiff_t offset) const
    {
        const auto buffer_size = m_columns.size();
        const auto sequence_size = m_columns[buffer_size - 1];

        if(sequence_size <= offset) {
            return encoder::end;
        }

        ptrdiff_t left = 0, right = buffer_size - 1;
        ptrdiff_t middle = (offset * buffer_size) / sequence_size;
        auto current = m_columns[middle];

        while(current != offset && left <= right) {
            if(current < offset) { left  = middle + 1; }
            else                 { right = middle - 1; }

            middle = (left + right) / 2;
            current = m_columns[middle];
        }

        return current == offset
            ? underlying_type::operator[](middle)
            : encoder::gap;
    }

    /**
     * Decodes a gapped sequence into a human-readable string.
     * @return The decoded sequence as a string.
     */
    std::string pgalign::sequence::decode() const
    {
        const auto block_count = underlying_type::size();
        const auto element_count = m_columns.size() - 1;

        std::string decoded (m_columns[element_count], encoder::decode(encoder::gap));

        for(size_t i = 0, n = 0; n < element_count; ++i) {
            const encoder::block& block = underlying_type::block(i);

            for(uint8_t j = 0; j < encoder::block_size && n < element_count; ++j, ++n)
                decoded[m_columns[n]] = encoder::decode(encoder::access(block, j));
        }

        return decoded;
    }

    /**
     * Formats a gapped sequence to be printed.
     * @param tgt The target gapped sequence to be formatted.
     * @return The formatted buffer.
     */
    auto fmt::formatter<pgalign::sequence>::parse(const pgalign::sequence& tgt) -> return_type
    {
        return adapt(tgt.decode());
    }
}

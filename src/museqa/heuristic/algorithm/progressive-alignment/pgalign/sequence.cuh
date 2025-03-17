/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the profile-aligner module's sequence data structure.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <cstdint>

#include "buffer.hpp"
#include "format.hpp"
#include "encoder.hpp"
#include "sequence.hpp"

namespace museqa
{
    namespace pgalign
    {
        /**
         * Represents an extendable sequence, in which gaps can be inserted in between
         * the sequence's elements.
         * @since 0.1.1
         */
        class sequence : protected museqa::sequence
        {
            public:
                using index_type = uint32_t;                /// The sequence element's index type.

            protected:
                using underlying_type = museqa::sequence;   /// The underlying sequence type.
                using index_buffer = buffer<index_type>;    /// The sequence's index buffer type.

            protected:
                index_buffer m_columns;                     /// The sequence's columns' indeces.

            public:
                inline sequence() noexcept = default;
                inline sequence(const sequence&) noexcept = default;
                inline sequence(sequence&&) noexcept = default;

                /**
                 * Initializes a new expandable sequence from a simple sequence instance.
                 * @param original The sequence to create the new instance from.
                 */
                inline sequence(const underlying_type& original) noexcept
                :   sequence {original, original.unpadded()}
                {}

                inline sequence& operator=(const sequence&) = default;
                inline sequence& operator=(sequence&&) = default;

                __host__ __device__ encoder::unit operator[](ptrdiff_t) const;

                /**
                 * Returns the total length of a sequence counting the gaps within it.
                 * @return The total number of elements and gaps in the sequence.
                 */
                __host__ __device__ size_t length() const noexcept
                {
                    return m_columns[m_columns.size() - 1];
                }

                std::string decode() const;

            private:
                /**
                 * Builds a new expandable sequence from a basic sequence instance.
                 * @param sequence The original sequence instance.
                 * @param length The unpadded sequence length.
                 */
                inline sequence(const underlying_type& sequence, size_t length)
                :   underlying_type {sequence}
                ,   m_columns {index_buffer::make(length + 1)}
                {
                    for (size_t i = 0; i <= length; ++i)
                        m_columns[i] = (index_type) i;
                }
        };
    }

    namespace fmt
    {
        /**
         * Formats an extendable sequence to be printed.
         * @since 0.1.1
         */
        template <>
        struct formatter<pgalign::sequence> : public adapter<std::string>
        {
            auto parse(const pgalign::sequence&) -> return_type;
        };
    }
}

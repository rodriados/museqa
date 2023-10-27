/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <string>

#include <museqa/environment.h>
#include <museqa/bio/sequence.hpp>
#include <museqa/io/format/reader.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::fasta
{
    /**
     * The reader for sequence files of FASTA format. The FASTA format is a widely
     * used format to store biological sequences: it consists simply of a sequence's
     * description and its contents represented in plain text.
     * @since 1.0
     */
    class reader_t : public io::format::reader_t<bio::sequence_t>
    {
        private:
            typedef io::format::reader_t<bio::sequence_t> underlying_t;

        public:
            inline reader_t() noexcept = default;
            inline reader_t(const reader_t&) noexcept = delete;
            inline reader_t(reader_t&&) noexcept = default;

            using underlying_t::reader_t;

            inline reader_t& operator=(const reader_t&) noexcept = delete;
            inline reader_t& operator=(reader_t&&) noexcept = default;

            /**
             * Reads a sequence from the FASTA file.
             * @return The sequence extracted from file.
             */
            auto read() -> bio::sequence_t override;
    };
}

MUSEQA_END_NAMESPACE

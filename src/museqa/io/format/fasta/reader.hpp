/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of FASTA format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <fstream>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>

#include <museqa/io/exception.hpp>
#include <museqa/io/format/reader.hpp>
#include <museqa/io/format/generic/dataset.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::fasta
{
    /**
     * The reader for sequence dataset files of FASTA format. The FASTA format is
     * a widely used format to store biological sequences: it consists simply of
     * a sequence's description and its contents represented in plain text.
     * @since 1.0
     */
    struct reader_t : public io::format::reader_t<bio::sequence::dataset_t>
    {
        /**
         * Extracts all sequences in FASTA format from the given stream.
         * @param stream The FASTA format stream to be parsed.
         * @return A pointer to the parsed sequence dataset.
         */
        auto read_from_stream(std::istream& stream) const
        -> memory::pointer::unique_t<bio::sequence::dataset_t>;

        /**
         * Parses an instance of the target type from a file.
         * @param path The path of the file to be parsed.
         * @return A pointer to an instance of the target type.
         */
        auto read(const std::filesystem::path& path) const
        -> memory::pointer::unique_t<bio::sequence::dataset_t> override
        {
            if (auto fstream = std::ifstream(path); !fstream.fail())
                return read_from_stream(fstream);
            throw io::exception_t("file does not exist or is not readable");
        }
    };
}

MUSEQA_END_NAMESPACE

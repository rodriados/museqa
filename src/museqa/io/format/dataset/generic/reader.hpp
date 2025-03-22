/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of a generic sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <istream>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/io/exception.hpp>
#include <museqa/io/format/reader.hpp>
#include <museqa/io/format/dataset/reader.hpp>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::dataset::generic
{
    /**
     * A generic file format reader for biological sequences datasets. This reader
     * uses the extension of files to determine which specialized reader to use.
     * @since 1.0
     */
    struct reader_t : io::format::dataset::reader_t
    {
        /**
         * Reads a sequence dataset instance from a file.
         * @param path The file to read a sequence dataset from.
         * @return A pointer to the sequence dataset read from the file.
         */
        dataset_ptr_t read(const std::filesystem::path&) const override;

        /**
         * Rejects reading stream as the format can only be known from a file's
         * path extension. Therefore, we bail out immediately.
         * @throws The file format is unknown and stream cannot be read.
         */
        MUSEQA_INLINE dataset_ptr_t read_from_stream(std::istream&) const override
        {
            throw io::exception_t("unable to read stream of unknown format");
        }
    };
}

/**
 * Creates a sequence dataset reader that automatically identifies the file format
 * reader to be used depending on the given filename extension.
 * @return A generic format reader instance for sequence datasets.
 */
template <>
MUSEQA_INLINE auto factory::io::format::reader<bio::sequence::dataset_t>() noexcept
-> museqa::memory::pointer::unique_t<museqa::io::format::reader_t<bio::sequence::dataset_t>>
{
    using generic_reader_t = museqa::io::format::dataset::generic::reader_t;
    return factory::memory::pointer::unique<generic_reader_t>();
}

MUSEQA_END_NAMESPACE

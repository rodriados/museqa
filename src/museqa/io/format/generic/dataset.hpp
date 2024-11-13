/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of a generic sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <filesystem>

#include <museqa/environment.h>
#include <museqa/io/format/reader.hpp>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::generic::dataset
{
    /**
     * A generic file format reader for biological sequences datasets. This reader
     * uses the extension of file paths to determine which parser must be used.
     * @since 1.0
     */
    struct reader_t : public io::format::reader_t<bio::sequence::dataset_t>
    {
        /**
         * Parses a sequence dataset instance from a generic format file.
         * @param path The path of the file to be parsed.
         * @return A pointer to an instance of a sequence dataset.
         */
        auto read(const std::filesystem::path& path) const
        -> memory::pointer::unique_t<bio::sequence::dataset_t> override;
    };
}

/**
 * Creates a sequence dataset file format reader that automatically identifies the
 * file format parser to be used depending on the given file extension.
 * @return A generic format reader instance for sequence datasets.
 */
template <>
MUSEQA_INLINE auto factory::io::format::reader<bio::sequence::dataset_t>() noexcept
-> museqa::memory::pointer::unique_t<
    museqa::io::format::reader_t<bio::sequence::dataset_t>>
{
    using generic_reader_t = museqa::io::format::generic::dataset::reader_t;
    return factory::memory::pointer::unique<generic_reader_t>();
}

MUSEQA_END_NAMESPACE

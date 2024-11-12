/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of a generic sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#include <filesystem>
#include <unordered_map>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/allocator.hpp>
#include <museqa/memory/pointer/unique.hpp>

#include <museqa/io/format/reader.hpp>
#include <museqa/io/format/fasta/reader.hpp>
#include <museqa/io/format/generic/dataset.hpp>

MUSEQA_BEGIN_NAMESPACE

using dataset_reader_t  = io::format::reader_t<bio::sequence::dataset_t>;
using dataset_pointer_t = memory::pointer::unique_t<bio::sequence::dataset_t>;
using lambda_reader_t   = dataset_pointer_t(const std::filesystem::path&);

/**
 * Parses a sequence dataset file of FASTA format.
 * @param path The path of the file to be parsed.
 * @return A pointer to an instance of a sequence dataset.
 */
static dataset_pointer_t parse_fasta_format(const std::filesystem::path& path)
{
    auto reader = io::format::fasta::reader_t();
    return reader.read(path);
}

/**
 * The mapping for sequence dataset file formats to their respective parsers. This
 * mapping should contain parsers for all known file extension types.
 * @since 1.0
 */
static const std::unordered_map<std::string, lambda_reader_t*> lambda_readers = {
    { ".fasta", &parse_fasta_format }
  , {   ".fas", &parse_fasta_format }
  , {   ".faa", &parse_fasta_format }
  , {    ".fa", &parse_fasta_format }
};

/**
 * A generic file format reader for biological sequences datasets. This reader uses
 * the extension of the given file paths to determine which parser must be used.
 * @since 1.0
 */
struct generic_dataset_reader_t : dataset_reader_t
{
    /**
     * Parses a sequence dataset instance from a generic format file.
     * @param path The path of the file to be parsed.
     * @return A pointer to an instance of a sequence dataset.
     */
    auto read(const std::filesystem::path& path) const
    -> memory::pointer::unique_t<bio::sequence::dataset_t> override
    try {
        const auto lambda_reader = lambda_readers.at(path.extension());
        return lambda_reader(path);
    } catch (const std::out_of_range&) {
        throw io::exception_t("no parser known for given file type");
    }
};

/**
 * Creates a sequence dataset file format reader that automatically identifies the
 * file format parser to be used depending on the given file extension.
 * @return A generic format reader instance for sequence datasets.
 */
template <>
auto factory::io::format::reader<bio::sequence::dataset_t>() noexcept
-> museqa::memory::pointer::unique_t<dataset_reader_t>
{
    return factory::memory::pointer::unique<generic_dataset_reader_t>();
}

MUSEQA_END_NAMESPACE

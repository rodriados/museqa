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
#include <museqa/memory/pointer/unique.hpp>

#include <museqa/io/format/reader.hpp>
#include <museqa/io/format/fasta/reader.hpp>
#include <museqa/io/format/generic/dataset.hpp>

MUSEQA_BEGIN_NAMESPACE

using dataset_pointer_t = memory::pointer::unique_t<bio::sequence::dataset_t>;
using lambda_reader_t   = dataset_pointer_t(const std::filesystem::path&);

/**
 * Reads a sequence dataset file using the given reader type.
 * @tparam The sequence dataset reader type to read file with.
 * @param path The path of the file to be parsed.
 * @return A pointer to an instance of a sequence dataset.
 */
template <typename R>
static dataset_pointer_t read_format(const std::filesystem::path& path)
{
    const auto reader = R();
    return reader.read(path);
}

/**
 * The mapping for sequence dataset file formats to their respective parsers. This
 * mapping should contain parsers for all known file extension types.
 * @since 1.0
 */
static const std::unordered_map<std::string, lambda_reader_t*> lambda_readers = {
    { ".fasta", &read_format<io::format::fasta::reader_t> }
  , {   ".fas", &read_format<io::format::fasta::reader_t> }
  , {   ".faa", &read_format<io::format::fasta::reader_t> }
  , {    ".fa", &read_format<io::format::fasta::reader_t> }
};

/**
 * Parses a sequence dataset instance from a generic format file.
 * @param path The path of the file to be parsed.
 * @return A pointer to an instance of a sequence dataset.
 */
dataset_pointer_t io::format::generic::dataset::reader_t::read(
    const std::filesystem::path& path
) const try {
    const auto lambda = lambda_readers.at(path.extension());
    return lambda (path);
} catch (const std::out_of_range&) {
    throw io::exception_t("no reader known for given file extension type");
}

MUSEQA_END_NAMESPACE

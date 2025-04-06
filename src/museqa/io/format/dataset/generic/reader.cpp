/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The implementation of a generic sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#include <string>
#include <stdexcept>
#include <filesystem>
#include <unordered_map>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>

#include <museqa/io/exception.hpp>
#include <museqa/io/format/dataset/reader.hpp>
#include <museqa/io/format/dataset/fasta/reader.hpp>
#include <museqa/io/format/dataset/generic/reader.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace
{
    using base_reader_t    = io::format::dataset::reader_t;
    using reader_ptr_t     = memory::pointer::unique_t<base_reader_t>;
    using reader_factory_t = reader_ptr_t();

    /**
     * Instantiates a sequence format reader of the given type.
     * @tparam R The sequence reader type to instantiate.
     * @return A pointer to an instance of the given reader type.
     */
    template <typename R>
    static reader_ptr_t make_reader()
    {
        return factory::memory::pointer::unique<R>();
    }

    /**
     * The mapping for sequence file extensions to their respective format readers.
     * This mapping should contain entries for all known format extensions.
     * @since 1.0
     */
    static const std::unordered_map<std::string, reader_factory_t*> reader_factory = {
        { ".fasta", &make_reader<io::format::dataset::fasta::reader_t> }
      , {   ".fas", &make_reader<io::format::dataset::fasta::reader_t> }
      , {   ".faa", &make_reader<io::format::dataset::fasta::reader_t> }
      , {    ".fa", &make_reader<io::format::dataset::fasta::reader_t> }
    };
}

/**
 * Reads a sequence dataset from a file into an existing instance.
 * @param dataset The dataset instance to read into from file.
 * @param path The file to read a sequence dataset from.
 */
void io::format::dataset::generic::reader_t::read_from_file(
    dataset_ptr_t& dataset
    const std::filesystem::path& path
) const try {
    const auto factory = reader_factory.at(path.extension());
    const auto reader  = factory ();
    reader->read_from_file(dataset, path);
} catch (const std::out_of_range&) {
    throw io::exception_t("no reader known for given file extension type");
}

MUSEQA_END_NAMESPACE

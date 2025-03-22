/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of the base sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <utility>
#include <fstream>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/io/exception.hpp>
#include <museqa/io/format/reader.hpp>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/bio/sequence/attribute.hpp>
#include <museqa/memory/pointer/unique.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::dataset
{
    /**
     * The base type for every biological sequence dataset reader. It uses a buffer
     * to temporary store sequence data until they are committed to a dataset.
     * @since 1.0
     */
    class reader_t : public io::format::reader_t<bio::sequence::dataset_t>
    {
        protected:
            using dataset_ptr_t = memory::pointer::unique_t<bio::sequence::dataset_t>;

        public:
            /**
             * Reads a sequence dataset instance from a stream.
             * @param stream The stream to read a sequence dataset from.
             * @return A pointer to the sequence dataset read from the stream.
             */
            virtual dataset_ptr_t read_from_stream(std::istream&) const = 0;

            /**
             * Reads a sequence dataset instance from a file.
             * @param path The file to read a sequence dataset from.
             * @return A pointer to the sequence dataset read from the file.
             */
            MUSEQA_INLINE dataset_ptr_t read(const std::filesystem::path& path) const override
            {
                if (auto fstream = std::ifstream(path); !fstream.fail())
                    return read_from_stream(fstream);
                else throw io::exception_t("file does not exist or it not readable");
            }
    };
}

MUSEQA_END_NAMESPACE

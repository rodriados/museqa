/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of FASTA sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <istream>

#include <museqa/environment.h>
#include <museqa/io/format/dataset/reader.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace io::format::dataset::fasta
{
    /**
     * A reader for the FASTA format of biological sequences datasets. The FASTA
     * is a widely used format to store biological sequences: it consists simply
     * of a sequence's description and its contents represented in plain text.
     * @since 1.0
     */
    struct reader_t : io::format::dataset::reader_t
    {
        /**
         * Reads a sequence dataset in FASTA format from a stream.
         * @param dataset The dataset instance to read into from stream.
         * @param stream The stream to read a sequence dataset from.
         */
        void read_from_stream(dataset_ptr_t&, std::istream&) const override;
    };
}

MUSEQA_END_NAMESPACE

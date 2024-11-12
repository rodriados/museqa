/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The definition of a generic sequence dataset format file reader.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/io/format/reader.hpp>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>

MUSEQA_BEGIN_NAMESPACE

/**
 * Creates a sequence dataset file format reader that automatically identifies the
 * file format parser to be used depending on the given file extension.
 * @return A generic format reader instance for sequence datasets.
 */
template <>
auto factory::io::format::reader<bio::sequence::dataset_t>() noexcept
-> museqa::memory::pointer::unique_t<
    museqa::io::format::reader_t<
        bio::sequence::dataset_t>>;

MUSEQA_END_NAMESPACE

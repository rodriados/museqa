/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Declaration of bootstrap module's functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/io/format/generic/dataset.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::bootstrap
{
    /**
     * Loads biological sequences from a list of source files into memory.
     * @param filenames The source files to load sequences from.
     * @return The dataset of loaded sequences.
     */
    MUSEQA_INLINE bio::sequence::dataset_t load_from_files(const std::vector<std::string>& filenames)
    {
        auto result = bio::sequence::dataset_t();
        auto reader = io::format::generic::dataset::reader_t();

        for (const auto& filename : filenames)
            if (auto db = reader.read(filename); !db.empty())
                result.merge(*db);

        return result;
    }
}

MUSEQA_END_NAMESPACE

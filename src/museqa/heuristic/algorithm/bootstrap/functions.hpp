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

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::bootstrap
{
    /**
     * Loads biological sequences from a list of source files.
     * @param files The source files to load sequences from.
     * @return The list of loaded sequences.
     */
    auto load_sequences(const std::vector<std::string>& files)
        -> std::vector<bio::sequence::data_t>;
}

MUSEQA_END_NAMESPACE

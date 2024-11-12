/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a dataset of biological sequences.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <unordered_map>

#include <museqa/environment.h>

#include <museqa/bio/sequence/buffer.hpp>
#include <museqa/bio/sequence/attribute.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace bio::sequence
{
    /**
     * Sequence buffer and information data. This structure groups a sequence's
     * symbol buffer with all of its available attributes.
     * @since 1.0
     */
    struct data_t
    {
        sequence::buffer_t buffer;
      #ifdef MUSEQA_ENABLE_SEQUENCE_ATTRIBUTES
        sequence::attribute::bag_t attribute;
      #endif
    };

    /**
     * Sequence dataset for gathering all biological sequences into a single reference.
     * This dataset can be freely modified, as well as merged with others to form
     * a single dataset instance.
     * @since 1.0
     */
    struct dataset_t : std::unordered_map<std::string, data_t> {};
}

MUSEQA_END_NAMESPACE

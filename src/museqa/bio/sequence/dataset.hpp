/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a dataset of biological sequences.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

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
        sequence::attribute::bag_t attribute;
    };

    /**
     * Sequence dataset for gathering all biological sequences into a single reference.
     * This dataset can be freely modified, as well as merged with others to form
     * a single dataset instance.
     * @since 1.0
     */
    struct dataset_t : std::vector<data_t> {
        using std::vector<data_t>::vector;
        using std::vector<data_t>::operator=;
    };
}

MUSEQA_END_NAMESPACE

/**
 * Implements a hash operator to uniquely identify a sequence from its description.
 * This operator may be used to deduplicate sequences from a dataset.
 * @since 1.0
 */
template <>
struct std::hash<MUSEQA_NAMESPACE::bio::sequence::data_t>
{
    typedef MUSEQA_NAMESPACE::bio::sequence::data_t target_t;

    /**
     * Calculates a hash for the given sequence description.
     * @param sequence The sequence to calculate a hash for.
     * @return The resulting hash for the given sequence.
     */
    MUSEQA_INLINE auto operator()(const target_t& sequence) const
    {
        constexpr auto hash = std::hash<std::string>();
        return hash (sequence.attribute.description);
    }
};

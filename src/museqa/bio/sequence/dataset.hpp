/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation of a dataset of biological sequences.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <unordered_set>

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
        std::string description;
        sequence::buffer_t buffer;
        sequence::attribute_bag_t attribute;
    };

    /**
     * Sequence dataset for gathering all biological sequences into a single reference.
     * This dataset can be freely modified, as well as merged with others to form
     * a single dataset instance.
     * @since 1.0
     */
    using dataset_t = std::unordered_set<data_t>;
}

MUSEQA_END_NAMESPACE

/**
 * The hash operator for a sequence data object. A sequence is always indexed by
 * its description, so that two sequences with the same description are considered
 * to be the same and will be deduplicated.
 * @since 1.0
 */
template <>
struct std::hash<MUSEQA_NAMESPACE::bio::sequence::data_t>
{
    using target_t = MUSEQA_NAMESPACE::bio::sequence::data_t;

    /**
     * Hashes a sequence data object through its description.
     * @param sequence The sequence data to be hashed.
     * @return The sequence description hash value.
     */
    MUSEQA_INLINE size_t operator()(const target_t& sequence) const noexcept
    {
        constexpr auto hasher = std::hash<std::string>();
        return hasher(sequence.description);
    }
};

/**
 * The equality operator between sequence data objects. We assume that sequences
 * always have different descriptions and, therefore, if two equal descriptions
 * are found, we consider both sequences to be the same.
 * @since 1.0
 */
template <>
struct std::equal_to<MUSEQA_NAMESPACE::bio::sequence::data_t>
{
    using target_t = MUSEQA_NAMESPACE::bio::sequence::data_t;

    /**
     * Checks whether two sequence data objects are equal.
     * @param a The first sequence to be compared.
     * @param b The second sequence to be compared.
     * @return Are both sequences considered the same?
     */
    MUSEQA_INLINE bool operator()(const target_t& a, const target_t& b) const
    {
        return a.description == b.description;
    }
};

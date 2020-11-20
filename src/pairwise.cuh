/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Exposes an interface for the heuristics' pairwise module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include "pairwise/pairwise.cuh"
#include "module/pairwise.cuh"

namespace museqa
{
    namespace pairwise
    {
        /**
         * Alias for the pairwise module's runner.
         * @since 0.1.1
         */
        using module = museqa::module::pairwise;

        /**
         * Alias for the pairwise module's conduit.
         * @since 0.1.1
         */
        using conduit = museqa::module::pairwise::conduit;
    }

    /**
     * The score of a sequence pair alignment. Represents the score of an alignment
     * of a pair of sequences.
     * @since 0.1.1
     */
    using score = pairwise::score;

    /**
     * Represents a reference for a sequence. This type is simply an index identification
     * for a sequence in the sequence database.
     * @since 0.1.1
     */
    using seqref = pairwise::seqref;

    /**
     * A pair of sequence identifiers. This is the union pair object the pairwise
     * module processes. The sequence references can be accessed either by their
     * respective names or by their indeces.
     * @since 0.1.1
     */
    using pair = pairwise::pair;
}

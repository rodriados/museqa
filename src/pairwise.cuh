/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Exposes an interface for the heuristics' pairwise module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include "io.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "pipeline.hpp"
#include "bootstrap.hpp"

/*
 * The heuristic's sequence pairwise alignment module.
 * This module is responsible for performing a sequence pairwise alignment, and
 * returning a score for each aligned pair of sequences. As an input, this module
 * must receive from the pipeline a database of sequences to be aligned, and it
 * will produce a distance matrix for the score of each pair's alignment.
 */

#include "pairwise/pairwise.cuh"

namespace museqa
{
    namespace module
    {
        /**
         * Defines the module's pipeline manager. This object will be the one responsible
         * for checking and managing the module's execution when on a pipeline.
         * @since 0.1.1
         */
        struct pairwise : public pipeline::module
        {
            struct conduit;                                 /// The module's conduit type.

            typedef museqa::module::bootstrap previous;     /// The expected previous module.
            typedef pointer<pipeline::conduit> pipe;        /// The generic conduit type alias.

            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "pairwise";
            }

            auto run(const io::manager&, const pipe&) const -> pipe override;
            auto check(const io::manager&) const -> bool override;
        };

        /**
         * Defines the module's conduit. This conduit is composed of the sequences
         * that have been aligned and their pairwise distance matrix.
         * @since 0.1.1
         */
        struct pairwise::conduit : public pipeline::conduit
        {
            typedef museqa::pairwise::distance_matrix distance_matrix;

            const pointer<database> db;         /// The loaded sequences' database.
            const distance_matrix distances;    /// The sequences' pairwise distances.
            const size_t total;                 /// The total number of sequences.

            inline conduit() noexcept = delete;
            inline conduit(const conduit&) = default;
            inline conduit(conduit&&) = default;

            /**
             * Instantiates a new conduit.
             * @param db The sequence database to transfer to the next module.
             * @param dmat The database's resulting pairwise distance matrix.
             */
            inline conduit(const pointer<database>& db, const distance_matrix& dmat) noexcept
            :   db {db}
            ,   distances {dmat}
            ,   total {db->count()}
            {}

            inline conduit& operator=(const conduit&) = delete;
            inline conduit& operator=(conduit&&) = delete;
        };
    }

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

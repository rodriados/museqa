/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the heuristic's pairwise module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

/*
 * The heuristic's sequence pairwise alignment module.
 * This module is responsible for performing a sequence pairwise alignment, and
 * returning a score for each aligned pair of sequences. As an input, this module
 * must receive from the pipeline a database of sequences to be aligned, and it
 * will produce a distance matrix for the score of each pair's alignment.
 */

#include "io.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "pipeline.hpp"

#include "module/bootstrap.hpp"
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
            struct conduit;                             /// The module's conduit type.

            typedef module::bootstrap previous;         /// The expected previous module.
            typedef pointer<pipeline::conduit> pipe;    /// The generic conduit type alias.

            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "pairwise";
            }

            auto run(const io::service&, const pipe&) const -> pipe override;
            auto check(const io::service&) const -> bool override;
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
}

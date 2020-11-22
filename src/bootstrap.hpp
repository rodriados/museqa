/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the heuristic's bootstrap module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

/*
 * The heuristic's pipeline bootstrap module.
 * This module is responsible for loading all sequence files given by the user via
 * command line. After all sequences have been loaded by the cluster's master node,
 * they are broadcasted and distributed to all other nodes. If no sequence files
 * have been given, then this module shall skip any processing.
 */

#include "io.hpp"
#include "pointer.hpp"
#include "database.hpp"
#include "pipeline.hpp"

namespace museqa
{
    namespace module
    {
        /**
         * The heuristics' bootstrap pipeline module. This module is responsible
         * for loading all sequences given as input into all nodes.
         * @since 0.1.1
         */
        struct bootstrap : public pipeline::module
        {
            struct conduit;                             /// The module's conduit type.

            typedef void previous;                      /// The expected previous module.
            typedef pointer<pipeline::conduit> pipe;    /// The generic conduit type alias.

            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "bootstrap";
            }

            /**
             * Checks whether command line arguments produce a valid module state.
             * @param io The pipeline's IO service instance.
             * @return Are the given command line arguments valid?
             */
            inline auto check(const io::service& io) const -> bool override
            {
                return io.filecount() > 0;
            }

            auto run(const io::service&, const pipe&) const -> pipe override;
        };

        /**
         * The bootstrap module conduit. This module's conduit is composed of all
         * sequences loaded from command line.
         * @since 0.1.1
         */
        struct bootstrap::conduit : public pipeline::conduit
        {
            const pointer<database> db;     /// The database with loaded sequences.
            const size_t total;             /// The total number of sequences.

            inline conduit() noexcept = delete;
            inline conduit(const conduit&) = default;
            inline conduit(conduit&&) = default;

            /**
             * Instantiates a new conduit.
             * @param db The sequence database to transfer to the next module.
             */
            inline explicit conduit(const pointer<database>& db) noexcept
            :   db {db}
            ,   total {db->count()}
            {}

            inline conduit& operator=(const conduit&) = default;
            inline conduit& operator=(conduit&&) = default;
        };
    }
}

/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Exposes an interface for the heuristics' profile-aligner module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include "io.hpp"
#include "pointer.hpp"
#include "pipeline.hpp"
#include "phylogeny.cuh"

/*
 * The heuristic's final stage profile-aligner.
 * This module is responsible for building a single global alignment of all sequences
 * according to the phylogenetic tree previously obtained.
 */

#include "pgalign/pgalign.cuh"

namespace museqa
{
    namespace module
    {
        /**
         * Defines the module's pipeline manager. This object will be the one responsible
         * for checking and managing the module's execution when on a pipeline.
         * @since 0.1.1
         */
        struct pgalign : public pipeline::module
        {
            struct conduit;                                 /// The module's conduit type.
            typedef museqa::module::phylogeny previous;     /// The expected previous module.

            /**
             * Returns an string identifying the module's name.
             * @return The module's name.
             */
            inline auto name() const -> const char * override
            {
                return "pgalign";
            }

            auto run(const io::manager&, pipeline::pipe&) const -> pipeline::pipe override;
            auto check(const io::manager&) const -> bool override;
        };

        /**
         * Defines the module's conduit. This conduit is composed of the final alignment
         * of all sequences according to the given phylogenetic tree.
         * @since 0.1.1
         */
        struct pgalign::conduit : public pipeline::conduit
        {
            typedef museqa::pgalign::alignment alignment;

            alignment aligned;

            inline conduit() noexcept = delete;
            inline conduit(const conduit&) = default;
            inline conduit(conduit&&) = default;

            inline conduit(alignment& alignment)
            :   aligned {std::move(alignment)}
            {}

            inline conduit& operator=(const conduit&) = delete;
            inline conduit& operator=(conduit&&) = delete;
        };
    }

    namespace pgalign
    {
        /**
         * Alias for the profile-aligner module's runner.
         * @since 0.1.1
         */
        using module = museqa::module::pgalign;

        /**
         * Alias for the profile-aligner module's conduit.
         * @since 0.1.1
         */
        using conduit = museqa::module::pgalign::conduit;
    }

    /**
     * Represents the alignment between one or more sequences.
     * @since 0.1.1
     */
    using alignment = pgalign::alignment;
}

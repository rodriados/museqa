/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the profile-aligner module's functionality.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2020-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include "utils.hpp"
#include "functor.hpp"
#include "database.hpp"
#include "phylogeny.cuh"

#include "pgalign/sequence.cuh"
#include "pgalign/alignment.cuh"

namespace museqa
{
    namespace pgalign
    {
        /**
         * Represents a common profile-aligner algorithm context.
         * @since 0.1.1
         */
        struct context
        {
            const museqa::database& db;         /// The loaded sequences' database.
            const phylogeny::phylotree& tree;   /// The sequences' alignment guiding tree.
            const size_t count;                 /// The total number of sequences being aligned.
        };

        /**
         * Functor responsible for instantiating an algorithm.
         * @see pgalign::run
         * @since 0.1.1
         */
        using factory = functor<struct algorithm *()>;

        /**
         * Represents a profile-aligner module algorithm.
         * @since 0.1.1
         */
        struct algorithm
        {
            inline algorithm() noexcept = default;
            inline algorithm(const algorithm&) noexcept = default;
            inline algorithm(algorithm&&) noexcept = default;

            virtual ~algorithm() = default;

            inline algorithm& operator=(const algorithm&) = default;
            inline algorithm& operator=(algorithm&&) = default;

            virtual auto run(const context&) const -> alignment = 0;

            static auto has(const std::string&) -> bool;
            static auto make(const std::string&) -> const factory&;
            static auto list() noexcept -> const std::vector<std::string>&;
        };

        /**
         * Runs the module when not on a pipeline.
         * @param db The database of sequences to align.
         * @param tree The multiple alignment's guiding tree.
         * @param count The total number of sequences to align.
         * @param algorithm The chosen profile-aligner algorithm.
         * @return The chosen algorithm's resulting multiple sequence alignment.
         */
        inline alignment run(
                const museqa::database& db
            ,   const phylogeny::phylotree& tree
            ,   const size_t count
            ,   const std::string& algorithm = "default"
            )
        {
            auto lambda = pgalign::algorithm::make(algorithm);
            
            const pgalign::algorithm *worker = lambda ();
            auto result = worker->run({db, tree, count});
            
            delete worker;
            return result;
        }
    }
}

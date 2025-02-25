/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Module for the sequence pairwise-alignment heuristic step.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/pipeline.hpp>

#include <museqa/heuristic/algorithm/pairwise/matrix.hpp>
#include <museqa/heuristic/algorithm/pairwise/parameters.hpp>

/*
 * The heuristic's sequence pairwise alignment module.
 * This module is responsible for performing a sequence pairwise-alignment, and
 * returning a score for each aligned pair of sequences. As an input, this module
 * must receive from the pipeline a list of sequences to be aligned, and it will
 * produce a distance matrix for the score of each pair's alignment.
 */

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::module
{
    /**
     * The pairwise module is responsible for generating a distance matrix of every
     * possible alignment between a pair of sequences for a target set of sequences.
     * As such, this method expects the set of sequences to be present on the pipeline.
     * @since 1.0
     */
    class pairwise_t : public pipeline::module_t
    {
        public:
            typedef heuristic::algorithm::pairwise::matrix_t matrix_t;
            typedef heuristic::algorithm::pairwise::parameters_t parameters_t;

        public:
            struct algorithm_t;

        public:
            inline static constexpr auto matrix = pipeline::key<matrix_t>("pairwise::matrix");

        protected:
            parameters_t m_params = {};

        public:
            inline pairwise_t() = default;
            inline pairwise_t(const pairwise_t&) = default;
            inline pairwise_t(pairwise_t&&) = default;

            /**
             * Initializes an instance of pairwise module with the given parameters.
             * @param params The parameters to use for module execution.
             */
            inline explicit pairwise_t(const parameters_t& params) noexcept
              : m_params (params)
            {}

            inline pairwise_t& operator=(const pairwise_t&) = default;
            inline pairwise_t& operator=(pairwise_t&&) = default;

            void run(pipeline::pipe_t&) const override;
    };

    /**
     * The abstraction of a pairwise-module algorithm, so that the module may offer
     * different algorithms and implementations where everyone produces the output.
     * @since 1.0
     */
    struct pairwise_t::algorithm_t
    {
        inline algorithm_t() noexcept = default;
        inline algorithm_t(const algorithm_t&) noexcept = default;
        inline algorithm_t(algorithm_t&&) noexcept = default;

        virtual ~algorithm_t() = default;

        inline algorithm_t& operator=(const algorithm_t&) = default;
        inline algorithm_t& operator=(algorithm_t&&) = default;

        /**
         * The algorithm context that wraps all user input values and relevant data
         * from previous modules in the pipeline. This will contain the data that
         * will be directly accessed and used by the implemented algorithms.
         * @since 1.0
         */
        struct context_t {
            const parameters_t& params;
        };

        virtual auto run(const context_t&) -> matrix_t = 0;
    };
}

MUSEQA_END_NAMESPACE

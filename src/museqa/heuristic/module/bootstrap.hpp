/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Alias for heuristic bootstrap module.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <vector>

#include <museqa/environment.h>
#include <museqa/pipeline.hpp>
#include <museqa/bio/sequence.hpp>
#include <museqa/memory/pointer.hpp>

#include <museqa/heuristic/algorithm/bootstrap/functions.hpp>
#include <museqa/heuristic/algorithm/bootstrap/parameters.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::module
{
    /**
     * This module usually is the first in a pipeline execution, as it is responsible
     * for loading biological sequences from their files into all cluster nodes.
     * @since 1.0
     */
    class bootstrap_t : public pipeline::module_t
    {
        public:
            typedef std::vector<bio::sequence::data_t> sequences_t;
            typedef heuristic::algorithm::bootstrap::parameters_t parameters_t;

        public:
            inline static constexpr auto sequences = pipeline::key<sequences_t>("bootstrap::sequences");

        protected:
            parameters_t m_params = {};

        public:
            inline bootstrap_t() = default;
            inline bootstrap_t(const bootstrap_t&) = default;
            inline bootstrap_t(bootstrap_t&&) = default;

            /**
             * Initializes an instance of bootstrap module with the given parameters.
             * @param params The parameters to use for module execution.
             */
            inline bootstrap_t(const parameters_t& params) noexcept
              : m_params (params)
            {}

            inline bootstrap_t& operator=(const bootstrap_t&) = default;
            inline bootstrap_t& operator=(bootstrap_t&&) = default;

            /**
             * Executes the module's task and loads sequences from source files.
             * @param pipe The pipeline's transitive state instance.
             */
            inline void run(pipeline::pipe_t& pipe) const override
            {
                auto sequences = factory::memory::pointer::shared<sequences_t>();
                    *sequences = heuristic::algorithm::bootstrap::load_sequences(m_params.input.sequences);

                pipe->set(bootstrap_t::sequences, sequences);
            }
    };
}

MUSEQA_END_NAMESPACE

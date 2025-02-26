/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Parameters definition for the pairwise-alignment heuristic step.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>

#include <museqa/environment.h>
#include <museqa/heuristic/algorithm/pairwise-alignment/common.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::pairwise
{
    /**
     * Defines the parameters for a pairwise algorithm implementation. These parameters
     * are the interface between the user, the module and the implemented algorithms.
     * @since 1.0
     */
    struct parameters_t
    {
        /**
         * The module's input parameters. These paremeters are responsible for giving
         * the module all required values to select and execute an algorithm.
         * @since 1.0
         */
        struct input_t {
            std::string algorithm = "default";
            std::string scoring_table = "default";
            score_t gap_cost = 0.f;
            score_t gap_extension_cost = 0.f;
        } input;

        /**
         * The module's output parameters. These parameters are responsible for
         * configuring how the module must output its produced result, if ever.
         * @since 1.0
         */
        struct output_t {
            std::string file = {};
        } output;
    };
}

MUSEQA_END_NAMESPACE

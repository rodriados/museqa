/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Parameters definition for the bootstrap heuristic step.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <string>
#include <vector>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::bootstrap
{
    /**
     * Defines the parameters for a bootstrap module implementation. These parameters
     * are the interface between the user, the module and the implemented algorithms.
     * @since 1.0
     */
    struct parameters_t
    {
        /**
         * The module's input parameters. These paremeters are indicate what are
         * the objects and files to load data from.
         * @since 1.0
         */
        struct input_t {
            std::vector<std::string> sequence_files = {};
        } input;
    };
}

MUSEQA_END_NAMESPACE

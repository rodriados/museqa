/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Parameters definition for the bootstrap heuristic step.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <vector>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/heuristic/algorithm/parameters.hpp>

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
         * The global heuristic parameters. These parameters are shared between
         * every step in a heuristic pipeline.
         * @since 1.0
         */
        heuristic::algorithm::parameters_t global;

        /**
         * The module's input parameters. These paremeters indicate what are the
         * objects or files to load sequence data from.
         * @since 1.0
         */
        struct input_t {
            std::vector<std::filesystem::path> files = {};
        } input;
    };
}

MUSEQA_END_NAMESPACE

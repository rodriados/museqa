/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Declaration of bootstrap module's functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/shared.hpp>

#include <museqa/heuristic/algorithm/bootstrap/parameters.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::bootstrap::impl
{
    /*
     * Forward declaration of every known implementation for the bootstrap heuristic
     * step. Ideally, these functions should load and distribute sequences, if applicable.
     */
    extern memory::pointer::shared_t<bio::sequence::dataset_t> load_and_distribute(const parameters_t&);
}

namespace heuristic::algorithm::bootstrap
{
    /**
     * Executes the bootstrap pipeline step logic.
     * @param params The bootstrap pipeline step parameters.
     * @return The produced pipeline step result.
     */
    MUSEQA_INLINE auto run(const parameters_t& params)
    {
        return impl::load_and_distribute(params);
    }
}

MUSEQA_END_NAMESPACE

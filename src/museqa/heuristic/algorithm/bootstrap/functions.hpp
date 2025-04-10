/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Declaration of bootstrap module's functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2024-present Rodrigo Siqueira
 */
#pragma once

#include <museqa/environment.h>
#include <museqa/thirdparty/mpiwcpp17.h>

#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>
#include <museqa/heuristic/algorithm/bootstrap/parameters.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace heuristic::algorithm::bootstrap::impl
{
    /*
     * Forward declaration of the default implementation for the bootstrap pipeline
     * step. Ideally, this function should load and distribute sequences, if applicable.
     */
    extern memory::pointer::unique_t<bio::sequence::dataset_t> run_default(const parameters_t&);
}

namespace heuristic::algorithm::bootstrap
{
    /**
     * Loads a dataset of biological sequences from a list of source files.
     * @param filelist The list of files to load the sequence dataset from.
     * @param comm The MPI communicator to distribute sequences to, if applicable.
     * @param root The MPI root process to load the files from, if applicable.
     * @return The dataset of loaded biological sequences.
     */
    MUSEQA_INLINE auto load_from_files(
        const std::vector<std::filesystem::path>& filelist
      #ifndef MUSEQA_AVOID_MPI
        , const mpi::communicator_t comm = mpi::world
        , const mpi::process_t root = mpi::process::root
      #endif
    ) {
        parameters_t params;
          #ifndef MUSEQA_AVOID_MPI
            params.global.comm = comm;
            params.global.root = root;
          #endif
            params.input.filelist = filelist;
        return impl::run_default(params);
    }

    /**
     * Executes the bootstrap pipeline step logic.
     * @param params The bootstrap pipeline step parameters.
     * @return The produced pipeline step result.
     */
    MUSEQA_INLINE auto run(const parameters_t& params)
    {
        return impl::run_default(params);
    }
}

MUSEQA_END_NAMESPACE

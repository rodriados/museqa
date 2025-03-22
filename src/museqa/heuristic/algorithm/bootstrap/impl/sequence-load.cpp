/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Declaration of bootstrap module's functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#include <vector>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/thirdparty/mpiwcpp17.h>

#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/shared.hpp>
#include <museqa/io/format/generic/dataset.hpp>
#include <museqa/heuristic/algorithm/bootstrap/parameters.hpp>

MUSEQA_BEGIN_NAMESPACE

using dataset_ptr_t = memory::pointer::shared_t<bio::sequence::dataset_t>;

namespace
{
    /**
     * Loads a dataset of biological sequences from a list of process local files.
     * @param filelist The list of process local files to load sequences from.
     * @return The dataset of loaded biological sequences.
     */
    static dataset_ptr_t load_from_local_file(const std::vector<std::filesystem::path>& filelist)
    {
        auto dataset = factory::memory::pointer::shared<bio::sequence::dataset_t>();
        auto reader = io::format::generic::dataset::reader_t();

        for (const auto& filepath : filelist)
            if (auto filedb = reader.read(filepath); !filedb.empty())
                dataset->merge(*filedb);

        return dataset;
    }

  #ifndef MUSEQA_AVOID_MPI
    /**
     * Loads a dataset of biological sequences to a cluster of MPI processes.
     * @param filelist The list of file names to load sequences from.
     * @param comm The MPI communicator to distribute sequences to.
     * @param root The MPI communicator's root process to load files from.
     * @return The dataset of loaded biological sequences.
     */
    static dataset_ptr_t load_from_mpi_cluster(
        const std::vector<std::filesystem::path>& filelist
      , mpi::communicator_t comm = mpi::world
      , mpi::process_t root = mpi::process::root
    ) {
        auto dataset = (root == mpi::communicator::rank(comm))
            ? load_from_local_file(filelist)
            : memory::pointer::shared_t<bio::sequence::dataset_t>();

        // TODO: Implement cluster distribution
        return dataset;
    }
  #endif
}

namespace heuristic::algorithm::bootstrap::impl
{
    /**
     * Loads a dataset of biological sequences and distributes it to all processes
     * in the MPI cluster, if applicable, with the given parameters.
     * @param params The parameters to load and distribute the sequences with.
     * @return The dataset of loaded biological sequences.
     */
    dataset_ptr_t load_and_distribute(const parameters_t& params)
    {
      #ifndef MUSEQA_AVOID_MPI
        return load_from_mpi_cluster(
            params.input.files
          , params.global.comm
          , params.global.root
        );
      #else
        return load_from_local_file(params.input.files);
      #endif
    }
}

MUSEQA_END_NAMESPACE

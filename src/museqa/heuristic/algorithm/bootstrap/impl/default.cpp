/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation of module's function for loading and distributing sequences.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2025-present Rodrigo Siqueira
 */
#include <vector>
#include <filesystem>

#include <museqa/environment.h>
#include <museqa/thirdparty/mpiwcpp17.h>

#include <museqa/bio/sequence/dataset.hpp>
#include <museqa/memory/pointer/unique.hpp>
#include <museqa/io/format/dataset/reader.hpp>
#include <museqa/io/format/dataset/generic/reader.hpp>
#include <museqa/heuristic/algorithm/bootstrap/parameters.hpp>

MUSEQA_BEGIN_NAMESPACE

using dataset_t = bio::sequence::dataset_t;
using dataset_ptr_t = memory::pointer::unique_t<dataset_t>;

namespace
{
    /**
     * Loads a dataset of biological sequences from a list of process local files.
     * @param reader The reader instance to load sequences with.
     * @param filelist The list of process local files to load sequences from.
     * @return The dataset of loaded biological sequences.
     */
    static dataset_ptr_t load_from_local_file(
        const io::format::dataset::reader_t& reader
      , const std::vector<std::filesystem::path>& filelist
    ) {
        auto dataset = memory::pointer::make_unique<dataset_t>();

        for (const auto& file : filelist)
            reader.read_from_file(dataset, file);

        return dataset;
    }

  #ifndef MUSEQA_AVOID_MPI
    /**
     * Loads a dataset of biological sequences to a cluster of MPI processes.
     * @param reader The reader instance to load sequences with.
     * @param filelist The list of file names to load sequences from.
     * @param comm The MPI communicator to distribute sequences to.
     * @param root The MPI communicator's root process to load files from.
     * @return The dataset of loaded biological sequences.
     */
    static dataset_ptr_t load_from_mpi_cluster(
        const io::format::dataset::reader_t& reader
      , const std::vector<std::filesystem::path>& filelist
      , const mpi::communicator_t comm = mpi::world
      , const mpi::process_t root = mpi::process::root
    ) {
        auto dataset = root == mpi::communicator::rank(comm)
            ? load_from_local_file(reader, filelist)
            : memory::pointer::make_unique<dataset_t>();

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
    dataset_ptr_t run_default(const parameters_t& params)
    {
        const auto reader = io::format::dataset::generic::reader_t();
      #ifndef MUSEQA_AVOID_MPI
        return load_from_mpi_cluster(
            reader
          , params.input.filelist
          , params.global.comm
          , params.global.root
        );
      #else
        return load_from_local_file(reader, params.input.filelist)
      #endif
    }
}

MUSEQA_END_NAMESPACE

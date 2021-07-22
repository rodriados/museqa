/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file MPI wrapper global variables and functions definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#if !defined(MUSEQA_AVOID_MPI)

#include <map>
#include <cstdint>

#include <museqa/node.hpp>
#include <museqa/utility.hpp>
#include <museqa/exception.hpp>
#include <museqa/mpi/common.hpp>
#include <museqa/mpi/lambda.hpp>
#include <museqa/mpi/communicator.hpp>

namespace museqa
{
    using namespace mpi;

    /**
     * The default global communicator instance.
     * @since 1.0
     */
    decltype(mpi::world) mpi::world;

    /**
     * Informs the current node's global rank within the instantiated MPI topology.
     * @see museqa::mpi::init
     */
    const node::id& node::rank = world.rank;

    /**
     * Informs the total number of nodes within the current MPI topology.
     * @see museqa::mpi::init
     */
    const int32_t& node::count = world.size;

    /**
     * Informs the currently active MPI operator function. This is necessary to recover
     * a wrapped function from within MPI execution.
     * @since 1.0
     */
    mpi::function::id mpi::function::active;

    /**
     * Maps a function identifier to its actual user-defined implementation. Unfortunately,
     * this is maybe the only reliable way to inject a wrapped function into the
     * operator actually called by MPI.
     * @since 1.0
     */
    std::map<mpi::function::id, void*> mpi::function::fmapper;

    /**
     * Builds a new communicator instance from a channel reference.
     * @param channel The channel reference to build a communicator from.
     * @return The new communicator instance.
     */
    auto mpi::communicator::build(reference_type channel) noexcept(museqa::unsafe) -> mpi::communicator
    {
        int32_t rank, size;

        museqa::assert<mpi::exception>(MPI_COMM_NULL != channel, mpi::error::comm, "invalid null communicator");

        mpi::check(MPI_Comm_set_errhandler(channel, MPI_ERRORS_RETURN));

        mpi::check(MPI_Comm_rank(channel, &rank));
        mpi::check(MPI_Comm_size(channel, &size));

        return {rank, size, channel};
    }

    /**
     * Splits nodes within the communicator into different channels according to
     * each node's individual selection.
     * @param color The color selected by current node.
     * @param key The key used to assigned a node id in new communicator.
     * @return The communicator obtained from the split.
     */
    auto mpi::communicator::split(int color, int key) noexcept(museqa::unsafe) -> mpi::communicator
    {
        reference_type channel;
        mpi::check(MPI_Comm_split(*this, color, (key > 0 ? key : rank), &channel));
        return build(channel);
    }

    /**
     * Duplicates the communicator with all its cached information.
     * @return The new duplicated communicator.
     */
    auto mpi::communicator::duplicate() noexcept(museqa::unsafe) -> mpi::communicator
    {
        reference_type channel;
        mpi::check(MPI_Comm_dup(*this, &channel));
        return build(channel);
    }

    /**
     * Initializes MPI's internal machinery and nodes communication.
     * @param argc The number of arguments received via command-line.
     * @param argv The program's command-line arguments.
     */
    void mpi::init(int& argc, char**& argv)
    {
        mpi::check(MPI_Init(&argc, &argv));

        auto global = mpi::communicator::global(MPI_COMM_WORLD);
        new (&world) mpi::communicator {global.duplicate()};
    }

    /**
     * Terminates MPI execution and cleans up all MPI state.
     * @see mpi::init
     */
    void mpi::finalize()
    {
        world.reset();
        mpi::check(MPI_Finalize());
    }
}

#endif

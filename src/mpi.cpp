/** 
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for the MPI wrapper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <map>
#include <vector>
#include <cstdint>

#include "mpi.hpp"
#include "node.hpp"

namespace museqa
{
    /**#@+
     * Node identification values in cluster.
     * @see mpi::init
     */
    node::id node::rank;
    int32_t node::count;
    /**#@-*/

    /**
     * The default communicator instance.
     * @see mpi::communicator
     */
    mpi::communicator mpi::world;

    /**
     * Keeps track of all generated datatypes throughout execution.
     * @since 0.1.1
     */
    std::vector<mpi::datatype::id> mpi::datatype::ref_type;

    /**
     * Keeps track of all user defined operators created during execution.
     * @since 0.1.1
     */
    std::vector<mpi::op::id> mpi::op::ref_op;

    /**
     * Maps a datatype to an user-created operator. This is necessary because
     * it is almost technically impossible to inject the operator inside the
     * wrapper without an extremelly convoluted mechanism.
     * @since 0.1.1
     */
    std::map<mpi::op::id, void *> mpi::op::op_list;

    /**
     * Informs the currently active operator. This will be useful for injecting
     * the correct operator inside the wrapper.
     * @since 0.1.1
     */
    mpi::op::id mpi::op::active;

    /**
     * Stores the last operation's status. As our collective operation functions
     * return their payloads, we need this so we can recover these operations statuses.
     * @since 0.1.1
     */
    mpi::status mpi::last_status;

    /**
     * Builds up a new communicator instance from built-in type.
     * @param comm Built-in communicator instance.
     * @return The new communicator instance.
     */
    auto mpi::communicator::build(raw_type comm) -> communicator
    {
        int rank, size;
        mpi::check(MPI_Comm_rank(comm, &rank));
        mpi::check(MPI_Comm_size(comm, &size));
        return communicator {rank, (uint32_t) size, comm};
    }

    /**
     * Splits nodes into different communicators according to selected color.
     * @param comm The original communicator to be split.
     * @param color The color selected by current node.
     * @param key The key used to assigned a node id in new communicator.
     * @return The obtained communicator from split operation.
     */
    auto mpi::communicator::split(const communicator& comm, int color, int key) -> communicator
    {
        raw_type newcomm;
        mpi::check(MPI_Comm_split(comm, color, (key > 0 ? key : comm.m_rank), &newcomm));
        return build(newcomm);
    }

    /**
     * Cleans up the resources used by communicator.
     * @param comm The communicator to be destroyed.
     */
    void mpi::communicator::free(communicator& comm)
    {
        mpi::check(MPI_Comm_free(&comm.m_raw));
        comm.m_raw = MPI_COMM_NULL;
    }

    /**
     * Initializes the cluster's communication and identifies the node in the cluster.
     * @param argc The number of arguments sent from terminal.
     * @param argv The arguments sent from terminal.
     */
    void mpi::init(int& argc, char **& argv)
    {
        mpi::check(MPI_Init(&argc, &argv));
        auto comm = mpi::communicator::build(MPI_COMM_WORLD);
        new (&world) mpi::communicator {comm};

        node::count = comm.size();
        node::rank = comm.rank();
    }

    /**
     * Finalizes all cluster communication operations between nodes.
     * @see mpi::init
     */
    void mpi::finalize()
    {
        for(datatype::id& typeref : datatype::ref_type)
            mpi::check(MPI_Type_free(&typeref));
     
        for(op::id& opref : op::ref_op)
            mpi::check(MPI_Op_free(&opref));
     
        MPI_Finalize();
    }
}

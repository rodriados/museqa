/** 
 * Multiple Sequence Alignment node file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <map>
#include <vector>

#include <mpi.hpp>
#include <node.hpp>

/**
 * The default communicator instance.
 * @see mpi::Communicator
 */
mpi::communicator::id mpi::world;

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

/**#@+
 * Node identification values in cluster.
 * @see mpi::init
 */
node::id& node::rank = mpi::world.rank;
uint32_t& node::count = mpi::world.size;
/**#@-*/

/**
 * Initializes the cluster's communication and identifies the node in the cluster.
 * @param argc The number of arguments sent from terminal.
 * @param argv The arguments sent from terminal.
 */
void mpi::init(int& argc, char **& argv)
{
    mpi::check(MPI_Init(&argc, &argv));
    mpi::world = mpi::communicator::build(MPI_COMM_WORLD);
}

/**
 * Finalizes all cluster communication operations between nodes.
 * @see mpi::init
 */
void mpi::finalize()
{
    for(mpi::datatype::id& typeref : mpi::datatype::ref_type)
        mpi::check(MPI_Type_free(&typeref));
 
    for(mpi::op::id& opref : mpi::op::ref_op)
        mpi::check(MPI_Op_free(&opref));
 
    MPI_Finalize();
}
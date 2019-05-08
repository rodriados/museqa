/** 
 * Multiple Sequence Alignment node file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <vector>

#include "mpi.hpp"
#include "node.hpp"

/**#@+
 * Node identification values in cluster.
 * @see mpi::init
 */
uint16_t node::rank = 0;
uint16_t node::size = 0;
/**#@-*/

/**
 * The default communicator instance.
 * @see mpi::Communicator
 */
mpi::Communicator mpi::world;

/**
 * Keeps track of all generated datatypes throughout execution.
 * @since 0.1.1
 */
std::vector<MPI_Datatype> mpi::datatype::dtypes;

/**
 * Keeps track of all user defined operators created during execution.
 * @since 0.1.1
 */
std::vector<MPI_Op> mpi::op::udefops;

/**
 * Initializes the cluster's communication and identifies the node in the cluster.
 * @param argc The number of arguments sent from terminal.
 * @param argv The arguments sent from terminal.
 */
void mpi::init(int& argc, char **& argv)
{
    mpi::call(MPI_Init(&argc, &argv));
    mpi::world = mpi::communicator::build(MPI_COMM_WORLD);
    
    node::rank = mpi::world.rank;
    node::size = mpi::world.size;
}

/**
 * Finalizes all cluster communication operations between nodes.
 * @see mpi::init
 */
void mpi::finalize()
{
    for(MPI_Datatype& dtype : mpi::datatype::dtypes)
        mpi::call(MPI_Type_free(&dtype));
 
    for(MPI_Op& op : mpi::op::udefops)
        mpi::call(MPI_Op_free(&op));
 
    mpi::call(MPI_Finalize());
}
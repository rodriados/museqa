/** 
 * Multiple Sequence Alignment node file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <vector>

#include "mpi.hpp"
#include "node.hpp"

uint16_t node::rank = 0;
uint16_t node::size = 0;
mpi::Communicator mpi::world;
std::vector<MPI_Datatype> mpi::datatype::dtypes;

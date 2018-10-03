/** 
 * Multiple Sequence Alignment cluster file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <vector>
#include <mpi.h>

#include "cluster.hpp"

namespace cluster
{
    /*
     * Declaring global variables.
     */
    int rank = 0;
    int size = 0;
    std::vector<MPI_Datatype> customDTypes;
};


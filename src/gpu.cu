/** @file gpu.cu
 * @brief Parallel Multiple Sequence Alignment GPU file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cuda.h>

#include "msa.h"
#include "gpu.hpp"
#include "interface.hpp"

extern clidata_t cli_data;
extern mpidata_t mpi_data;

/** @fn int gpu::count()
 * @brief Informs the number of GPU devices connected.
 * @return The number of GPU devices found.
 */
int gpu::count()
{
    int count = 0;

    __cudacheck(cudaGetDeviceCount(&count));
    return cli_data.multigpu ? count : 1;
}

/** @fn bool gpu::check()
 * @brief Checks whether at least one GPU device is connected.
 * @return Is there any GPU device connected?
 */
bool gpu::check()
{
    return count() > 0;
}

/** @fn bool gpu::multi()
 * @brief Checks whether more than one GPU devices are connected.
 * @return Are there more than one GPU devices connected?
 */
bool gpu::multi()
{
    return cli_data.multigpu && count() > 1;
}

/** @fn int gpu::assign()
 * @brief Assigns a GPU device according to the process rank.
 * @return Assigned GPU identifier.
 */
int gpu::assign()
{
    return cli_data.multigpu
        ? (mpi_data.rank - 1) % count()
        : 0;
}

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

namespace gpu
{
/** @fn int gpu::count()
 * @brief Informs the number of GPU devices connected.
 * @return The number of GPU devices found.
 */
int count()
{
    int count = 0;
    cudaGetDeviceCount(&count);

    return cli_data.multigpu ? count : 1;
}

/** @fn bool gpu::check()
 * @brief Checks whether at least one GPU device is connected.
 * @return Is there any GPU device connected?
 */
bool check()
{
    return count() > 0;
}

/** @fn bool gpu::multi()
 * @brief Checks whether more than one GPU devices are connected.
 * @return Are there more than one GPU devices connected?
 */
bool multi()
{
    return cli_data.multigpu && count() > 1;
}

/** @fn int gpu::assign()
 * @brief Assigns a GPU device according to the process rank.
 * @return Assigned GPU identifier.
 */
int assign()
{
    return cli_data.multigpu
        ? (mpi_data.rank - 1) % count()
        : 0;
}

/** @fn unsigned gpu::align(unsigned)
 * @brief Calculates the alignment for a given size.
 * @param size The size to be byte-aligned.
 * @return The new aligned size.
 */
unsigned align(unsigned size)
{
    return (size / NV_ALIGN_BYTES + !!(size % NV_ALIGN_BYTES)) * NV_ALIGN_BYTES;
}

}
/** 
 * Multiple Sequence Alignment device tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstring>
#include <cuda.h>

#include "msa.hpp"
#include "cli.hpp"
#include "device.cuh"

int deviceId = -1;
int deviceCount = -1;
DeviceProperties deviceProps;

/**
 * Informs the identifier of currently selected device.
 * @return The currently selected device identifier.
 */
int device::get()
{
    if(deviceId < 0) {
        cudacall(cudaGetDevice(&deviceId));
        cudacall(cudaGetDeviceProperties(&deviceProps, deviceId));
    }

    return deviceId;
}

/**
 * Informs the number of devices available.
 * @return The number of compute-capable devices.
 */
int device::count()
{
    if(deviceCount < 0) {
        cudacall(cudaGetDeviceCount(&deviceCount));
    }

    return deviceCount;
}

/**
 * Checks whether at least one device is available.
 * @return Is there at least one compute-capable device available?
 */
bool device::exists()
{
    return device::count() > 0;
}

/**
 * Selects the compute-capable device to be used.
 * @return The selected device identification.
 */
int device::select()
{
    const int devices = cli.has("multigpu")
        ? device::count()
        : device::exists();

    cudacall(cudaSetDevice(cluster::rank % devices));
    return device::get();
}

/**
 * Gathers information about chosen device.
 * @return All information available about chosen device.
 */
const DeviceProperties& device::properties()
{
    device::get();
    return deviceProps;
}

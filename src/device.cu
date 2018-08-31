/** 
 * Multiple Sequence Alignment device tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstring>
#include <cuda.h>

#include "msa.hpp"
#include "input.hpp"
#include "device.cuh"

/**
 * Informs the number of devices available.
 * @return The number of compute-capable devices.
 */
int device::count()
{
    int devices;

    cudacall(cudaGetDeviceCount(&devices));
    return devices;
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
    const int devices = cmd.has("multigpu")
        ? device::count()
        : device::exists();
    int id = (cluster::rank - 1) % devices;

    cudacall(cudaSetDevice(id));
    return id;
}

/**
 * Gathers information about chosen device.
 * @return All information available about chosen device.
 */
const DeviceProperties& device::properties()
{
    static DeviceProperties props;
    int id = device::select();

    cudacall(cudaGetDeviceProperties(&props, id));
    return props;
}

/**
 * Creates an error instance for no device.
 * @return The error instance.
 */
const DeviceError DeviceError::noGPU()
{
    return DeviceError("No compatible GPU has been found.");
}

/**
 * Creates an error instance for execution error.
 * @param msg The acquired error message.
 * @return The error instance.
 */
 const DeviceError DeviceError::execution(const char *msg)
 {
    return DeviceError(msg);
 }
 
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

/*
 * Declaring global variables.
 */
static int devices = 0;
static DeviceProperties d_props;

/**
 * Informs the number of devices available.
 * @return The number of compute-capable devices.
 */
int device::count()
{
    if(!devices && cudaGetDeviceCount(&devices) == cudaSuccess)
        devices = !cmd.has("multigpu")
            ? (devices > 0)
            : devices;

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
 * @param device_id The device with priority to be chosen.
 * @return The selected device identification.
 */
int device::select(int device_id)
{
    cudacall(cudaSetDevice(
        (device_id < 0 || device_id > device::count())
            ? (cluster::rank - 1) % device::count()
            : device_id
    ));

    cudacall(cudaGetDevice(&device_id));
    return device_id;
}

/**
 * Gathers information about chosen device.
 * @return All information available about chosen device.
 */
const DeviceProperties& device::properties()
{
    int device_id;

    cudacall(cudaGetDevice(&device_id));
    cudacall(cudaGetDeviceProperties(&d_props, device_id));
    return d_props;
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
 
/** 
 * Multiple Sequence Alignment device tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <cstring>
#include <cuda.h>

#include "device.cuh"

/*
 * Declaring global variables.
 */
static int devices = 0;
static DeviceProperties devprops;

/**
 * Informs the number of devices available.
 * @return The number of compute-capable devices.
 */
int Device::count()
{
    return devices || cudaGetDeviceCount(&devices) == cudaSuccess
        ? devices
        : 0;
}

/**
 * Checks whether at least one device is available.
 * @return Is there at least one compute-capable device available?
 */
bool Device::check()
{
    return !!Device::count();
}

/**
 * Gathers information about chosen device.
 * @return All information available about chosen device.
 */
const DeviceProperties& Device::properties()
{
    int id;

    __cudacall(cudaGetDevice(&id));
    __cudacall(cudaGetDeviceProperties(&devprops, id));

    return devprops;
}
/** 
 * Multiple Sequence Alignment device tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018 Rodrigo Siqueira
 */
#include <vector>
#include <cuda.h>

#include "device.cuh"

/*
 * Declaring global variables.
 */
static int deviceCount = 0;

/**
 * Informs the number of devices available.
 * @return The number of compute-capable devices.
 */
__host__
int Device::count()
{
    return (deviceCount != 0 || cudaGetDeviceCount(&deviceCount) == cudaSuccess)
        ? deviceCount
        : 0;
}

/**
 * Checks whether at least one device is available.
 * @return Is there at least one compute-capable device available?
 */
__host__
bool Device::check()
{
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return deviceCount && error == cudaSuccess;
}

/**
 * Informs the device currently selected.
 * @return The identification of currently chosen device.
 */
__host__
int Device::get()
{
    int id;

    return cudaGetDevice(&id) == cudaSuccess
        ? id
        : -1;
}

/**
 * Chooses a device to be selected from now on.
 * @param id The identification of chosen device.
 * @return Was the operation successful?
 */
__host__
bool Device::set(int id)
{
    return cudaSetDevice(id) == cudaSuccess;
}

/**
 * Resets a device so no memory is allocated.
 * @param id The identification of chosen device.
 * @return Was the operation successful?
 */
__host__
bool Device::reset(int id)
{
    int old = Device::get();
    
    return (id == ~0 || Device::set(id))
        && cudaDeviceReset() == cudaSuccess
        && Device::set(old);
}

/**
 * Gathers information about chosen device.
 * @param id The identification of chosen device.
 * @return All information available about chosen device.
 */
__host__
DeviceProperty Device::properties(int id)
{
    DeviceProperty prop;

    __cudacall(cudaGetDeviceProperties(&prop, id == ~0 ? Device::get() : id));
    return prop;
}
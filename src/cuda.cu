/** 
 * Multiple Sequence Alignment CUDA tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <string>

#include "cuda.cuh"

using namespace cuda;
using namespace cuda::device;

/**
 * Gets the number of devices available.
 * @return The number of devices or runtime error.
 */
int getCount()
{
    int devices;
    call(cudaGetDeviceCount(&devices));
    return devices;
}

/**
 * Gets the current device id.
 * @return The device id or runtime error.
 */
Device getCurrent()
{
    Device device;
    call(cudaGetDevice(&device));
    return device;
}

/**
 * Sets the current device to given id.
 * @param device The device to be used.
 */
void setCurrent(const Device& device)
{
    call(cudaSetDevice(device));
}

/**
 * Retrieves information and properties about the chosen device.
 * @param device The device of which properties will be retrieved.
 * @return The device properties.
 */
Properties getProperties(const Device& device)
{
    Properties props;
    call(cudaGetDeviceProperties(&props, device));
    return props;
}

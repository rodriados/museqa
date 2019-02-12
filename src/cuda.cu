/** 
 * Multiple Sequence Alignment CUDA tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <cuda.h>
#include <string>

#include "cuda.cuh"

/**
 * Obtain a brief textual explanation for a specified kind of CUDA Runtime API status
 * or error code.
 * @param status The error status obtained.
 * @return The error description.
 */
std::string cuda::status::describe(cuda::Status status) noexcept
{
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

/**
 * Gets the number of devices available.
 * @return The number of devices or runtime error.
 */
int cuda::device::getCount()
{
    int devices;
    cuda::call(cudaGetDeviceCount(&devices));
    return devices;
}

/**
 * Gets the current device id.
 * @return The device id or runtime error.
 */
cuda::Device cuda::device::getCurrent()
{
    cuda::Device device;
    cuda::call(cudaGetDevice(&device));
    return device;
}

/**
 * Sets the current device to given id.
 * @param device The device to be used.
 */
void cuda::device::setCurrent(const cuda::Device& device)
{
    cuda::call(cudaSetDevice(device));
}

/**
 * Retrieves information and properties about the chosen device.
 * @param device The device of which properties will be retrieved.
 * @return The device properties.
 */
cuda::device::Properties cuda::device::getProperties(const cuda::Device& device)
{
    cuda::device::Properties props;
    cuda::call(cudaGetDeviceProperties(&props, device));
    return props;
}

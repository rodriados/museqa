/** 
 * Multiple Sequence Alignment CUDA tools file.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-2019 Rodrigo Siqueira
 */
#include <cuda.h>
#include <string>

#include <cuda.cuh>
#include <allocator.hpp>

namespace msa
{
    /**
     * The allocator instance for reserving and managing pointers of memory regions
     * allocated in device's global memory space.
     * @since 0.1.1
     */
    allocator cuda::allocator::device = {
        [](void **ptr, size_t size, size_t n) { cuda::check(cudaMalloc(ptr, size * n)); }
    ,   [](void *ptr) { cuda::check(cudaFree(ptr)); }
    };

    /**
     * The allocator instance for reserving and managing pointers to pinned host-side
     * memory regions. Pinned memory is unpaginated and thus can be accessed faster
     * by the device's internal instructions.
     * @since 0.1.1
     */
    allocator cuda::allocator::pinned = {
        [](void **ptr, size_t size, size_t n) { cuda::check(cudaMallocHost(ptr, size * n)); }
    ,   [](void *ptr) { cuda::check(cudaFreeHost(ptr)); }
    };

    /**
     * Obtain a brief textual explanation for a specified kind of CUDA Runtime 
     * API status or error code.
     * @param code The error code to be described.
     * @return The error description.
     */
    std::string cuda::status::describe(cuda::status_code code) noexcept
    {
        return cudaGetErrorString(static_cast<cudaError_t>(code));
    }

    /**
     * Gets the number of devices available.
     * @return The number of devices or runtime error.
     */
    auto cuda::device::count() -> size_t
    {
        int devices;
        cuda::check(cudaGetDeviceCount(&devices));
        return static_cast<size_t>(devices);
    }

    /**
     * Gets the current device id.
     * @return The device id or runtime error.
     */
    auto cuda::device::current() -> cuda::device::id
    {
        int device;
        cuda::check(cudaGetDevice(&device));
        return device;
    }

    /**
     * Sets the current device to given id.
     * @param device The device to be used.
     */
    auto cuda::device::select(const cuda::device::id& device) -> void
    {
        cuda::check(cudaSetDevice(device));
    }

    /**
     * Retrieves information and properties about the chosen device.
     * @param device The device of which properties will be retrieved.
     * @return The device properties.
     */
    auto cuda::device::properties(const cuda::device::id& device) -> cuda::device::props
    {
        cuda::device::props props;
        cuda::check(cudaGetDeviceProperties(&props, device));
        return props;
    }
}
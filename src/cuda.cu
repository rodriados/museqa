/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implementation for CUDA functions and structures wrappers.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#include <cuda.h>
#include <limits>
#include <string>

#include "cuda.cuh"
#include "utils.hpp"
#include "allocator.hpp"

namespace
{
    namespace current
    {
        using namespace museqa;

        /**
         * Stores the ID of the compute-capable device currently selected.
         * @since 0.1.1
         */
        cuda::device::id id = std::numeric_limits<cuda::word>::max();

        /**
         * Stores the properties of the currently selected compute-capable device.
         * @since 0.1.1
         */
        cuda::device::property property;
    }
}

namespace museqa
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
    std::string cuda::status::describe(cuda::status::code code) noexcept
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
    void cuda::device::select(cuda::device::id device)
    {
        cuda::check(cudaSetDevice(::current::id = device));
        ::current::property = cuda::device::properties(device);
    }

    /**
     * Retrieves information and properties about the currently selected device.
     * @return The currently selected device properties.
     */
    auto cuda::device::properties() -> cuda::device::property
    {
        if(::current::id == std::numeric_limits<cuda::word>::max())
            cuda::device::select(cuda::device::init);
        return ::current::property;
    }

    /**
     * Retrieves information and properties about the chosen device.
     * @param device The device of which properties will be retrieved.
     * @return The device properties.
     */
    auto cuda::device::properties(cuda::device::id device) -> cuda::device::property
    {
        cuda::device::property property;
        cuda::check(cudaGetDeviceProperties(&property, device));
        return property;
    }

    /**
     * Informs the total number of blocks supported by a single grid on the device.
     * @param needed The number of blocks needed for a specific computation.
     * @return The maximum number of blocks available.
     */
    auto cuda::device::blocks(size_t needed) -> size_t
    {
        return utils::min<size_t>(needed, ::current::property.maxGridSize[0]);
    }

    /**
     * Informs the total number of threads supported by a single grid on the device.
     * @param needed The number of threads needed for a specific computation.
     * @return The maximum number of threads available.
     */
    auto cuda::device::threads(size_t needed) -> size_t
    {
        return utils::min<size_t>(needed, ::current::property.maxThreadsDim[0]);
    }
}

/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA wrapper global variables and functions definitions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#if !defined(MUSEQA_AVOID_CUDA)

#include <cuda.h>

#include <string>
#include <cstdint>

#include <museqa/utility.hpp>

#include <museqa/cuda/common.cuh>
#include <museqa/cuda/device.cuh>
#include <museqa/cuda/stream.cuh>
#include <museqa/cuda/event.cuh>

namespace museqa
{
    using namespace cuda;

    /**
     * Retrieves a brief textual explanation for a specified kind of CUDA runtime
     * API status or error code.
     * @param code The error code to be described.
     * @return The error description.
     */
    auto cuda::error::describe(error::code code) noexcept -> std::string
    {
        return cudaGetErrorString(code);
    }

    /**
     * Retrieves the total amount of memory available for immediate usage within
     * the currently active device.
     * @return The amount of memory available in the current device.
     */
    auto cuda::device::memory::available() noexcept(!safe) -> size_t
    {
        size_t available, total;
        cuda::check(cudaMemGetInfo(&available, &total));
        return available;
    }

    /**
     * Retrieves the total amount of memory available for immediate usage within
     * the given target device.
     * @param target The device to be introspected.
     * @return The amount of memory available in the selected device.
     */
    auto cuda::device::memory::available(device::id target) noexcept(!safe) -> size_t
    {
        cuda::device::current::scope temporary {target};
        return cuda::device::memory::available();
    }

    /**
     * Retrieves the total amount of global memory present in the currently active
     * device's hardware, independently whether this memory is available or not.
     * @return The total amount of global memory within the current device.
     */
    auto cuda::device::memory::total() noexcept(!safe) -> size_t
    {
        size_t available, total;
        cuda::check(cudaMemGetInfo(&available, &total));
        return total;
    }

    /**
     * Retrieves the total amount of global memory present in the given device's
     * hardware, independently whether this memory is available or not.
     * @param target The device to be introspected.
     * @return The total amount of global memory within the selected device.
     */
    auto cuda::device::memory::total(device::id target) noexcept(!safe) -> size_t
    {
        cuda::device::current::scope temporary {target};
        return cuda::device::memory::total();
    }

    /**
     * Retrieves the currently active device.
     * @return The current active device.
     */
    __host__ __device__ auto cuda::device::current::get() noexcept(!safe) -> cuda::device::id
    {
        cuda::device::id device;
        cuda::check(cudaGetDevice(&device));
        return device;
    }

    /**
     * Changes the currently active device to the given one.
     * @param target The new device to be active.
     */
    void cuda::device::current::set(cuda::device::id target) noexcept(!safe)
    {
        cuda::check(cudaSetDevice(target));
    }

    /**
     * Changes the currently active device and returns the one previously active.
     * @param target The new device to be active.
     * @return The previously active device.
     */
    auto cuda::device::current::scope::replace(cuda::device::id target) noexcept(!safe) -> cuda::device::id
    {
        auto previous = cuda::device::current::get();
        cuda::device::current::set(target);
        return previous;
    }

    /**
     * Retrieves the total number of compute-capable devices currently available
     * and directly accessible to the current process.
     * @return The total number of directly accessible devices.
     */
    __host__ __device__ auto cuda::device::count() noexcept(!safe) -> size_t
    {
        int total = 0;
        cuda::check(cudaGetDeviceCount(&total));
        return static_cast<size_t>(total);
    }

    /**
     * Checks whether the given stream has finished executing its queue.
     * @param stream The stream to check completion of.
     * @return Has the stream completed all its tasks?
     */
    bool cuda::stream::ready(const cuda::stream& stream) noexcept(!safe)
    {
        return cuda::ready(cudaStreamQuery(stream));
    }

    /**
     * Blocks execution and waits for all stream tasks to complete.
     * @param stream The stream to be synchronized.
     * @see museqa::cuda::synchronize
     */
    void cuda::stream::synchronize(const cuda::stream& stream) noexcept(!safe)
    {
        cuda::device::current::scope temporary {stream.m_device};
        cuda::check(cudaStreamSynchronize(stream));
    }

    /**
     * Blocks stream execution and waits for the given event to be fired.
     * The event does not need to be on the same device as the stream, thus
     * allowing synchronization between different devices.
     * @param stream The stream to wait on given the event.
     * @param event The event to waited on.
     * @see museqa::cuda::event
     */
    void cuda::stream::wait(const cuda::stream& stream, cuda::event::id event) noexcept(!safe)
    {
        cuda::check(cudaStreamWaitEvent(stream, event, 0u));
    }

    /**
     * Checks whether event's recorded stream has completed its the work.
     * @param event The event to check completion of.
     * @return Has all recorded work been completed?
     */
    bool cuda::event::ready(const cuda::event& event) noexcept(!safe)
    {
        return cuda::ready(cudaEventQuery(event));
    }

    /**
     * Captures the contents of a stream at the time of this call.
     * @param event The target event to capture the given stream.
     * @param stream The stream to have its contents captured by the event.
     * @note Both the event and the stream must be in the same device.
     */
    void cuda::event::record(cuda::event& event, cuda::stream::id stream) noexcept(!safe)
    {
        cuda::check(cudaEventRecord(event, stream));
    }

    /**
     * Waits until the completion of all work currently captured by event.
     * @param event The event to be synchronized.
     * @see museqa::cuda::synchronize
     */
    void cuda::event::synchronize(const cuda::event& event) noexcept(!safe)
    {
        cuda::device::current::scope temporary {event.m_device};
        cuda::check(cudaEventSynchronize(event));
    }
}

#endif

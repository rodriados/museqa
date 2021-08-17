/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA device memory pointers, buffers and helper functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2018-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <cuda.h>

#include <vector>
#include <cstdint>
#include <utility>

#include <museqa/utility.hpp>
#include <museqa/memory/buffer.hpp>
#include <museqa/memory/allocator.hpp>

#include <museqa/cuda/common.cuh>
#include <museqa/cuda/device.cuh>
#include <museqa/cuda/stream.cuh>

namespace museqa
{
    namespace cuda
    {
        namespace memory
        {
          #if defined(MUSEQA_COMPILER_NVCC)
            /**
             * Asynchronously copies data between the source and target pointers,
             * potentially between the host and a compute-capable device.
             * @note The call may return before the copy is complete.
             * @tparam T The pointer type to copied.
             * @param stream The stream to associate the copy with.
             * @param target The copy's target pointer.
             * @param source The copy's source pointer.
             * @param count The total number of elements to be copied.
             */
            template <typename T>
            __host__ __device__ inline void copy(
                cuda::stream::id stream
              , T *target
              , const T *source
              , size_t count = 1
            ) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemcpyAsync(target, source, size * count, cudaMemcpyDefault, stream));
            }

            /**
             * Asynchronously copies data between two buffers, potentially pointing
             * to memory regions in the host or different compute-capable devices.
             * @note The call may return before the copy is complete.
             * @tparam T The buffer's elements type.
             * @param stream The stream to associate the copy with.
             * @param target The copy's target buffer.
             * @param source The copy's source buffer.
             */
            template <typename T>
            __host__ __device__ inline void copy(
                cuda::stream::id stream
              , museqa::memory::buffer<T>& target
              , const museqa::memory::buffer<T>& source
            ) noexcept(!safe)
            {
                const auto count = utility::min(target.capacity(), source.capacity());
                cuda::memory::copy(stream, target.begin(), source.begin(), count);
            }

            /**
             * Synchronously copies data between the source and target pointers,
             * potentially between the host and a compute-capable device.
             * @tparam T The pointer type to copied.
             * @param target The copy's target pointer.
             * @param source The copy's source pointer.
             * @param count The total number of elements to be copied.
             */
            template <typename T>
            inline void copy(
                T *target
              , const T *source
              , size_t count = 1
            ) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemcpy(target, source, size * count, cudaMemcpyDefault));
            }

            /**
             * Synchronously copies data between two buffers, potentially pointing
             * to memory regions in the host or different compute-capable devices.
             * @tparam T The buffer's elements type.
             * @param target The copy's target buffer.
             * @param source The copy's source buffer.
             */
            template <typename T>
            inline void copy(
                museqa::memory::buffer<T>& target
              , const museqa::memory::buffer<T>& source
            ) noexcept(!safe)
            {
                const auto count = utility::min(target.capacity(), source.capacity());
                cuda::memory::copy(target.begin(), source.begin(), count);
            }

            /**
             * Asynchronously initializes or sets device memory to a value.
             * @note The call may return before the initiazation is complete.
             * @tparam T The device memory pointer type.
             * @param stream The stream to associate the operation with.
             * @param target The device memory pointer to be initialized.
             * @param byte The value to set each byte of the target memory region.
             * @param count The total number of elements to be initialized.
             */
            template <typename T>
            __host__ __device__ inline void set(
                cuda::stream::id stream
              , T *target
              , uint8_t byte
              , size_t count = 1
            ) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemsetAsync(target, (int) byte, size * count, stream));
            }

            /**
             * Synchronously initializes or sets device memory to a value.
             * @tparam T The device memory pointer type.
             * @param target The device memory pointer to be initialized.
             * @param byte The value to set each byte of the target memory region.
             * @param count The total number of elements to be initialized.
             */
            template <typename T>
            inline void set(T *target, uint8_t byte, size_t count = 1) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemset(target, (int) byte, size * count));
            }

            /**
             * Asynchronously initializes the given device memory region to zero.
             * @note The call may return before the initiazation is complete.
             * @tparam T The device memory pointer type.
             * @param stream The stream to associate the operation with.
             * @param target The device memory pointer to be initialized.
             * @param count The total number of elements to be initialized.
             */
            template <typename T>
            __host__ __device__ inline void zero(
                cuda::stream::id stream
              , T *target
              , size_t count = 1
            ) noexcept(!safe)
            {
                cuda::memory::set(stream, target, 0, count);
            }

            /**
             * Asynchronously initializes the given memory buffer to zero.
             * @note The call may return before the initiazation is complete.
             * @tparam T The buffer's elements type.
             * @param stream The stream to associate the operation with.
             * @param target The buffer to be initialized.
             */
            template <typename T>
            __host__ __device__ inline void zero(
                cuda::stream::id stream
              , museqa::memory::buffer<T>& target
            ) noexcept(!safe)
            {
                cuda::memory::zero(stream, target.begin(), target.capacity());
            }

            /**
             * Synchronously initializes the given device memory region to zero.
             * @tparam T The device memory pointer type.
             * @param target The device memory pointer to be initialized.
             * @param count The total number of elements to be initialized.
             */
            template <typename T>
            inline void zero(T *target, size_t count = 1) noexcept(!safe)
            {
                cuda::memory::set(target, 0, count);
            }

            /**
             * Synchronously initializes the given memory buffer to zero.
             * @tparam T The buffer's elements type.
             * @param target The buffer to be initialized.
             */
            template <typename T>
            inline void zero(museqa::memory::buffer<T>& target) noexcept(!safe)
            {
                cuda::memory::zero(target.begin(), target.capacity());
            }

            /**
             * Enumeration of flags available for registering a memory region for
             * pinning, page-locking it and allowing it to be directly accessed
             * by a compute-capable device without copies.
             * @since 1.0
             */
            enum : uint32_t
            {
                portable  = cudaHostRegisterPortable
              , mapped    = cudaHostRegisterMapped
              , io_memory = cudaHostRegisterIoMemory
              // , read_only = cudaHostRegisterReadOnly
            };

            /**
             * Page-locks and pins the given memory region, allowing it to be directly
             * accessed by a compute-capable device without copies.
             * @tparam T The pointer type to be pinned and page-locked.
             * @param ptr The pointer to the start of the region to be pinned.
             * @param count The total number of elements in the region to be pinned.
             * @param flags The flags for the pinning operation.
             */
            template <typename T>
            inline void pin(T *ptr, size_t count = 1, uint32_t flags = 0) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaHostRegister(ptr, size * count, flags));
            }

            /**
             * Page-locks and pins the given memory buffer, allowing it to be directly
             * accessed by a compute-capable device without copies.
             * @tparam T The buffer's elements type.
             * @param buffer The buffer to be page-locked and pinned.
             * @param flags The flags for the pinning operation.
             */
            template <typename T>
            inline void pin(museqa::memory::buffer<T>& buffer, uint32_t flags = 0) noexcept(!safe)
            {
                cuda::memory::pin(buffer.begin(), buffer.capacity(), flags);
            }

            /**
             * Ends the memory pinning and page-unlocks the given memory region,
             * thus prohibiting direct access by a compute-capable device.
             * @tparam T The pointer type to be unpinned and page-unlocked.
             * @param ptr The pointer to the start of region to be unpinned.
             */
            template <typename T>
            inline void unpin(T *ptr) noexcept(!safe)
            {
                cuda::check(cudaHostUnregister(ptr));
            }

            /**
             * Ends the memory pinning and page-unlocks the given memory buffer,
             * thus prohibiting direct access by a compute-capable device.
             * @tparam T The buffer's elements type.
             * @param buffer The buffer to be page-unlocked and unpinned.
             */
            template <typename T>
            inline void unpin(museqa::memory::buffer<T>& buffer) noexcept(!safe)
            {
                cuda::memory::unpin(buffer.begin());
            }

            /**
             * Enumeration of managed memory usage pattern advices. Memory advices
             * allow the CUDA runtime optimize the usage of host memory within a
             * device, applying heuristics to avoid page misses, for instance.
             * @since 1.0
             */
            enum advice
            {
                read_mostly        = cudaMemAdviseSetReadMostly
              , preferred_location = cudaMemAdviseSetPreferredLocation
              , accessed_by        = cudaMemAdviseSetAccessedBy
            };

            /**
             * Advise about the usage pattern of the given memory region.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The memory region's contents type.
             * @param advice The advice to apply for the given region.
             * @param ptr The memory region pointer to advise for.
             * @param count The total number of elements in the region.
             * @param enable Should the given advice be enabled?
             * @param device The device into which the advice is applied.
             */
            template <typename T>
            inline void advise(
                cuda::memory::advice advice
              , const T *ptr
              , size_t count = 1
              , bool enable = true
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemAdvise(ptr, size * count, (cudaMemoryAdvise)(advice + !enable), device));
            }

            /**
             * Advise about the usage pattern of the given memory buffer.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The buffer's elements type.
             * @param advice The advice to apply for the given buffer.
             * @param buffer The buffer to apply the memory advice for.
             * @param enable Should the given advice be enabled?
             * @param device The device into which the advice is applied.
             */
            template <typename T>
            inline void advise(
                cuda::memory::advice advice
              , const museqa::memory::buffer<T>& buffer
              , bool enable = true
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                cuda::memory::advise(advice, buffer.begin(), buffer.capacity(), enable, device);
            }

            /**
             * Asynchronously prefetches a memory region to the specified device.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The memory region's contents type.
             * @param stream The stream to associate the operation with.
             * @param ptr The memory region pointer to be prefetched.
             * @param count The total number of elements to be prefetched.
             * @param device The device to prefetch the region to.
             */
            template <typename T>
            inline void prefetch(
                cuda::stream::id stream
              , const T *ptr
              , size_t count = 1
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                constexpr size_t size = sizeof(typename std::conditional<std::is_void<T>::value, char, T>::type);
                cuda::check(cudaMemPrefetchAsync(ptr, size * count, device, stream));
            }

            /**
             * Asynchronously prefetches a memory buffer to the specified device.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The buffer's elements type.
             * @param stream The stream to associate the operation with.
             * @param buffer The buffer to be prefetched to the target device.
             * @param device The device to prefetch the region to.
             */
            template <typename T>
            inline void prefetch(
                cuda::stream::id stream
              , const museqa::memory::buffer<T>& buffer
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                cuda::memory::prefetch(stream, buffer.begin(), buffer.capacity(), device);
            }

            /**
             * Synchronously prefetches a memory region to the specified device.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The memory region's contents type.
             * @param ptr The memory region pointer to be prefetched.
             * @param count The total number of elements to be prefetched.
             * @param device The device to prefetch the region to.
             */
            template <typename T>
            inline void prefetch(
                const T *ptr
              , size_t count = 1
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                cuda::memory::prefetch(cuda::stream::default_stream, ptr, count, device);
            }

            /**
             * Synchronously prefetches a memory buffer to the specified device.
             * @note The given pointer must be allocated as managed memory.
             * @tparam The buffer's elements type.
             * @param buffer The buffer to be prefetched to the target device.
             * @param device The device to prefetch the region to.
             */
            template <typename T>
            inline void prefetch(
                const museqa::memory::buffer<T>& buffer
              , cuda::device::id device = cuda::device::default_device
            ) noexcept(!safe)
            {
                cuda::memory::prefetch(buffer.begin(), buffer.capacity(), device);
            }
          #endif
        }
    }

  #if defined(MUSEQA_COMPILER_NVCC)
    namespace factory
    {
        namespace cuda
        {
            namespace device
            {
                /**
                 * Creates an allocator for a generic pointer in the current device's
                 * global memory space. This allocator does not initialize the allocated
                 * space nor does it call any type's default constructor.
                 * @return The new device global memory allocator.
                 */
                inline auto allocator() noexcept -> memory::allocator
                {
                    return memory::allocator {
                        [](void **ptr, size_t sz, size_t n) { museqa::cuda::check(cudaMalloc(ptr, n * sz)); }
                      , [](void *ptr) { museqa::cuda::check(cudaFree(ptr)); }
                    };
                }
            }

            namespace host
            {
                /**
                 * Creates an allocator for a generic pointer in an unmapped and
                 * unpaginated region of the host's memory, which can be accessed
                 * faster by the device's internal instructions.
                 * @return The new pinned memory allocator.
                 */
                inline auto allocator() noexcept -> memory::allocator
                {
                    return memory::allocator {
                        [](void **ptr, size_t sz, size_t n) { museqa::cuda::check(cudaMallocHost(ptr, n * sz)); }
                      , [](void *ptr) { museqa::cuda::check(cudaFreeHost(ptr)); }
                    };
                }
            }

            namespace managed
            {
                /**
                 * Creates an allocator for a generic pointer that will be automatically
                 * managed by CUDA's unified memory system.
                 * @return The new memory allocator instance.
                 */
                inline auto allocator() noexcept -> memory::allocator
                {
                    return memory::allocator {
                        [](void **ptr, size_t sz, size_t n) { museqa::cuda::check(cudaMallocManaged(ptr, n * sz)); }
                      , [](void *ptr) { museqa::cuda::check(cudaFree(ptr)); }
                    };
                }
            }

            using namespace device;
        }
    }
  #endif
}

#endif

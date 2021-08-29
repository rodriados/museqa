/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file CUDA kernel configurations and utility functions.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2021-present Rodrigo Siqueira
 */
#pragma once

#if !defined(MUSEQA_AVOID_CUDA)

#include <cuda.h>

#include <cstdint>
#include <utility>

#include <museqa/environment.h>

#include <museqa/utility.hpp>
#include <museqa/utility/functor.hpp>
#include <museqa/geometry/coordinate.hpp>

#include <museqa/cuda/common.cuh>
#include <museqa/cuda/stream.cuh>

namespace museqa
{
    namespace cuda
    {
        namespace kernel
        {
            /**
             * Represents the raw kernel-function type. Although quite generic,
             * every kernel must be void-declared, and have no return type.
             * @tparam P The kernel's parameters' types.
             * @since 1.0
             */
            template <typename ...P>
            using raw = void(P...);

            /**
             * Represents a functor that wraps a kernel. The functor must be manually
             * unwrapped in order to call the kernel.
             * @tparam P The kernel's parameters' types.
             * @since 1.0
             */
            template <typename ...P>
            using functor = utility::functor<kernel::raw<P...>>;

          #if defined(MUSEQA_COMPILER_NVCC)
            namespace config
            {
                /**
                 * In some GPU micro-architectures, it's possible to have the multiprocessors
                 * change the balance in the allocation of L1-cache-like resources
                 * between actual L1 cache and shared memory. This enumerates all
                 * possible choices for cache balance allocation.
                 * @since 1.0
                 */
                enum class cache : std::underlying_type<cudaFuncCache>::type
                {
                    none    = cudaFuncCachePreferNone
                  , equal   = cudaFuncCachePreferEqual
                  , shared  = cudaFuncCachePreferShared
                  , l1      = cudaFuncCachePreferL1
                };

                /**
                 * On devices with configurable shared memory banks, it's possible
                 * to force all launches of a device function to have one of the
                 * following shared memory bank size configuration.
                 * @since 1.0
                 */
                enum class shared_memory : std::underlying_type<cudaSharedMemConfig>::type
                {
                    four_byte   = cudaSharedMemBankSizeFourByte
                  , eight_byte  = cudaSharedMemBankSizeEightByte
                };

                /**
                 * Sets the cache configuration to the given kernel.
                 * @tparam P The kernel's function parameter types.
                 * @param kernel The kernel to set the configuration to.
                 * @param choice The chosen cache configuration to apply.
                 */
                template <typename ...P>
                inline void apply(kernel::raw<P...> *kernel, config::cache choice) noexcept(!safe)
                {
                    const auto f = reinterpret_cast<const void*>(kernel);
                    cuda::check(cudaFuncSetCacheConfig(f, static_cast<cudaFuncCache>(choice)));
                }

                /**
                 * Sets the shared memory configuration to the given kernel.
                 * @tparam P The kernel's function parameter types.
                 * @param kernel The kernel to set the configuration to.
                 * @param choice The chosen shared memory configuration to apply.
                 */
                template <typename ...P>
                inline void apply(kernel::raw<P...> *kernel, config::shared_memory choice) noexcept(!safe)
                {
                    const auto f = reinterpret_cast<const void*>(kernel);
                    cuda::check(cudaFuncSetSharedMemConfig(f, static_cast<cudaSharedMemConfig>(choice)));
                }
            }
          #endif
        }

        namespace launch
        {
            /**
             * Represents a kernel block or thread grid which specifies the total
             * amount of device threads to allocate for the launching kernel.
             * @since 1.0
             */
            class grid : public geometry::coordinate<3, uint32_t>
            {
              private:
                typedef uint32_t dimension_type;
                typedef geometry::coordinate<3, dimension_type> underlying_type;

              public:
                __host__ __device__ inline constexpr grid(const grid&) noexcept = default;
                __host__ __device__ inline constexpr grid(grid&&) noexcept = default;

                /**
                 * Instantiates a new grid with the given dimensions.
                 * @param x The grid's first dimension value.
                 * @param y The grid's second dimension value.
                 * @param z The grid's third dimension value.
                 */
                __host__ __device__ inline constexpr grid(
                    dimension_type x = 1
                  , dimension_type y = 1
                  , dimension_type z = 1
                ) noexcept
                  : underlying_type {x, y, z}
                {}

              #if defined(MUSEQA_COMPILER_NVCC)
                /**
                 * Instantiates a new grid from CUDA's native dimension type.
                 * @param value The value to create a new grid from.
                 */
                __host__ __device__ inline constexpr grid(const dim3& value) noexcept
                  : grid {value.x, value.y, value.z}
                {}
              #endif

                __host__ __device__ inline grid& operator=(const grid&) noexcept = default;
                __host__ __device__ inline grid& operator=(grid&&) noexcept = default;

              #if defined(MUSEQA_COMPILER_NVCC)
                /**
                 * Converts the grid into CUDA's native dimension type, so that
                 * a grid can be seamlessly used with native CUDA functions.
                 * @return The resulting CUDA native dimension instance.
                 */
                __host__ __device__ inline operator dim3() const noexcept
                {
                    using type = decltype(dim3::x);
                    return dim3 {(type) this->x, (type) this->y, (type) this->z};
                }
              #endif

                __host__ __device__ inline static constexpr grid cube(dimension_type x) noexcept;
                __host__ __device__ inline static constexpr grid square(dimension_type x) noexcept;
                __host__ __device__ inline static constexpr grid line(dimension_type x) noexcept;
                __host__ __device__ inline static constexpr grid point() noexcept;
            };

            /**
             * Holds the necessary parameters to launch a CUDA kernel. These parameters
             * describe the number and topology of blocks and threads in each block
             * that will be available as processing and memory resources to a kernel
             * to execute on a stream of a device.
             * @since 1.0
             */
            struct config
            {
                cuda::launch::grid blocks;      /// The kernel's block grid topology.
                cuda::launch::grid threads;     /// The kernel's threads within block topology.
                uint32_t memory = 0;            /// The number of shared memory bytes per block.

                __host__ __device__ inline constexpr config() noexcept = delete;
                __host__ __device__ inline constexpr config(const config&) noexcept = default;
                __host__ __device__ inline constexpr config(config&&) noexcept = default;

                /**
                 * Instantiates a new kernel launch configuration from parameters.
                 * @param blocks The kernel's block grid topology.
                 * @param threads The kernel's threads within block topology.
                 * @param memory The amount of shared memory bytes per block.
                 */
                __host__ __device__ inline constexpr config(
                    const cuda::launch::grid& blocks
                  , const cuda::launch::grid& threads
                  , uint32_t memory = 0
                ) noexcept
                  : blocks {blocks}
                  , threads {threads}
                  , memory {memory}
                {}

                __host__ __device__ inline config& operator=(const config&) noexcept = default;
                __host__ __device__ inline config& operator=(config&&) noexcept = default;
            };

            /**
             * Creates a new grid representing the topology of a cube.
             * @param x The grid's size for cube side.
             * @return The cube grid instance.
             */
            __host__ __device__ inline constexpr grid grid::cube(dimension_type x) noexcept
            {
                return grid {x, x, x};
            }

            /**
             * Creates a new grid representing the topology of a square.
             * @param x The grid's size for square side.
             * @return The square grid instance.
             */
            __host__ __device__ inline constexpr grid grid::square(dimension_type x) noexcept
            {
                return grid {x, x, 1};
            }

            /**
             * Creates a new grid representing the topology of a line.
             * @param x The grid's line length.
             * @return The line grid instance.
             */
            __host__ __device__ inline constexpr grid grid::line(dimension_type x) noexcept
            {
                return grid {x, 1, 1};
            }

            /**
             * Creates a new grid representing the topology of a point.
             * @return The point grid instance.
             */
            __host__ __device__ inline constexpr grid grid::point() noexcept
            {
                return grid {1, 1, 1};
            }
        }

        namespace kernel
        {
          #if defined(MUSEQA_COMPILER_NVCC)
          #if MUSEQA_COMPILER_NVCC >= 900
            /**
             * Asynchronously launches a kernel function where threads from different
             * blocks can cooperate and synchronize as they execute.
             * @tparam P The kernel function's parameters' types.
             * @param kernel The kernel function to be launched into the device.
             * @param stream The stream in which the kernel must be enqueued to.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void cooperate(
                kernel::raw<P...> *kernel
              , cuda::stream::id stream
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                const auto f = reinterpret_cast<const void*>(kernel);
                void *bag[] = {const_cast<void*>(static_cast<const void*>(&args))..., nullptr};
                cuda::check(cudaLaunchCooperativeKernel(f, cfg.blocks, cfg.threads, bag, cfg.memory, stream));
            }

            /**
             * Asynchronously launches a kernel functor where threads from different
             * blocks can cooperate and synchronize as they execute.
             * @tparam P The kernel function's parameters' types.
             * @param functor The kernel functor to be launched into the device.
             * @param stream The stream in which the kernel must be enqueued to.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void cooperate(
                const kernel::functor<P...>& functor
              , cuda::stream::id stream
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                cuda::kernel::cooperate(*functor, stream, cfg, args...);
            }
          #endif

            /**
             * Enqueues the asynchronous execution of a kernel function on a stream
             * of the currently active device.
             * @tparam P The kernel function's parameters' types.
             * @param kernel The kernel function to be launched into the device.
             * @param stream The stream in which the kernel must be enqueued to.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void enqueue(
                kernel::raw<P...> kernel
              , cuda::stream::id stream
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                (kernel)<<<cfg.blocks, cfg.threads, cfg.memory, stream>>>(args...);
                cuda::check(cuda::error::clear());
            }

            /**
             * Enqueues the asynchronous execution of a kernel functor on a stream
             * of the currently active device.
             * @tparam P The kernel function's parameters' types.
             * @param functor The kernel functor to be launched into the device.
             * @param stream The stream in which the kernel must be enqueued to.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void enqueue(
                const kernel::functor<P...>& functor
              , cuda::stream::id stream
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                cuda::kernel::enqueue(*functor, stream, cfg, args...);
            }

            /**
             * Launches a kernel function for synchronous execution on the default
             * stream of the currently active device.
             * @tparam P The kernel function's parameters' types.
             * @param kernel The kernel function to be launched into the device.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void launch(
                kernel::raw<P...> *kernel
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                cuda::kernel::enqueue(kernel, stream::default_stream, cfg, args...);
            }

            /**
             * Launches a kernel functor for synchronous execution on the default
             * stream of the currently active device.
             * @tparam P The kernel function's parameters' types.
             * @param functor The kernel functor to be launched into the device.
             * @param cfg The launch configuration parameters.
             * @param args The kernel function's parameters.
             */
            template <typename ...P>
            inline void launch(
                const kernel::functor<P...>& functor
              , const cuda::launch::config& cfg
              , const P&... args
            ) noexcept(!safe)
            {
                cuda::kernel::launch(*functor, cfg, args...);
            }
          #endif
        }
    }
}

#endif

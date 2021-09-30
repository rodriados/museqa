/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Generic type memory allocator implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/utility.hpp>

namespace museqa
{
    namespace memory
    {
        /**
         * Describes and manages the memory allocation and deallocation routines
         * for generic variable types and memory kinds.
         * @since 1.0
         */
        class allocator
        {
          public:
            using allocator_type = void (*)(void**, size_t, size_t);
            using deallocator_type = void (*)(void*);

          protected:
            allocator_type m_allocator = nullptr;       /// The memory allocator functor.
            deallocator_type m_deallocator = nullptr;   /// The memory deallocator functor.

          public:
            __host__ __device__ inline constexpr allocator() noexcept = default;
            __host__ __device__ inline constexpr allocator(const allocator&) noexcept = default;
            __host__ __device__ inline constexpr allocator(allocator&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given functors.
             * @param allocf The allocator functor.
             * @param deallocf The deallocator functor.
             */
            __host__ __device__ inline constexpr allocator(allocator_type allocf, deallocator_type deallocf) noexcept
              : m_allocator {allocf}
              , m_deallocator {deallocf}
            {}

            /**
             * Instantiates a new allocator from given lambdas.
             * @tparam A The allocator functor lambda type.
             * @tparam D The deallocator functor lambda type.
             * @param allocf The allocator lambda.
             * @param deallocf The deallocator lambda.
             */
            template <typename A, typename D>
            __host__ __device__ inline constexpr allocator(const A& allocf, const D& deallocf) noexcept
              : allocator {allocator_type {allocf}, deallocator_type {deallocf}}
            {}

            /**
             * Instantiates a new allocator containing only a deallocator.
             * @tparam D The deallocator functor lambda type.
             * @param deallocf The deallocator lambda.
             */
            template <typename D>
            __host__ __device__ inline constexpr allocator(const D& deallocf) noexcept
              : allocator {nullptr, deallocator_type {deallocf}}
            {}

            __host__ __device__ inline allocator& operator=(const allocator&) noexcept = default;
            __host__ __device__ inline allocator& operator=(allocator&&) noexcept = default;

            /**
             * Invokes the allocator functor and allocates a given pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param ptr The target pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline auto allocate(T **ptr, size_t n = 1) const -> T*
            {
                using target_type = typename std::conditional<std::is_void<T>::value, char, T>::type;
                (m_allocator)(reinterpret_cast<void**>(ptr), sizeof(target_type), n);
                return *ptr;
            }

            /**
             * Invokes the allocator functor for a new memory pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline auto allocate(size_t n = 1) const -> T*
            {
                T *ptr;
                return allocate<T>(&ptr, n);
            }

            /**
             * Invokes the deallocator functor and frees a memory pointer.
             * @tparam T The type of pointer to deallocate memory from.
             * @param ptr The pointer of which memory must be freed.
             */
            template <typename T>
            __host__ __device__ inline void deallocate(T *ptr) const
            {
                (m_deallocator)(reinterpret_cast<void*>(ptr));
            }
        };
    }

    namespace factory
    {
        /**
         * Creates an allocator for the specified pointer type, which is not an
         * array and has its default constructor called.
         * @tparam T The type of pointer element to build.
         * @return The new allocator for given type.
         */
        template <typename T>
        __host__ __device__ inline auto allocator() noexcept
        -> typename std::enable_if<!std::is_array<T>::value, memory::allocator>::type
        {
            return memory::allocator {
                [](void **ptr, size_t, size_t) { *ptr = new pure<T>; }
              , [](void *ptr) { delete (reinterpret_cast<pure<T>*>(ptr)); }
            };
        }

        /**
         * Creates an allocator for the specified pointer type, which has its default
         * constructor called for each instance.
         * @tparam T The type of pointer element to build.
         * @return The new allocator for given type.
         */
        template <typename T>
        __host__ __device__ inline auto allocator() noexcept
        -> typename std::enable_if<std::is_array<T>::value, memory::allocator>::type
        {
            return memory::allocator {
                [](void **ptr, size_t, size_t n) { *ptr = new pure<T>[n]; }
              , [](void *ptr) { delete[] (reinterpret_cast<pure<T>*>(ptr)); }
            };
        }

        /**
         * Creates an allocator for a generic, untyped pointer. Due to its untypeness,
         * the allocator does not call any constructors on the allocated memory.
         * @return The new generic memory allocator.
         */
        template <>
        __host__ __device__ inline auto allocator<void>() noexcept -> memory::allocator
        {
            return memory::allocator {
                [](void **ptr, size_t, size_t n) { *ptr = operator new(n); }
              , [](void *ptr) { operator delete(ptr); }
            };
        }
    }
}

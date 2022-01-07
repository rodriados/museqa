/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Generic type memory allocator implementation.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include <museqa/environment.h>

#include <museqa/utility.hpp>
#include <museqa/utility/delegate.hpp>

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Describes and manages the memory allocation and deletion routines for pointers
     * of generic types and located in generic memory spaces.
     * @since 1.0
     */
    class allocator
    {
        public:
            using allocator_type = utility::delegate<void, void**, size_t, size_t>;
            using deleter_type = utility::delegate<void, void*>;

        protected:
            allocator_type m_allocator {};
            deleter_type m_deleter {};

        public:
            __host__ __device__ inline constexpr allocator() noexcept = default;
            __host__ __device__ inline constexpr allocator(const allocator&) noexcept = default;
            __host__ __device__ inline constexpr allocator(allocator&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given delegates.
             * @param falloc The allocation delegate.
             * @param fdelete The deletion delegate.
             */
            __host__ __device__ inline constexpr allocator(allocator_type falloc, deleter_type fdelete) noexcept
              : m_allocator {falloc}
              , m_deleter {fdelete}
            {}

            /**
             * Instantiates a new allocator from given lambdas.
             * @tparam A The allocation lambda type.
             * @tparam D The deletion lambda type.
             * @param falloc The allocation lambda.
             * @param fdelete The deletion lambda.
             */
            template <typename A, typename D>
            __host__ __device__ inline constexpr allocator(const A& falloc, const D& fdelete) noexcept
              : allocator {allocator_type(falloc), deleter_type(fdelete)}
            {}

            /**
             * Instantiates a new allocator containing only a deleter.
             * @tparam D The deletion lambda type.
             * @param fdelete The deletion lambda.
             */
            template <typename D>
            __host__ __device__ inline constexpr allocator(const D& fdelete) noexcept
              : allocator {nullptr, deleter_type(fdelete)}
            {}

            __host__ __device__ inline allocator& operator=(const allocator&) noexcept = default;
            __host__ __device__ inline allocator& operator=(allocator&&) noexcept = default;

            /**
             * Invokes the allocator and allocates the given pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param ptr The target pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline T *allocate(T **ptr, size_t n = 1) const
            {
                using target_type = typename std::conditional<std::is_void<T>::value, uint8_t, T>::type;
                (m_allocator)(reinterpret_cast<void**>(ptr), sizeof(target_type), n);
                return *ptr;
            }

            /**
             * Invokes the allocator for a new memory pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline T *allocate(size_t n = 1) const
            {
                T *ptr;
                return allocate<T>(&ptr, n);
            }

            /**
             * Invokes the deleter and frees a memory pointer.
             * @tparam T The type of pointer to free memory from.
             * @param ptr The pointer of which memory must be freed.
             */
            template <typename T = void>
            __host__ __device__ inline void deallocate(T *ptr) const
            {
                (m_deleter)(reinterpret_cast<void*>(ptr));
            }
    };
}

namespace factory::memory
{
    /**
     * Creates an allocator for pointers of the specified type, which must have
     * a public and well-formed default constructor.
     * @tparam T The type of pointer to allocate memory to.
     * @return An allocator for given type.
     */
    template <typename T>
    __host__ __device__ inline constexpr auto allocator() noexcept
    -> typename std::enable_if<
        std::is_default_constructible<T>::value
      , museqa::memory::allocator
    >::type
    {
        return museqa::memory::allocator {
            [](void **ptr, size_t, size_t n) { *ptr = new pure<T>[n]; }
          , [](void *ptr) { delete[] (reinterpret_cast<pure<T>*>(ptr)); }
        };
    }

    /**
     * Creates an allocator for a void pointer, effectively allocating the requested
     * amount of bytes in memory, without the initialization of any specific type.
     * @return A generic type-less memory allocator.
     */
    template <typename T = void>
    __host__ __device__ inline constexpr auto allocator() noexcept
    -> typename std::enable_if<
        std::is_void<T>::value
      , museqa::memory::allocator
    >::type
    {
        return museqa::memory::allocator {
            [](void **ptr, size_t size, size_t n) { *ptr = operator new (size * n); }
          , [](void *ptr) { operator delete(ptr); }
        };
    }
}

MUSEQA_END_NAMESPACE

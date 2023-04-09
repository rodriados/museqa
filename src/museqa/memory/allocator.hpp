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
    class allocator_t
    {
        public:
            using allocf_t = utility::delegate_t<void, void**, size_t, size_t>;
            using deletef_t = utility::delegate_t<void, void*>;

        protected:
            allocf_t m_alloc {};
            deletef_t m_delete {};

        public:
            __host__ __device__ inline constexpr allocator_t() noexcept = default;
            __host__ __device__ inline constexpr allocator_t(const allocator_t&) noexcept = default;
            __host__ __device__ inline constexpr allocator_t(allocator_t&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given delegates.
             * @param allocf The allocation delegate.
             * @param deletef The deletion delegate.
             */
            __host__ __device__ inline constexpr allocator_t(allocf_t allocf, deletef_t deletef) noexcept
              : m_alloc (allocf)
              , m_delete (deletef)
            {}

            /**
             * Instantiates a new allocator from given lambdas.
             * @tparam A The allocation lambda type.
             * @tparam D The deletion lambda type.
             * @param allocf The allocation lambda.
             * @param deletef The deletion lambda.
             */
            template <typename A, typename D>
            __host__ __device__ inline constexpr allocator_t(const A& allocf, const D& deletef) noexcept
              : allocator_t (allocf_t(allocf), deletef_t(deletef))
            {}

            /**
             * Instantiates a new allocator containing only a deleter.
             * @tparam D The deletion lambda type.
             * @param deletef The deletion lambda.
             */
            template <typename D>
            __host__ __device__ inline constexpr allocator_t(const D& deletef) noexcept
              : allocator_t (nullptr, deletef_t(deletef))
            {}

            __host__ __device__ inline allocator_t& operator=(const allocator_t&) noexcept = default;
            __host__ __device__ inline allocator_t& operator=(allocator_t&&) noexcept = default;

            /**
             * Invokes the allocator and allocates the given pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param ptr The target pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline T* allocate(T** ptr, size_t n = 1) const
            {
                using target_t = typename std::conditional<std::is_void<T>::value, uint8_t, T>::type;
                (m_alloc)(reinterpret_cast<void**>(ptr), sizeof(target_t), n);
                return *ptr;
            }

            /**
             * Invokes the allocator for a new memory pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            __host__ __device__ inline T* allocate(size_t n = 1) const
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
            __host__ __device__ inline void deallocate(T* ptr) const
            {
                (m_delete)(reinterpret_cast<void*>(ptr));
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
      , museqa::memory::allocator_t
    >::type
    {
        return museqa::memory::allocator_t {
            [](void **ptr, size_t, size_t n) { *ptr = new pure_t<T>[n]; }
          , [](void *ptr) { delete[] reinterpret_cast<pure_t<T>*>(ptr); }
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
      , museqa::memory::allocator_t
    >::type
    {
        return museqa::memory::allocator_t {
            [](void **ptr, size_t size, size_t n) { *ptr = operator new(size * n); }
          , [](void *ptr) { operator delete(ptr); }
        };
    }
}

MUSEQA_END_NAMESPACE

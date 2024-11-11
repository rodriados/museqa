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

MUSEQA_BEGIN_NAMESPACE

namespace memory
{
    /**
     * Reference to a memory region freeing routine for pointers of generic types
     * and located in any kind of memory space.
     * @since 1.0
     */
    class deleter_t
    {
        protected:
            using fdelete_t = void (*)(void*);

        protected:
            fdelete_t m_fdelete = nullptr;

        public:
            MUSEQA_CONSTEXPR deleter_t() noexcept = default;
            MUSEQA_CONSTEXPR deleter_t(const deleter_t&) noexcept = default;
            MUSEQA_CONSTEXPR deleter_t(deleter_t&&) noexcept = default;

            /**
             * Instantiates a new deleter with the given functor.
             * @param fdelete The deleter functor.
             */
            MUSEQA_CUDA_CONSTEXPR deleter_t(fdelete_t fdelete) noexcept
              : m_fdelete (fdelete)
            {}

            MUSEQA_CONSTEXPR deleter_t& operator=(const deleter_t&) noexcept = default;
            MUSEQA_CONSTEXPR deleter_t& operator=(deleter_t&&) noexcept = default;

            /**
             * Invokes the deleter and frees a memory pointer.
             * @tparam T The type of pointer to free memory from.
             * @param ptr The pointer of which memory must be freed.
             */
            template <typename T = void>
            MUSEQA_CUDA_INLINE void deallocate(T *ptr) const
            {
                utility::invoke(m_fdelete, static_cast<void*>(ptr));
            }

            /**
             * Checks whether the deleter is not bound to any functor.
             * @return Is the deleter empty?
             */
            MUSEQA_CUDA_CONSTEXPR bool empty() const noexcept
            {
                return m_fdelete == nullptr;
            }
    };

    /**
     * Reference to a memory region allocating and freeing routines for pointers
     * of generic types and located in any kind of memory space.
     * @since 1.0
     */
    class allocator_t : public deleter_t
    {
        protected:
            using falloca_t = void*(*)(size_t, size_t);
            using fdelete_t = deleter_t::fdelete_t;

        protected:
            falloca_t m_falloca = nullptr;

        public:
            MUSEQA_CONSTEXPR allocator_t() noexcept = default;
            MUSEQA_CONSTEXPR allocator_t(const allocator_t&) noexcept = default;
            MUSEQA_CONSTEXPR allocator_t(allocator_t&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given functors.
             * @param falloca The allocator functor.
             * @param fdelete The deleter functor.
             */
            MUSEQA_CUDA_CONSTEXPR allocator_t(falloca_t falloca, fdelete_t fdelete) noexcept
              : deleter_t (fdelete)
              , m_falloca (falloca)
            {}

            using deleter_t::deleter_t;

            MUSEQA_CONSTEXPR allocator_t& operator=(const allocator_t&) noexcept = default;
            MUSEQA_CONSTEXPR allocator_t& operator=(allocator_t&&) noexcept = default;

            /**
             * Allocates memory for a new memory pointer.
             * @tparam T The type of pointer to allocate memory to.
             * @param n The number of elements to allocate memory for.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            MUSEQA_CUDA_INLINE T *allocate(size_t n = 1) const
            {
                constexpr auto size = sizeof(std::conditional_t<std::is_void_v<T>, uint8_t, T>);
                return (T*) utility::invoke(m_falloca, size, n);
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
    MUSEQA_CUDA_CONSTEXPR auto allocator() noexcept
    -> std::enable_if_t<std::is_default_constructible_v<T>, museqa::memory::allocator_t>
    {
        return museqa::memory::allocator_t {
            [](size_t, size_t n) { return (void*) new pure_t<T>[n]; }
          , [](void *ptr) { delete[] reinterpret_cast<pure_t<T>*>(ptr); }
        };
    }

    /**
     * Creates an allocator for a void pointer, effectively allocating the requested
     * amount of bytes in memory, without the initialization of any specific type.
     * @return A generic type-less memory allocator.
     */
    template <typename T = void>
    MUSEQA_CUDA_CONSTEXPR auto allocator() noexcept
    -> std::enable_if_t<std::is_void_v<T>, museqa::memory::allocator_t>
    {
        return museqa::memory::allocator_t {
            [](size_t size, size_t n) { return operator new(size * n); }
          , [](void *ptr) { operator delete(ptr); }
        };
    }
}

MUSEQA_END_NAMESPACE

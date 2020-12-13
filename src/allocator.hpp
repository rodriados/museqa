/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file Implements the base for generic type allocators.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2019-present Rodrigo Siqueira
 */
#pragma once

#include <utility>

#include "utils.hpp"
#include "functor.hpp"

namespace museqa
{
    /**
     * Describes the allocation and deallocation routines for a given type.
     * @tparam T The pointer's element type.
     * @since 0.1.1
     */
    class allocator
    {
        public:
            using ptr_type = void *;                                    /// The type of pointer to allocate.

            using up_type = functor<void(ptr_type *, size_t, size_t)>;  /// The allocator's functor type.
            using down_type = functor<void(ptr_type)>;                  /// The deallocator's functor type.

        protected:
            up_type m_up;                                               /// The allocator's up functor.
            down_type m_down;                                           /// The allocator's down functor.

        public:
            inline constexpr allocator() noexcept = default;
            inline constexpr allocator(const allocator&) noexcept = default;
            inline constexpr allocator(allocator&&) noexcept = default;

            /**
             * Instantiates a new allocator with the given functors.
             * @param up_functor The allocator functor.
             * @param down_functor The deallocator functor.
             */
            inline constexpr allocator(up_type up_functor, down_type down_functor) noexcept
            :   m_up {up_functor}
            ,   m_down {down_functor}
            {}

            /**
             * Instantiates a new allocator from given lambdas.
             * @tparam A The allocator functor lambda type.
             * @tparam D The deallocator functor lambda type.
             * @param up_lambda The allocator lambda.
             * @param down_lambda The deallocator lambda.
             */
            template <typename A, typename D>
            inline constexpr allocator(A up_lambda, D down_lambda) noexcept
            :   allocator {up_type(up_lambda), down_type(down_lambda)}
            {}

            inline allocator& operator=(const allocator&) noexcept = default;
            inline allocator& operator=(allocator&&) noexcept = default;

            /**
             * Invokes the allocator functor and creates a new allocated pointer.
             * @param ptr The target pointer to allocated memory to.
             * @param n The number of elements to allocate memory to.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            inline auto allocate(T **ptr, size_t n) const -> T *
            {
                using type = typename std::conditional<std::is_void<T>::value, char, T>::type;
                m_up.operator()(reinterpret_cast<ptr_type *>(ptr), sizeof(type), n);
                return *ptr;
            }

            /**
             * Invokes the allocator functor and creates a new allocated pointer.
             * @param n The number of elements to allocate memory to.
             * @return The newly allocated pointer.
             */
            template <typename T = void>
            inline auto allocate(size_t n) const -> T *
            {
                T *ptr;
                return allocate(&ptr, n);
            }

            /**
             * Invokes the deallocator functor and frees the pointer's memory.
             * @param ptr The pointer of which memory must be freed.
             */
            inline void deallocate(ptr_type ptr) const
            {
                m_down.operator()(ptr);
            }

            /**
             * Creates a builtin allocator for a specified pointer type, which
             * has its default contructor called for each instance.
             * @tparam T The type of pointer element to build.
             * @return The new allocator for given type.
             */
            template <typename T>
            inline static auto builtin() -> allocator
            {
                return {
                    !std::is_array<T>::value
                        ? [](void **ptr, size_t, size_t n) { *ptr = new pure<T>; }
                        : [](void **ptr, size_t, size_t n) { *ptr = new pure<T> [n]; }
                ,   !std::is_array<T>::value
                        ? [](void *ptr) { delete (static_cast<pure<T> *>(ptr)); }
                        : [](void *ptr) { delete[] (static_cast<pure<T> *>(ptr)); }
                };
            }
    };
}
